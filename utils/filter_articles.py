#!/usr/bin/env python3

import argparse
import dataclasses
import logging
import sys
from typing import Callable
from pathlib import Path
import re
import os
import json

from toolz import juxt, curry

logging.basicConfig(format='[%(asctime)s] [%(name)s]\t%(levelname)s:\t %(message)s',
                    level=logging.DEBUG)


@dataclasses.dataclass
class Filter:
    name: str
    filter: Callable[..., bool]


def handle_args():
    parser = argparse.ArgumentParser(description="Filter articles from WikiExtractor based on some "
                                                 "basic criteria.")
    parser.add_argument('input_dir', type=Path, help="Folder containing outputs of "
                                                     "WikiExtractor.py")
    parser.add_argument('output_dir', type=Path, help="Articles matching all filters will be written to this dir, "
                                                      "mirroring the layout of input_dir")
    parser.add_argument('--title-regex-whitelist', type=Path,
                        help="List of regexes. If specified, article is included if its title "
                             "matches at least one of these regexes.")
    parser.add_argument('--title-whitelist', type=Path, help="List of allowed article titles.")
    return parser.parse_args()


@curry
def filter_titles_regex_whitelist(whitelist_regexes, article):
    return any(juxt([x.match for x in whitelist_regexes])(article['title']))


@curry
def filter_titles_whitelist(whitelist, article):
    for title in whitelist:
        if article['title'] == title:
            return True
    return False


def run_filters(filters, inpath, outpath, file):
    logging.debug(f"Processing file {file}")
    os.makedirs(os.path.join(outpath, os.path.dirname(file)), exist_ok=True)
    with open(os.path.join(inpath, file), 'r') as fin:
        with open(os.path.join(outpath, file), 'w') as fout:
            for article_json in fin:
                article = json.loads(article_json)

                filter_results = juxt([f.filter for f in filters])(article)

                if all(filter_results):
                    logging.info(f'Including article "{article["title"]}"')
                    fout.writelines([article_json])
                else:
                    failed_filters = [filters[i].name for i, filter_ok in enumerate(filter_results) if not filter_ok]
                    logging.info(f'Excluding article "{article["title"]}". Filters [{", ".join(failed_filters)}] '
                                 f'failed')


def main():
    args = handle_args()

    filters = []

    if args.title_regex_whitelist:
        with open(args.title_regex_whitelist, 'r') as title_regex_whitelist_file:
            whitelist_filter = filter_titles_regex_whitelist(
                [re.compile(x.strip()) for x in title_regex_whitelist_file.readlines()]
            )
        filters.append(Filter('title-regex-whitelist', whitelist_filter))

    if args.title_whitelist:
        with open(args.title_whitelist, 'r') as title_whitelist_file:
            whitelist_filter = filter_titles_whitelist([x.strip() for x in title_whitelist_file.readlines()])
        filters.append(Filter('title-whitelist', whitelist_filter))

    wikifiles = []
    for dirpath, dirnames, filenames in os.walk(args.input_dir):
        for filename in filenames:
            file = os.path.join(dirpath, filename)
            wikifiles.append(os.path.relpath(file, args.input_dir))

    for file in wikifiles:
        run_filters(filters, args.input_dir, args.output_dir, file)


if __name__ == '__main__':
    sys.exit(main())
