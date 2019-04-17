#!/usr/bin/env python3
"""
Merges output files from WikiExtractor, and prepares them to a format that can be further processed
by other tools.

This script was taken from
https://github.com/fastai/fastai/blob/master/courses/dl2/imdb_scripts/merge_wiki.py

Original license:
Apache License, Version 2.0 Apache License Version 2.0, January 2004 http://www.apache.org/licenses/
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import sys

logging.basicConfig(format='[%(asctime)s]\t%(levelname)s:\t %(message)s',
                    level=logging.DEBUG)


def get_texts(root):
    for dir_ in root.iterdir():
        for wiki_file in dir_.iterdir():
            with open(wiki_file, encoding='utf-8') as f_in:
                for line in f_in:
                    article = json.loads(line)
                    text = article['text']
                    yield text


def main(args):

    if not args.input.exists():
        logging.critical(f"{input_path} does not exist!")
        return 1
    text_iter = get_texts(args.input)

    logging.info(f"Writing training set to {args.output_train}, and validation set to "
                 f"{args.output_val}")

    train_toks, val_toks, total_num_tokens = 0, 0, 0
    with open(args.output_train, 'w', encoding='utf-8') as f_train:
        with open(args.output_val, 'w', encoding='utf-8') as f_val:
            for i, text in enumerate(text_iter):
                new_toks = len(text.split())
                total_num_tokens += new_toks

                if i % 100 < args.split:
                    target = f_val
                    val_toks += new_toks
                else:
                    target = f_train
                    train_toks += new_toks

                target.write(text)

                if i % 10000 == 0:
                    logging.info(f"Processed {i} documents, {total_num_tokens} tokens.")

    logging.info(f"Processed {train_toks} tokens in training set, {val_toks} in validation set.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=Path,
                        help='the directory where the Wikipedia data extracted with '
                             'WikiExtractor.py is located. Consists of directories AA, AB, AC, '
                             'etc.')
    parser.add_argument('--output-train', type=Path, default='train.csv',
                        help='Output file for the training dataset')
    parser.add_argument('--output-val', type=Path, default='val.csv',
                        help='Output file for the validation dataset')
    parser.add_argument('-s', '--split', default=10, type=int, choices=range(0,101),
                        help='percentual split between training and validation dataset sizes')
    args = parser.parse_args()
    sys.exit(main(args))
