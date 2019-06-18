#!/usr/bin/env python3
# This is specific for Slovak wikipedia. It outputs names of articles that contain
# {{Perfektny clanok}} (featured articles)

import sys
import xml.etree.ElementTree as ET

FEATURED_MARKER = "{{Perfektný článok}}"


def main():
    tree = ET.iterparse(sys.stdin, events=["start", "end"])
    page_name = None
    for event, elem in tree:
        if event == 'start' and elem.tag == '{http://www.mediawiki.org/xml/export-0.10/}page':
            page_name = None
        if event == 'end' and elem.tag == '{http://www.mediawiki.org/xml/export-0.10/}title':
            page_name = elem.text
        if event == 'end' and elem.tag == '{http://www.mediawiki.org/xml/export-0.10/}text':
            if elem.text and FEATURED_MARKER in elem.text and page_name is not None:
                print(page_name)


if __name__ == '__main__':
    sys.exit(main())
