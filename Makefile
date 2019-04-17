LANG := sk

.DEFAULT: all
.PRECIOUS: dataset/$(LANG)wiki-latest-pages-articles.xml

all: dataset/train.csv dataset/val.csv

dataset/$(LANG)wiki-latest-pages-articles.xml:
	curl https://dumps.wikimedia.org/$(LANG)wiki/latest/$(LANG)wiki-latest-pages-articles.xml.bz2 | \
	bunzip2 -c > $@

dataset/train.csv dataset/val.csv: dataset/$(LANG)wiki-latest-pages-articles.xml
	tmp=$$(mktemp -d) \
	&& vendor/wikiextractor/WikiExtractor.py --json -o "$${tmp}" $^ \
	&& utils/prepare-training-dataset.py -i "$${tmp}" --output-train dataset/train.csv \
																								    --output-val dataset/val.csv \
	&& rm -rf "$${tmp}"
