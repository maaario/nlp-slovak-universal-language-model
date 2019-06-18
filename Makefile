LANG := sk

.DEFAULT: all
.PRECIOUS: dataset/$(LANG)wiki-latest-pages-articles.xml

all: dataset/train.csv dataset/val.csv dataset/featured-train.csv dataset/featured-val.csv

dataset/$(LANG)wiki-latest-pages-articles.xml:
	mkdir -p dataset
	curl https://dumps.wikimedia.org/$(LANG)wiki/latest/$(LANG)wiki-latest-pages-articles.xml.bz2 | \
	bunzip2 -c > $@

dataset/extracted-wiki.stamp: dataset/$(LANG)wiki-latest-pages-articles.xml
	mkdir -p dataset/extracted-wiki
	vendor/wikiextractor/WikiExtractor.py --json -o "dataset/extracted-wiki" $^
	touch $@

dataset/featured-articles.txt: dataset/$(LANG)wiki-latest-pages-articles.xml
	cat $^ | utils/extract_perfect.py > $@

dataset/featured-wiki.stamp: dataset/extracted-wiki.stamp dataset/featured-articles.txt
	mkdir -p dataset/featured-wiki
	utils/filter_articles.py dataset/extracted-wiki dataset/featured-wiki \
		--title-whitelist dataset/featured-articles.txt
	touch $@

dataset/train.csv dataset/val.csv: dataset/extracted-wiki.stamp
	utils/prepare-training-dataset.py -i dataset/extracted-wiki --output-train dataset/train.csv \
		--output-val dataset/val.csv

dataset/featured-train.csv dataset/featured-val.csv: dataset/featured-wiki.stamp
	utils/prepare-training-dataset.py -i dataset/featured-wiki \
		--output-train dataset/featured-train.csv --output-val dataset/featured-val.csv
