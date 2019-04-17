LANG := "sk"

.PRECIOUS: dataset/$(LANG)wiki-latest-pages-articles.xml

dataset/%wiki-latest-pages-articles.xml:
	curl https://dumps.wikimedia.org/$*wiki/latest/$*wiki-latest-pages-articles.xml.bz2 | bunzip2 -c > $@

dataset/extracted-%wiki: dataset/%wiki-latest-pages-articles.xml
	mkdir -p $@
	vendor/wikiextractor/WikiExtractor.py --json -o $@ $^

all: dataset/extracted-$(LANG)wiki
