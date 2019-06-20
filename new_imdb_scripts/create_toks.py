from fastai.text import *
import html
import fire
import spacy

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

re1 = re.compile(r'  +')


def fixup(x):
    """Replaces character codes with the actual characters."""
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df, n_lbls, lang='en'):
    """
    <df>: data frame row
    <n_lbls>: consider only the first n_lbls columns of the frame
    <lang>: tokenizer for which language?
    """
    if len(df.columns) == 1:
        labels = []
        texts = f'\n{BOS} {FLD} 1 ' + df[0].astype(str)
    else:
        labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
        texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
        for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls+1} ' + df[i].astype(str)
    
    # fixup the words and convert: dataframe -> list_of_words
    texts = list(texts.apply(fixup).values)
    
    tok = Tokenizer(lang=lang).process_all(partition_by_cores(texts, num_cpus()))
    return tok, list(labels)


def get_all(df, n_lbls, lang='en', num_chunks=1):
    """
    <df>: data frame
    <n_lbls>: consider only the first n_lbls columns of the frame
    <lang>: language of the tokenizer
    """
    tok, labels = [], []
    for i, r in enumerate(df):  # for each chunk of data
        if num_chunks >= 0 and i >= num_chunks:
            break
        print(i)
        tok_, labels_ = get_texts(r, n_lbls, lang=lang)
        tok += tok_
        labels += labels_
    return tok, labels


def create_toks(dir_path, chunksize=24000, n_lbls=1, lang='en', max_total=None):
    """
    Create tokenized text from location <dir_path>.
    <chunksize> determines how large chunks of data should be loaded in
        memory at any single time.
    <n_lbls>: consider only the first n_lbls labels (columns) of the data
        files (in `csv` format).
    <lang>: language to be used by the tokenizer
    <max_total>: process only the first max_total entries of the table
    """
    num_chunks = ((max_total // chunksize) if max_total is not None else -1)
    print(f'dir_path {dir_path} chunksize {chunksize} n_lbls {n_lbls} lang {lang} max_total {max_total}')
    try:
        spacy.load(lang)
    except OSError:
        # TODO handle tokenization of Chinese, Japanese, Korean
        print(f'spacy tokenization model is not installed for {lang}.')
        lang = lang if lang in ['en', 'de', 'es', 'pt', 'fr', 'it', 'nl'] else 'xx'
        print(f'Command: python -m spacy download {lang}')
        sys.exit(1)
    
    dir_path = Path(dir_path)
    assert dir_path.exists(), f'Error: {dir_path} does not exist.'
    
    # Load the csv files into pandas dataframes. Each cell should be one
    # article, articles should be stored in the last column, and the previous
    # columns should be labels (values associated with the article). Tokenize
    # to obtain a list of list of tokens (1 article ==> list of tokens) and
    # labels.
    df_trn = pd.read_csv(dir_path / 'train.csv', header=None, chunksize=chunksize)
    df_val = pd.read_csv(dir_path / 'val.csv', header=None, chunksize=chunksize)
    tok_trn, trn_labels = get_all(df_trn, n_lbls, lang=lang, num_chunks=num_chunks)
    tok_val, val_labels = get_all(df_val, n_lbls, lang=lang, num_chunks=num_chunks)
    
    # Store both in 'tmp' folder as numpy objects (dtype = string...?).
    tmp_path = dir_path / 'tmp'
    tmp_path.mkdir(exist_ok=True)
    np.save(tmp_path / 'tok_trn.npy', tok_trn)
    np.save(tmp_path / 'tok_val.npy', tok_val)
    np.save(tmp_path / 'lbl_trn.npy', trn_labels)
    np.save(tmp_path / 'lbl_val.npy', val_labels)
    
    trn_joined = [' '.join(o) for o in tok_trn]
    open(tmp_path / 'joined.txt', 'w', encoding='utf-8').writelines(trn_joined)


if __name__ == '__main__': fire.Fire(create_toks)
