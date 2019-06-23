import os

from fastai.text import *
import fire

from utils import csv_to_train_valid_df


def prepare_clas_dataset(input_path, output_dir=None, valid_split=0.2, min_freq=2, seed=42):
    """
    Reads .csv file with texts in first column and labels in second column.
    Splits it into training and validation sets, tokenizes and saves datasets for fine-tuning and
    for classification.
    """
    output_dir = (output_dir or os.path.dirname(input_path))
    train_df, valid_df = csv_to_train_valid_df(input_path, valid_split, seed)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = Path(output_dir)

    data_finetune_lm = TextLMDataBunch.from_df(
        output_dir, train_df, valid_df, tokenizer=Tokenizer(lang="xx"), text_cols=0,
        min_freq=min_freq)
    data_clas = TextClasDataBunch.from_df(
        output_dir, train_df, valid_df, tokenizer=Tokenizer(lang="xx"), text_cols=0, label_cols=1,
        vocab=data_finetune_lm.train_ds.vocab, bs=32, min_freq=min_freq)

    data_finetune_lm.save('data_finetune_lm.pkl')
    data_clas.save('data_clas.pkl')

    # TODO: if we want e.g. for fastext, also store .txt versions of tokenized texts


if __name__ == '__main__':
    fire.Fire(prepare_clas_dataset)