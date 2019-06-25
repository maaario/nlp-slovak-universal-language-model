from pathlib import Path

from fastai.text import TextLMDataBunch, TextClasDataBunch, Tokenizer
import fire
import pandas as pd

from utils import csv_to_train_valid_df


def prepare_clas_dataset(input_path, output_dir=None, valid_split=0.2, tokenizer_lang="xx",
                         min_freq=2, seed=42):
    """
    Reads a CSV file with texts and labels, splits it into training and validation sets,
    tokenizes texts and saves datasets for fine-tuning and for classification.

    Attributes:
        input_path (str): Path to CSV file with texts in the first and labels in second column.
        output_dir (str): Folder where to store the processed dataset.
        valid_split (float): A fraction of data used for validation.
        tokenizer_lang (str): Language setting for tokenizer.
        min_freq (int): Minimal number of occurrences of a word to be conidered for adding to
            vocabulary.
        seed (int): Random seed that determines the training-validation split.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir or input_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df = csv_to_train_valid_df(input_path, valid_split, seed)

    data_finetune_lm = TextLMDataBunch.from_df(
        output_dir, train_df, valid_df, tokenizer=Tokenizer(lang=tokenizer_lang), text_cols=0,
        min_freq=min_freq)
    data_clas = TextClasDataBunch.from_df(
        output_dir, train_df, valid_df, tokenizer=Tokenizer(lang=tokenizer_lang), text_cols=0,
        label_cols=1, vocab=data_finetune_lm.train_ds.vocab, bs=32, min_freq=min_freq)

    data_finetune_lm.save("data_finetune_lm.pkl")
    data_clas.save("data_clas.pkl")

    # TODO: if we want e.g. for fastext, also store .txt versions of tokenized texts


if __name__ == "__main__":
    fire.Fire(prepare_clas_dataset)
