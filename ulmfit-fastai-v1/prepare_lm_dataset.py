import os

from fastai.text import *
import fire

from utils import csv_to_train_valid_df


def prepare_lm_dataset(input_path, output_dir=None, valid_split=0.2, min_freq=2, seed=42):
    """
    Reads .csv file with texts for training LM model, splits it into training and validation sets,
    tokenizes and saves the dataset.
    """
    output_dir = (output_dir or os.path.dirname(input_path))
    train_df, valid_df = csv_to_train_valid_df(input_path, valid_split, seed)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = Path(output_dir)

    data_lm = TextLMDataBunch.from_df(
        output_dir, train_df, valid_df, text_cols=0, tokenizer=Tokenizer(lang="xx"),
        min_freq=min_freq)
    data_lm.save('data_lm.pkl')

    with open(output_dir / "data_lm_tokenized_train.txt", "w") as f:
        f.write("\n".join(map(str, list(data_lm.train_ds.x))))
    with open(output_dir / "data_lm_tokenized_valid.txt", "w") as f:
        f.write("\n".join(map(str, list(data_lm.valid_ds.x))))


if __name__ == '__main__':
    fire.Fire(prepare_lm_dataset)
