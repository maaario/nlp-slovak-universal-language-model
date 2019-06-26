from pathlib import Path

from fastai.text import TextLMDataBunch, Tokenizer
import fire

from utils import csv_to_train_valid_df


def prepare_lm_dataset(input_path, output_dir=None, valid_split=0.2, tokenizer_lang="xx",
                       min_freq=2, seed=42):
    """
    Reads CSV file with texts for training language model, splits it into training and validation
    sets, tokenizes and saves the dataset.

    Attributes:
        input_path (str): Path to CSV file where there are texts in the first column.
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

    data_lm = TextLMDataBunch.from_df(
        output_dir, train_df, valid_df, text_cols=0, tokenizer=Tokenizer(lang=tokenizer_lang),
        min_freq=min_freq)
    data_lm.save("data_lm.pkl")

    with open(output_dir / "data_lm_tokenized_train.txt", "w") as f:
        f.write("\n".join(map(str, list(data_lm.train_ds.x))))
    with open(output_dir / "data_lm_tokenized_valid.txt", "w") as f:
        f.write("\n".join(map(str, list(data_lm.valid_ds.x))))


if __name__ == "__main__":
    fire.Fire(prepare_lm_dataset)
