import json
from math import exp
from pathlib import Path
import pickle

from fastai.text import AWD_LSTM, TextLMDataBunch, Tokenizer, Vocab
import fire
import pandas as pd

from utils import lm_learner, evaluate_perplexity


def evaluate_lm(data_path, model_dir, tokenizer_lang="xx", evaluate_custom_perplexity=False):
    """
    Evaluate metrics of a trained language model using any dataset of texts from CSV file.

    Attributes:
        data_path (str): Path to CSV file with texts in the first column.
        model_dir (str): Directory with a trained language model.
        tokenizer_lang (str): Language setting for tokenizer.
        evaluate_custom_perplexity (bool): The perplexity estimated as e^(avg. loss),
            but the average loss changes slightly with batch size. To get perplexity computed in
            slower but controlled fashion, set `evaluate_custom_perplexity` to True. Discrepancy
            between perplexity and custom perplexity is empirically approximately 1%.
    """
    model_dir = Path(model_dir)
    with open(model_dir / "lm_itos.pkl", "rb") as f:
        itos = pickle.load(f)

    data_df = pd.read_csv(data_path, header=None)
    data = TextLMDataBunch.from_df(
        "", data_df, data_df, text_cols=0, tokenizer=Tokenizer(lang=tokenizer_lang),
        vocab=Vocab(itos))

    with open(model_dir / "model_hparams.json", "r") as model_hparams_file:
        model_hparams = json.load(model_hparams_file)
    learner = lm_learner(data, AWD_LSTM, model_dir, pretrained=True, config=model_hparams)

    loss, acc = learner.validate()
    print("Loss: {}, Perplexity: {}, Accuracy: {}".format(loss, exp(loss), acc))
    if evaluate_custom_perplexity:
        print("Custom perplexity: {}, Fraction OOV: {}, OOV perplexity contribution: {}".format(
            *evaluate_perplexity(learner, data.valid_ds.x)))

if __name__ == "__main__":
    fire.Fire(evaluate_lm)
