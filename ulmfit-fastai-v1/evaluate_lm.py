import json
from math import exp
import os
import pickle

from fastai.text import *
import fire

from utils import lm_learner, evaluate_perplexity


def evaluate_lm(data_path, model_dir, custom_pp=False):
    """
    Evaluate metrics of a trained LM using any dataset of texts in csv.

    The perplexity estimated as e^(avg. loss), but the average loss changes slightly with batch
    size. To get perplexity computed in slower but controlled fashion, set `custom_pp` to True.
    The discrepancy between perplexity and custom perplexity is empirically approximately 1%.
    """
    model_dir = Path(model_dir)
    with open(model_dir / "lm_itos.pkl", "rb") as f:
        itos = pickle.load(f)

    data_df = pd.read_csv(data_path, header=None)
    data = TextLMDataBunch.from_df(
        "", data_df, data_df, text_cols=0, tokenizer=Tokenizer(lang="xx"),
        vocab=Vocab(itos))

    with open(model_dir / "model_hparams.json", "r") as model_hparams_file:
        model_hparams = json.load(model_hparams_file)
    learner = lm_learner(data, AWD_LSTM, model_dir, pretrained=True, config=model_hparams)

    loss, acc = learner.validate()
    print("Loss: {}, Perplexity: {}, Accuracy: {}".format(loss, exp(loss), acc))
    if custom_pp:
        print("Custom perplexity: {}, Fraction OOV: {}, PPwp: {}".format(
            *evaluate_perplexity(learner, data.valid_ds.x)))

if __name__ == '__main__':
    fire.Fire(evaluate_lm)
