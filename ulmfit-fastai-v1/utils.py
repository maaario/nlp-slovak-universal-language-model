from math import log10

from fastai.text import *
from fastai.text.learner import _model_meta
import numpy as np
import pandas as pd
from tqdm import tqdm


def csv_to_train_valid_df(csv_path, valid_split):
    """
    Reads csv file and splits records to training and validation pandas dataframes.
    """
    df = pd.read_csv(csv_path, header=None)

    df = df.iloc[np.random.permutation(len(df))]
    cut = int(valid_split * len(df)) + 1
    train_df, valid_df = df[cut:], df[:cut]

    return train_df, valid_df


def evaluate_perplexity(learner, text):
    """
    Evaluates perplexity of a model using model learner and text string.
    """
    # TODO: paralellize / batch perplexity computation
    learner.model.reset()
    num_words = 0
    log_prob = 0

    word_tensor, y = learner.data.one_item("")

    for word in tqdm(text.split()):
        idx = learner.data.vocab.stoi[word]
        predicted_probs = learner.pred_batch(batch=(word_tensor, y))[0][-1]
        log_prob += log10(predicted_probs[idx])
        num_words += 1
        word_tensor = word_tensor.new_tensor([idx])[None]

    return 10 ** (- log_prob / num_words)


def lm_learner(data, arch, model_dir, config=None, drop_mult=1., pretrained=True,
               **learn_kwargs):
    """
    Set up a model - either a new untrained instance or a pretrained model.

    Simplified version of fastai's language_model_learner, where no models are downloaded
    and where the data location is decoupled from model location.
    """
    model = get_language_model(arch, len(data.vocab.itos), config=config, drop_mult=drop_mult)
    meta = _model_meta[arch]
    learner = LanguageLearner(data, model, split_func=meta['split_lm'], **learn_kwargs)
    learner.path = Path(model_dir)
    if pretrained:
        learner.load_pretrained(
            learner.path / "lm_wgts.pth",
            learner.path / "lm_itos.pkl")
        learner.freeze()
    return learner
