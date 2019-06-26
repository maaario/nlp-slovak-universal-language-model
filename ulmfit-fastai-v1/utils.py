from math import log10
from pathlib import Path

from fastai.text import get_language_model, get_text_classifier, LanguageLearner
from fastai.text.learner import _model_meta
import numpy as np
import pandas as pd
from tqdm import tqdm


def csv_to_train_valid_df(csv_path, valid_split, seed=42):
    """
    Reads csv file and splits records to training and validation pandas dataframes.
    """
    gen = np.random.RandomState(seed=seed)

    df = pd.read_csv(csv_path, header=None)

    df = df.iloc[gen.permutation(len(df))]
    cut = int(valid_split * len(df)) + 1
    train_df, valid_df = df[cut:], df[:cut]

    return train_df, valid_df


def evaluate_perplexity(learner, text_list):
    """
    Evaluates perplexity of a model using model learner and a list of texts.
    """
    num_words = 0
    log_prob = 0
    num_unk = 0
    log_prob_unk = 0

    for text in tqdm(text_list):
        learner.model.reset()
        word_tensor, y = learner.data.one_item("xxbos")

        for word in str(text).split()[1:]:
            idx = learner.data.vocab.stoi[word]
            predicted_probs = learner.pred_batch(batch=(word_tensor, y))[0][-1]

            log_prob += log10(predicted_probs[idx])
            num_words += 1

            if learner.data.vocab.itos[idx] == "xxunk":
                num_unk += 1
                log_prob_unk += log10(predicted_probs[idx])

            word_tensor = word_tensor.new_tensor([idx])[None]

    perplexity = 10 ** (- log_prob / num_words)

    # Contribution of OOV words to perplexity - to compare with irstlm's PPwp (compile-lm.cpp:406)
    # This implementation is a bit incorrect since the final number of words in PPwp should be
    # num_words - num_unk, but to enable comparison, we stick to original implementation.
    perplexity_oov = perplexity * (1 - 10 ** (log_prob_unk / num_words))

    return perplexity, num_unk / num_words, perplexity_oov


def lm_learner(data, arch, model_dir, config=None, drop_mult=1., pretrained=True,
               **learn_kwargs):
    """
    Set up a model - either a new untrained instance or a pretrained model.

    Simplified version of fastai's language_model_learner, where no models are downloaded
    and where the data location is decoupled from model location.
    """
    model = get_language_model(arch, len(data.vocab.itos), config=config, drop_mult=drop_mult)
    meta = _model_meta[arch]
    learner = LanguageLearner(data, model, split_func=meta["split_lm"], **learn_kwargs)
    learner.path = Path(model_dir)
    learner.model_dir = Path()
    if pretrained:
        learner.load_pretrained(
            learner.path / "lm_wgts.pth",
            learner.path / "lm_itos.pkl")
        learner.freeze()
    return learner


def clas_learner(data, arch, model_dir, config=None, drop_mult=1., pretrained=0,
                 **learn_kwargs):
    """
    Constructs a classifier for `data`, and binds it to the folder `model_dir`.
    Returns the resulting learner.

    If `pretrained` is 1, a finetuned language model is loaded from `model_dir` - use this option
    when training a new classifier.
    If `pretrained` is 2, an entire classifier is loaded from `model_dir`.
    Otherwise, an uninitialized encoder is used.
    """
    model_dir = Path(model_dir)

    num_classes = len(set(data.label_list.y))
    model = get_text_classifier(
        arch, len(data.vocab.itos), num_classes, config=config, drop_mult=drop_mult)
    meta = _model_meta[arch]
    learner = LanguageLearner(data, model, split_func=meta['split_clas'], **learn_kwargs)
    learner.path = model_dir
    learner.model_dir = Path()

    if pretrained > 0:
        if pretrained == 1:
            learner.load_encoder("lm_finetuned_encoder")
        elif pretrained == 2:
            learner.load_pretrained(
                learner.path / "clas_wgts.pth",
                learner.path / "clas_itos.pkl")
        learner.freeze()

    return learner
