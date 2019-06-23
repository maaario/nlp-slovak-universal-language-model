import json
from math import exp
import os
import pickle

from fastai.text import *
import fire
import torch

from utils import lm_learner


def train_lm(data_dir, model_dir, epochs=12, lr=3e-4, pretrained=False):
    """
    Trains new language model (or continues in training if `pretrained is set`) using
    the provided dataset.
    """
    data_lm = load_data(data_dir, 'data_lm.pkl')

    model_hparams = dict(
        emb_sz=100, n_hid=256, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.1,
        hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)

    learner = lm_learner(
        data_lm, AWD_LSTM, model_dir, pretrained=pretrained, config=model_hparams)
    learner.fit(epochs, lr)

    loss, acc = learner.validate(learner.data.train_dl)
    print("Training - Loss: {}, Perplexity: {}, Accuracy: {}".format(loss, exp(loss), acc))
    loss, acc = learner.validate()
    print("Validation - Loss: {}, Perplexity: {}, Accuracy: {}".format(loss, exp(loss), acc))

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_dir = Path(model_dir)
    torch.save(learner.model.state_dict(), model_dir / "lm_wgts.pth")
    with open(model_dir / "lm_itos.pkl", "wb") as itos_file:
        pickle.dump(learner.data.vocab.itos, itos_file)
    with open(model_dir / "model_hparams.json", "w") as model_hparams_file:
        json.dump(model_hparams, model_hparams_file, indent=2)

    # TODO: store train / val metrics


if __name__ == "__main__":
    fire.Fire(train_lm)