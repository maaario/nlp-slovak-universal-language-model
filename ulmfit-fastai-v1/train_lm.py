import json
from math import exp
from pathlib import Path
import pickle

from fastai.text import AWD_LSTM, load_data
from fastai.text.models import awd_lstm_lm_config
import fire
import torch

from utils import lm_learner


def train_lm(data_dir, model_dir, epochs=12, lr=3e-4, pretrained=False, hparam_updates=dict()):
    """
    Trains a new language model using the provided dataset.

    Attributes:
        data_dir (str): A directory with processed training and validation data.
        model_dir (str): A directory where to store the trained model.
        epochs (int): Number of epochs for model training.
        lr (float): Learning rate.
        pretrained (bool): If `pretrained` is set, a trained model is first loaded from `model_dir`
            and then it is trained with the provided dataset.
        hparam_updates (dict): A dictionary with updates of model hyper-parametrs. By default,
            a default configuration of fastai's model is used.
    """
    data_lm = load_data(data_dir, "data_lm.pkl")

    model_hparams = awd_lstm_lm_config.update(hparam_updates)
    learner = lm_learner(
        data_lm, AWD_LSTM, model_dir, pretrained=pretrained, config=model_hparams)
    learner.fit(epochs, lr)

    loss, acc = learner.validate(learner.data.train_dl)
    print("Training - Loss: {}, Perplexity: {}, Accuracy: {}".format(loss, exp(loss), acc))
    loss, acc = learner.validate()
    print("Validation - Loss: {}, Perplexity: {}, Accuracy: {}".format(loss, exp(loss), acc))

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(learner.model.state_dict(), model_dir / "lm_wgts.pth")
    with open(model_dir / "lm_itos.pkl", "wb") as itos_file:
        pickle.dump(learner.data.vocab.itos, itos_file)
    with open(model_dir / "model_hparams.json", "w") as model_hparams_file:
        json.dump(model_hparams, model_hparams_file, indent=2)

    # TODO: store train / val metrics


if __name__ == "__main__":
    fire.Fire(train_lm)
