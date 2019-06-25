import json
from pathlib import Path
import pickle

from fastai.text import accuracy, AWD_LSTM, FBeta, load_data
import fire
import torch

from utils import clas_learner


def train_clas(data_dir, model_dir, dest_dir=None, cyc_len=1, lr=4e-3, lr_factor=1/2.6,
               pretrained=1):
    """
    Trains a classifier on the given classification dataset, starting with the given language model.

    Attributes:
        data_dir (str): The folder where the dataset is located.
        model_dir (str): The folder where the (finetuned) language model is located.
        dest_dir (str): The folder where to store the trained classifier. Defaults
            to `model_dir` / name of the last folder of `data_dir`.
        cyc_len (int): Determines the number of epochs dedicated to finetuning each
            layer. That is, firstly the last layer group is unfrozen and trained for
            `cyc_len` epochs, then the last but one group is unfrozen and
            trained for `cyc_len` epochs, ... In the last iteration, all layer groups are
            unfrozen and trained for `cyc_len` epochs. Cyclic learning rate
            scheduling is used. The total number of epochs is thus
            `cyc_len` * number of layer groups.
        lr (float): Learning rate at the last layer.
        lr_factor (float): Learning rate of layer n is learning rate at layer (n+1) times
            `lr_factor`.
        pretrained (int): If 0, starts from with untrained language model.
            If 1, loads a finetuned language model from `model_dir`.
            If 2, loads an already trained classifier from `model_dir`.
            [2 is CURRENTLY BROKEN, seems like load_pretrained does not
            work with classifiers...?]
    """
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    dest_dir = (Path(dest_dir) if dest_dir else data_dir.name)
    dest_dir.mkdir(parents=True, exist_ok=True)

    data_lm = load_data(data_dir, "data_clas.pkl")

    # Load config, but remove entries that do not affect the classifier.
    hparams_fname = ("model_hparams.json" if pretrained != 2 else "clas_hparams.json")
    with open(model_dir / hparams_fname, "r") as model_hparams_file:
        model_hparams = json.load(model_hparams_file)
    for key in ["tie_weights", "out_bias"]:
        model_hparams.pop(key, None)

    fmacro = FBeta(average="macro", beta=1)
    fmacro.name = "f1_macro"
    fweighted = FBeta(average="weighted", beta=1)
    fweighted.name = "f1_weighted"
    metrics = [accuracy, fmacro, fweighted]

    learner = clas_learner(
        data_lm, AWD_LSTM, model_dir, pretrained=pretrained, config=model_hparams, metrics=metrics)
    learner.path = dest_dir

    # Calculate learning rates for each layer.
    num_layers = len(learner.layer_groups)
    lrs = [lr * lr_factor**i for i in range(num_layers)][::-1]

    # Gradual unfreezing, discriminative fine-tuning, and slanted
    # triangular learning rates.
    for i in range(num_layers)[::-1]:
        learner.freeze_to(i)
        learner.fit_one_cycle(cyc_len=cyc_len, max_lr=lrs, div_factor=32, pct_start=0.1)

    # Save everything.
    learner.save_encoder("clas_encoder")
    torch.save(learner.model.state_dict(), dest_dir / "clas_wgts.pth")
    with open(dest_dir / "clas_itos.pkl", "wb") as itos_file:
        pickle.dump(learner.data.vocab.itos, itos_file)
    with open(dest_dir / "clas_hparams.json", "w") as model_hparams_file:
        json.dump(model_hparams, model_hparams_file, indent=2)


if __name__ == "__main__":
    fire.Fire(train_clas)
