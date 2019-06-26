import json
from pathlib import Path
import pickle

from fastai.text import AWD_LSTM, load_data
import fire
import torch

from utils import lm_learner


def finetune_lm(data_dir, model_dir, dest_dir=None, cyc_len=25, lr=4e-3, lr_factor=1/2.6):
    """
    Finetunes the provided language model on the given data, uses
    discriminative fine-tuning and slanted triangular learning rates.

    Attributes:
        data_dir (str): A directory from which to take input data
        model_dir (str): A directory where the pretrained model is located
        dest_dir (str): A directory where to store the finetuned language model. Defaults
            to `model_dir` / name of the `data_dir` last folder.
        cyc_len (int): Number of epochs for one cycle learning rate scheduler. For more details
            refer to https://docs.fast.ai/callbacks.one_cycle.html#The-1cycle-policy.
        lr (float): Learning rate at the last layer.
        lr_factor (float): Learning rate at layer n is learning rate at layer (n+1) times
            `lr_factor`.
    """
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    dest_dir = (Path(dest_dir) if dest_dir else model_dir / data_dir.name)
    dest_dir.mkdir(parents=True, exist_ok=True)

    data_lm = load_data(data_dir, "data_finetune_lm.pkl")

    with open(model_dir / "model_hparams.json", "r") as model_hparams_file:
        model_hparams = json.load(model_hparams_file)
    learner = lm_learner(data_lm, AWD_LSTM, model_dir, pretrained=True, config=model_hparams)
    learner.path = dest_dir

    # Calculate learning rates for each layer.
    num_layers = len(learner.layer_groups)
    lrs = [lr * lr_factor**i for i in range(num_layers)][::-1]

    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=cyc_len, max_lr=lrs, div_factor=32, pct_start=0.1)

    # Save everything.
    learner.save_encoder("lm_finetuned_encoder")
    torch.save(learner.model.state_dict(), dest_dir / "lm_finetuned_wgts.pth")
    with open(dest_dir / "lm_finetuned_itos.pkl", "wb") as itos_file:
        pickle.dump(learner.data.vocab.itos, itos_file)
    with open(dest_dir / "model_hparams.json", "w") as model_hparams_file:
        json.dump(model_hparams, model_hparams_file, indent=2)


if __name__ == "__main__":
    fire.Fire(finetune_lm)
