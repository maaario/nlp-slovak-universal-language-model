import sys
import os

from fastai.text import *
import fire

from utils import *


def train_clas(
    data_dir, model_dir, dest_dir=None,
    cyc_len=1, lr=4e-3, lr_factor=1/2.6,
):
    """
    Finetunes the classifier on the given classification dataset, starting
    with the given language model.
    <data_dir>: folder where the dataset is located
    <model_dir>: where the (finetuned) language model is located
    <dest_dir>: where should we stored the finetuned classifier? Defaults
        to <model_dir>/<dataset_name> (where <dataset_name> is the name
        of the last folder of <data_dir>).
    
    <cyc_len>: Determines the number of epochs dedicated to finetuning each
        layer. That is, firstly the last layer group is unfrozen and train for
        <cyc_len> epochs, then the second to last group is unfrozen and
        trained for <cyc_len> epochs, ... until the very first group is
        unfrozen and trained for <cyc_len> epochs. Cyclic learning rate
        scheduling is used. The total number of epochs is thus
        <cyc_len>*<number_of_layer_groups>.
    <lr>: learning rate at the last layer
    <lr_factor>: learning rate of layer n is layerning rate at layer (n+1)
        times this
    """
    data_lm = load_data(data_dir, "data_clas.pkl")
    model_dir = Path(model_dir)
    dest_dir = (Path(dest_dir) if dest_dir else os.path.basename(Path(data_dir)))

    # Load config, but remove entries that do not affect the classifier.
    with open(model_dir / "model_hparams.json", "r") as model_hparams_file:
        model_hparams = json.load(model_hparams_file)
    for key in ["tie_weights", "out_bias"]:
        model_hparams.pop(key, None)
    
    fmacro = FBeta(average="macro", beta=1)
    fmacro.name = "f1_macro"
    fweighted = FBeta(average="weighted", beta=1)
    fweighted.name = "f1_weighted"
    metrics = [accuracy, fmacro, fweighted]
    
    learner = clas_learner(data_lm, AWD_LSTM, model_dir, pretrained=1, config=model_hparams, metrics=metrics)
    learner.path = dest_dir

    # Calculate learning rates for each layer.
    num_layers = len(learner.layer_groups)
    lrs = [lr * lr_factor**i for i in range(num_layers)][::-1]

    # Gradual unfreezing, discriminative fine-tuning, and slanted
    # triangular learning rates.
    for i in range(num_layers)[::-1]:
        learner.freeze_to(i)
        learner.fit_one_cycle(cyc_len=cyc_len, max_lr=lrs, div_factor=32, pct_start=0.1)

    # Save EVERYthing.
    learner.save_encoder("clas_encoder")
    torch.save(learner.model.state_dict(), dest_dir / "clas_wgts.pth")
    with open(dest_dir / "clas_itos.pkl", "wb") as itos_file:
        pickle.dump(learner.data.vocab.itos, itos_file)
    with open(dest_dir / "model_hparams.json", "w") as model_hparams_file:
        json.dump(model_hparams, model_hparams_file, indent=2)


if __name__ == '__main__':
    fire.Fire(train_clas)
