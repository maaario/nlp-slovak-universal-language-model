import sys
import os

from fastai.text import *
import fire

from utils import *


def train_clas(data_dir, model_dir, class_dir=None, cyc_len=1, lr=4e-3, lr_factor=1/2.6):
    """
    Finetunes the classifier on the given classification dataset, starting
    with the given language model.
    <data_dir>: folder where the dataset is located
    <model_dir>: where the language model is located
    <class_dir>: relative to model_dir, where is the finetuned LM and
        where should the classifier model be stored?
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
    data_lm = load_data(data_dir, "data_lm.pkl")
    model_dir = Path(model_dir)
    class_dir = (class_dir or os.path.basename(os.path.dirname(data_dir)))
    class_dir = Path(class_dir)

    with open(model_dir / "model_hparams.json", "r") as model_hparams_file:
        model_hparams = json.load(model_hparams_file)
    learner = lm_learner(data_lm, AWD_LSTM, model_dir, pretrained=False, config=model_hparams)
    learner.model_dir = class_dir
    learner.load_encoder("lm_encoder")

    # Calculate learning rates for each layer.
    num_layers = len(learner.layer_groups)
    lrs = [lr * lr_factor**i for i in range(num_layers)][::-1]

    # Gradual unfreezing, discriminative fine-tuning, and slanted
    # triangular learning rates.
    for i in range(num_layers)[::-1]:
        learner.freeze_to(i)
        learner.fit_one_cycle(cyc_len=cyc_len, max_lr=lrs, div_factor=32, pct_start=0.1)

    learner.save_encoder("lm_class")


if __name__ == '__main__':
    fire.Fire(train_clas)
