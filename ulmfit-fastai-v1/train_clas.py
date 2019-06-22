import sys, fire

from fastai.text import *
from utils import *


def train_clas(data_dir, model_dir, cyc_len = 1, lr = 4e-3, lr_factor = 1/2.6):
    """
    Finetunes the classifier on the given classification dataset, starting
    with the given language model.
    <data_dir>: folder where the dataset is located
    <model_dir>: where the model (or more concretely, the encoder part)
        is located; the decoder is constructed ad-hoc
    <cyc_len>: cycle length (number of epochs for each layer)
    <lr>: learning rate at the last layer
    <lr_factor>: learning rate of layer n is layerning rate at layer (n+1)
        times this
    """
    data_lm = load_data(data_dir, "data_lm.pkl")
    
    awd_lstm_lm_config = dict(
        emb_sz=100, n_hid=256, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.1,
        hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True
    )
    learner = lm_learner(data_lm, AWD_LSTM, model_dir, pretrained=False, config=awd_lstm_lm_config)
    learner.model_dir = ""
    learner.load_encoder("lm_encoder")
    
    # Calculate learning rates for each layer.
    num_layers = len(learner.layer_groups)
    lrs = [lr * lr_factor**i for i in range(num_layers)][::-1]
    
    # Gradual unfreezing, discriminative fine-tuning, and slanted
    # triangular learning rates.
    for i in range(num_layers)[::-1]:
        learner.freeze_to(i)
        learner.fit_one_cycle(cyc_len = cyc_len, max_lr = lrs, div_factor = 32, pct_start = 0.1)
    
    learner.save_encoder("lm_class")


if __name__ == '__main__': fire.Fire(train_clas)
