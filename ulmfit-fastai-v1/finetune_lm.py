import sys, fire

from fastai.text import *
from utils import *


def finetune_lm(data_dir, model_dir, cyc_len = 25, lr = 4e-3, lr_factor = 1/2.6):
    """
    Finetunes the provided language model on the given data, uses
    techniques from ULMFIT paper (Discriminative fine-tuning, Slanted
    triangular learning rates).
    
    <data_dir>: the directory from which to take input data
    <model_dir>: where the pretrained model is located
    <cyc_len>: cycle length (also the number of epochs -- only 1 cycle is done)
    <lr>: learning rate at the last layer
    <lr_factor>: learning rate of layer n is layerning rate at layer (n+1)
        times this
    """
    data_lm = load_data(data_dir, "data_lm.pkl")
    
    awd_lstm_lm_config = dict(
        emb_sz=100, n_hid=256, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.1,
        hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True
    )
    learner = lm_learner(data_lm, AWD_LSTM, model_dir, pretrained=True, config=awd_lstm_lm_config)
    learner.model_dir = ""
    learner.unfreeze()
    
    # Calculate learning rates for each layer.
    num_layers = len(learner.layer_groups)
    lrs = [lr * lr_factor**i for i in range(num_layers)][::-1]
    
    learner.fit_one_cycle(cyc_len = cyc_len, max_lr = lrs, div_factor = 32, pct_start = 0.1)
    learner.save_encoder("lm_encoder")


if __name__ == '__main__': fire.Fire(finetune_lm)
