from math import exp
import os
import pickle
import sys

from fastai.text import *
import fire
import numpy as np
import torch

from utils import *

"""
python main.py prepare_lm_dataset sk-featured-smaller.csv lm_dataset --min_freq=2
python main.py prepare_lm_dataset zlocin-a-trest.csv zlocin-a-trest --min_freq=1

python main.py train_lm lm_dataset trained_lm --epochs=5

python main.py evaluate_lm zlocin-a-trest trained_lm
python main.py evaluate_lm lm_dataset trained_lm

python main.py prepare_clas_dataset clas-alza.csv clas_dataset
"""


def prepare_lm_dataset(input_path, output_dir, valid_split=0.2, min_freq=2, seed=42):
    """
    Reads .csv file with texts for training LM model, splits it into training and validation sets,
    tokenizes and saves the dataset.
    """
    np.random.seed(seed)
    train_df, valid_df = csv_to_train_valid_df(input_path, valid_split)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = Path(output_dir)

    data_lm = TextLMDataBunch.from_df(
        output_dir, train_df, valid_df, text_cols=0, tokenizer=Tokenizer(lang="xx"),
        min_freq=min_freq)
    data_lm.save('data_lm.pkl')

    with open(output_dir / "data_lm_tokenized_train.txt", "w") as f:
        f.write("\n".join(map(str, list(data_lm.train_ds.x))))
    with open(output_dir / "data_lm_tokenized_valid.txt", "w") as f:
        f.write("\n".join(map(str, list(data_lm.valid_ds.x))))


def prepare_clas_dataset(input_path, output_dir, valid_split=0.2, seed=42):
    """
    Reads .csv file with labels in first column, texts in second columns.
    Splits it into training and validation sets, tokenizes and saves datasets for fine-tuning and
    for classification.
    """
    np.random.seed(seed)
    train_df, valid_df = csv_to_train_valid_df(input_path, valid_split)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = Path(output_dir)

    data_finetune_lm = TextLMDataBunch.from_df(
        output_dir, train_df, valid_df, tokenizer=Tokenizer(lang="xx"))
    data_clas = TextClasDataBunch.from_df(
        output_dir, train_df, valid_df, tokenizer=Tokenizer(lang="xx"),
        vocab=data_finetune_lm.train_ds.vocab, bs=32)

    data_finetune_lm.save('data_finetune_lm.pkl')
    data_clas.save('data_clas.pkl')

    # TODO: if we want e.g. for fastext, also store .txt versions of tokenized texts


def train_lm(data_dir, model_dir, epochs=12, lr=3e-4):
    """
    Trains new language model using the provided dataset.
    """
    data_lm = load_data(data_dir, 'data_lm.pkl')

    awd_lstm_lm_config = dict(
        emb_sz=100, n_hid=256, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.1,
        hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)

    learner = language_model_learner(data_lm, AWD_LSTM, pretrained=False, config=awd_lstm_lm_config)
    learner.fit(epochs, lr)

    loss, acc = learner.validate(learner.data.train_dl)
    print("Training - Loss: {}, Perplexity: {}, Accuracy: {}".format(loss, exp(loss), acc))
    loss, acc = learner.validate()
    print("Validation - Loss: {}, Perplexity: {}, Accuracy: {}".format(loss, exp(loss), acc))

    model_dir = Path(model_dir)
    torch.save(learner.model.state_dict(), model_dir / "lm_wgts.pth")
    with open(model_dir / "lm_itos.pkl", "wb") as f:
        pickle.dump(learner.data.vocab.itos, f)

    # TODO: store train / val metrics
    # store model architecture config -> needed for model loading, add tokenizer to config


def evaluate_lm(data_path, model_dir, train=False, custom_pp=False):
    """
    Evaluate metrics of a trained LM using any dataset of texts in csv.

    The perplexity estimated as e^(avg. loss), but the average loss changes slightly with batch
    size. To get perplexity computed in slower but controlled fashion, set `custom_pp` to True.
    The discrepancy between perplexity and custom perplexity is empirically approximately 1%.
    """
    model_dir = Path(model_dir)
    with open(model_dir / "lm_itos.pkl", "rb") as f:
        itos = pickle.load(f)

    data_df = pd.read_csv(data_path, header=None)
    data = TextLMDataBunch.from_df(
        "", data_df, data_df, text_cols=0, tokenizer=Tokenizer(lang="xx"),
        vocab=Vocab(itos))

    awd_lstm_lm_config = dict(
        emb_sz=100, n_hid=256, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.1,
        hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)

    learner = lm_learner(data, AWD_LSTM, model_dir, pretrained=True, config=awd_lstm_lm_config)
    loss, acc = learner.validate()
    print("Loss: {}, Perplexity: {}, Accuracy: {}".format(loss, exp(loss), acc))

    if custom_pp:
        print("Custom perplexity: {}".format(evaluate_perplexity(learner, data_lm.valid_ds.x)))

if __name__ == '__main__':
    fire.Fire()
