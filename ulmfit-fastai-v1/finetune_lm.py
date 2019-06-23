import sys

from fastai.text import *
import fire

from utils import *


def finetune_lm(data_dir, model_dir, dest_dir=None, cyc_len=25, lr=4e-3, lr_factor=1/2.6):
    """
    Finetunes the provided language model on the given data, uses
    discriminative fine-tuning and slanted triangular learning rates.

    <data_dir>: the directory from which to take input data
    <model_dir>: where the pretrained model is located
    <dest_dir>: where to store the finetuned language model. Defaults
        to <model_dir>/<dataset_name> (name of the data_dir last folder).
    <cyc_len>: number of epochs. Since we are using the one cycle learning
        rate scheduler, we call it <cyc_len> (as "cycle length").
    <lr>: learning rate at the last layer
    <lr_factor>: learning rate of layer n is layerning rate at layer (n+1)
        times this
    """
    data_lm = load_data(data_dir, "data_finetune_lm.pkl")
    model_dir = Path(model_dir)
    dest_dir = (Path(dest_dir) if dest_dir else model_dir / os.path.basename(Path(data_dir)))

    with open(model_dir / "model_hparams.json", "r") as model_hparams_file:
        model_hparams = json.load(model_hparams_file)
    learner = lm_learner(data_lm, AWD_LSTM, model_dir, pretrained=True, config=model_hparams)
    learner.path = dest_dir

    # Calculate learning rates for each layer.
    num_layers = len(learner.layer_groups)
    lrs = [lr * lr_factor**i for i in range(num_layers)][::-1]

    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=cyc_len, max_lr=lrs, div_factor=32, pct_start=0.1)

    # Save EVERYthing.
    learner.save_encoder("lm_finetuned_encoder")
    torch.save(learner.model.state_dict(), dest_dir / "lm_finetuned_wgts.pth")
    with open(dest_dir / "lm_finetuned_itos.pkl", "wb") as itos_file:
        pickle.dump(learner.data.vocab.itos, itos_file)
    with open(dest_dir / "model_hparams.json", "w") as model_hparams_file:
        json.dump(model_hparams, model_hparams_file, indent=2)


if __name__ == '__main__':
    fire.Fire(finetune_lm)
