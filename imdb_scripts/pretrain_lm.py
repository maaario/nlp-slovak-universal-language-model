import fire
from fastai.text import *

from sampled_sm import *


def train_lm(dir_path, cuda_id, cl=1, bs=64, backwards=False, lr=3e-4, sampled=True,
             pretrain_id=''):
    """
    <dir_path>: where the dataset is located, should contains files
        `train.csv`, `val.csv` and folder `tmp` with numpy objects.
    <cuda_id>: (???)
    <cl>: cycle_len for learner (???)
    <bs>: batch size
    <backwards>: (???) but seems not to be used in straightforward use
        where the `tmp` folder does not contain any `*_bwd.npy` files.
        Flips the batch and other dimension???
    <sampled>: used by `get_learner` (???)
    """
    print(f'dir_path {dir_path}; cuda_id {cuda_id}; cl {cl}; bs {bs}; '
          f'backwards {backwards}; lr {lr}; sampled {sampled}; '
          f'pretrain_id {pretrain_id}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)
    PRE  = 'bwd_' if backwards else 'fwd_'
    IDS = 'ids'
    p = Path(dir_path)
    assert p.exists(), f'Error: {p} does not exist.'
    bptt=70
    em_sz,nh,nl = 400,1150,3
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    # load the `trn_ids.npy` and `val_ids.npy` files and store them in
    # `trn_lm` and `val_lm` (what does suffix _lm mean?)
    if backwards:
        trn_lm = np.load(p / f'tmp/trn_{IDS}_bwd.npy', allow_pickle=True)
        val_lm = np.load(p / f'tmp/val_{IDS}_bwd.npy', allow_pickle=True)
    else:
        trn_lm = np.load(p / f'tmp/trn_{IDS}.npy', allow_pickle=True)
        val_lm = np.load(p / f'tmp/val_{IDS}.npy', allow_pickle=True)
    trn_lm = np.concatenate(trn_lm)
    val_lm = np.concatenate(val_lm)

    # load the map from ids to words
    itos = pickle.load(open(p / 'tmp/itos.pkl', 'rb'))
    vs = len(itos)      # vocabulary size

    """
    LanguageModelLoader:
    not really a language model or anything, just something wraps around
    data and can conveniently give us pairs (inputs, target outputs)
    through iteration.
    
    parameters:
        *_lm: data
        bs: batch_size --- into how many separate "documents" should the
            text be separated into? batch size determines the size of one
            such document
        bptt: sequence length --- how many consecutive words should be
            predicted at the same time? aka 2nd batch dimension
    """
    trn_dl = LanguageModelLoader(trn_lm, bs, bptt)
    val_dl = LanguageModelLoader(val_lm, bs//5 if sampled else bs, bptt)
    md = LanguageModelData(p, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

    tprs = get_prs(trn_lm, vs)
    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.5      # dropout rates?
    learner,crit = get_learner(drops, 15000, sampled, md, em_sz, nh, nl, opt_fn, tprs)
    wd=1e-7
    learner.metrics = [accuracy]

    lrs = np.array([lr/6,lr/3,lr,lr])
    #lrs=lr

    learner.fit(lrs, 1, wds=wd, use_clr=(32,10), cycle_len=cl)
    learner.save(f'{PRE}{pretrain_id}')
    learner.save_encoder(f'{PRE}{pretrain_id}_enc')

if __name__ == '__main__': fire.Fire(train_lm)
