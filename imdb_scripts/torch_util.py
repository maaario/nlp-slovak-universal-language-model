import torch


def make_cuda(tensor):
    """Turn the tensor into cuda if possible."""
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor
