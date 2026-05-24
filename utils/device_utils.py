import torch

def get_device():

    if torch.cuda.is_available():
        return "cuda"

    elif torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_dtype(device):

    if device == "cpu":
        return torch.float32

    return torch.float16