import torch
import numpy as np


def to_tensor(x):
    return torch.from_numpy(x).contiguous().float() if isinstance(x, np.ndarray) else x
