import torch
import numpy as np


def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)