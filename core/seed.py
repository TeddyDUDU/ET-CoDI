"""Global random seed and PyTorch device selection."""

import random

import numpy as np
import torch

GLOBAL_SEED = 2026

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(GLOBAL_SEED)
