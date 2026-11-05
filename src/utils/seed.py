"""
Reproducibility Seed Management.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets deterministic seeds across Python, NumPy, and PyTorch for full reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[REPRODUCIBILITY] All random seeds fixed to {seed}")
