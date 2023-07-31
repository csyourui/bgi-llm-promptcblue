
import random

import numpy as np
import torch


def set_random_seed(seed: int):
    """
    Set the random seed for `random`, `numpy`, `torch`, `torch.cuda`.

    Parameters
    ------------
    seed : int
        The default seed.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
