# config.py
import os
from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    """Central project configuration with sensible defaults for experiments."""

    dataset: str = 'ml1m'
    data_path: str = './data/ml1m/'

    num_factors: int = 20
    lr: float = 0.01
    reg: float = 1e-4
    epochs_per_slice: int = 5
    batch_size: int = 512

    num_groups: int = 5
    num_slices: int = 2
    del_percentage: int = 2
    del_type: str = 'rand'

    seed: int = 42
    verbose: int = 1
    device: str = field(init=False)
    checkpoint_dir: str = './checkpoints'

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        os.makedirs(self.checkpoint_dir, exist_ok=True)


config = Config()
