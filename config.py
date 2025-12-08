# config.py
import os
import numpy as np
import torch

class Config:
    def __init__(self):
        # Dataset configuration
        self.dataset = 'ml1m'
        self.data_path = './data/ml1m/'

        # Model parameters
        self.num_factors = 20  # latent dims
        self.lr = 0.01
        self.reg = 1e-4
        self.epochs_per_slice = 5   # epochs to run for each slice (keeps training time reasonable)
        self.batch_size = 512

        # SISA / unlearning parameters
        self.num_groups = 5       # number of shards
        self.num_slices = 2       # slices per shard (simple default)
        self.del_percentage = 2
        self.del_type = 'rand'    # 'rand' or 'targeted'

        # Training settings / reproducibility
        self.seed = 42
        self.verbose = 1

        # Device selection (robust for Mac M-series)
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        # Where to save shard models / checkpoints
        self.checkpoint_dir = './checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

config = Config()
