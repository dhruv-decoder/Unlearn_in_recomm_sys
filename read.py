# read.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data_path = config.data_path

    def load_data(self):
        """Load MovieLens 1M dataset ratings.dat and convert to implicit feedback"""
        ratings = pd.read_csv(
            f'{self.data_path}ratings.dat',
            sep='::',
            header=None,
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )

        # implicit label
        ratings['implicit'] = (ratings['rating'] >= 3.5).astype(int)

        # continuous indexing
        users = np.sort(ratings['user_id'].unique())
        items = np.sort(ratings['item_id'].unique())

        user_map = {u: idx for idx, u in enumerate(users)}
        item_map = {it: idx for idx, it in enumerate(items)}

        ratings['user_idx'] = ratings['user_id'].map(user_map)
        ratings['item_idx'] = ratings['item_id'].map(item_map)

        num_users = len(users)
        num_items = len(items)

        # return dataframe and sizes
        return ratings.reset_index(drop=True), num_users, num_items

    def split_data(self, ratings, test_size=0.2):
        """Simple random split (keeps users/items in both sets)."""
        train_data, test_data = train_test_split(
            ratings,
            test_size=test_size,
            random_state=self.config.seed
        )
        return train_data.reset_index(drop=True), test_data.reset_index(drop=True)

    def create_interaction_matrix(self, data, num_users, num_items):
        """Dense interaction matrix (0/1) used for grouping and mask creation."""
        matrix = np.zeros((num_users, num_items), dtype=np.int8)
        for _, row in data.iterrows():
            matrix[int(row['user_idx']), int(row['item_idx'])] = int(row['implicit'])
        return matrix
