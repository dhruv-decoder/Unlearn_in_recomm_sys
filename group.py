# group.py
import numpy as np
from sklearn.cluster import KMeans

class DataGrouper:
    def __init__(self, config):
        self.config = config
        self.num_groups = config.num_groups
        self.seed = config.seed

    def create_groups_kmeans(self, interaction_matrix, k_emb=20):
        """SVD -> user embeddings -> KMeans"""
        U, s, Vt = np.linalg.svd(interaction_matrix, full_matrices=False)
        k = min(k_emb, U.shape[1])
        user_embeddings = U[:, :k] * s[:k]  # broadcasting s
        kmeans = KMeans(n_clusters=self.num_groups, random_state=self.seed)
        labels = kmeans.fit_predict(user_embeddings)
        groups = [np.where(labels == g)[0] for g in range(self.num_groups)]
        return groups

    def create_groups_random(self, num_users):
        idx = np.arange(num_users)
        rng = np.random.RandomState(self.seed)
        rng.shuffle(idx)
        groups = np.array_split(idx, self.num_groups)
        return groups

    def create_groups_stratified(self, interaction_matrix):
        """
        Stratified by user positive-rate: sorts users by fraction positive interactions, then round-robin assign to shards.
        This helps balance class/interaction rate across shards.
        """
        user_pos = interaction_matrix.sum(axis=1)
        user_total = (interaction_matrix != 0).sum(axis=1)
        pos_rate = np.divide(user_pos, np.maximum(user_total, 1))
        # sort users by pos_rate
        order = np.argsort(pos_rate)
        groups = [[] for _ in range(self.num_groups)]
        # assign in round-robin to balance
        for i, u in enumerate(order):
            groups[i % self.num_groups].append(u)
        groups = [np.array(g, dtype=int) for g in groups]
        return groups
