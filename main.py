# main.py
import argparse
import time
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from config import config
from read import DataLoader
from group import DataGrouper

# --------------------
# Model
# --------------------
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, num_factors=20):
        super().__init__()
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)

    def forward(self, users, items):
        u = self.user_factors(users)
        v = self.item_factors(items)
        return (u * v).sum(dim=1)

    def score_user_all_items(self, user_idx):
        device = next(self.parameters()).device
        u = self.user_factors(torch.LongTensor([user_idx]).to(device))  # (1, d)
        all_items = self.item_factors.weight  # (num_items, d)
        scores = all_items @ u.view(-1, 1)    # (num_items, 1)
        return scores.view(-1)                # (num_items,)


# --------------------
# Metrics
# --------------------
def precision_recall_ndcg_at_k(recommended_lists, ground_truth, k=10):
    """
    recommended_lists: dict user -> list of recommended item indices (ordered)
    ground_truth: dict user -> set of positive item indices in test
    """
    precisions, recalls, ndcgs = [], [], []

    for u, recs in recommended_lists.items():
        gt = ground_truth.get(u, set())
        if len(gt) == 0:
            continue
        topk = recs[:k]
        hits = [1 if it in gt else 0 for it in topk]
        num_hits = sum(hits)

        precisions.append(num_hits / k)
        recalls.append(num_hits / len(gt))

        # DCG
        dcg = sum(hits[i] / np.log2(i + 2) for i in range(len(hits)))
        # IDCG
        ideal_hits = min(len(gt), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    if not precisions:
        return 0.0, 0.0, 0.0
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(ndcgs))


# --------------------
# Global (baseline) recommender – one model, naive full retrain
# --------------------
class GlobalRecommender:
    def __init__(self, config, num_users, num_items, train_df, test_df):
        self.config = config
        self.num_users = num_users
        self.num_items = num_items
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        self.device = torch.device(self.config.device)

        self.model = None
        self.train_user_items = self._build_user_items_map(self.train_df)
        self.test_user_items = self._build_user_items_map(self.test_df)

    def _build_user_items_map(self, df):
        d = {}
        for _, row in df.iterrows():
            u = int(row['user_idx']); i = int(row['item_idx'])
            d.setdefault(u, set()).add(i)
        return d

    def _init_model(self):
        model = MatrixFactorization(self.num_users, self.num_items, self.config.num_factors)
        return model.to(self.device)

    def train(self, total_epochs=None):
        """Train global MF model on full train_df."""
        if total_epochs is None:
            total_epochs = self.config.epochs_per_slice * self.config.num_slices

        self.model = self._init_model()
        model = self.model
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.reg)
        criterion = nn.BCEWithLogitsLoss()

        users = torch.LongTensor(self.train_df['user_idx'].values).to(self.device)
        items = torch.LongTensor(self.train_df['item_idx'].values).to(self.device)
        ratings = torch.FloatTensor(self.train_df['implicit'].values).to(self.device)

        n = len(users)
        bs = self.config.batch_size

        for ep in range(total_epochs):
            perm = torch.randperm(n, device=self.device)
            total_loss = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i+bs]
                u_b = users[idx]; it_b = items[idx]; r_b = ratings[idx]
                optimizer.zero_grad()
                preds = model(u_b, it_b)
                loss = criterion(preds, r_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)
            if self.config.verbose:
                print(f"[Global] Epoch {ep+1}/{total_epochs} - loss {total_loss / max(1, n):.4f}")

    def _predict_topk_for_user(self, user_idx, k=10):
        self.model.eval()
        with torch.no_grad():
            scores = self.model.score_user_all_items(user_idx).cpu().numpy()
        train_items = self.train_user_items.get(user_idx, set())
        if train_items:
            scores[list(train_items)] = -np.inf
        topk_idx = np.argpartition(-scores, k)[:k]
        topk_sorted = topk_idx[np.argsort(-scores[topk_idx])]
        return topk_sorted.tolist()

    def evaluate(self, k=10, users_sample=None):
        if users_sample is None:
            users = list(self.test_user_items.keys())
        else:
            users = users_sample
        recs = {u: self._predict_topk_for_user(u, k=k) for u in users}
        p, r, n = precision_recall_ndcg_at_k(recs, self.test_user_items, k=k)
        return {
            f'precision@{k}': p,
            f'recall@{k}': r,
            f'ndcg@{k}': n,
        }


# --------------------
# SISA Recommender – shards + slices + checkpoints
# --------------------
class SISARecommender:
    def __init__(self, config, num_users, num_items, train_df, test_df, groups):
        self.config = config
        self.num_users = num_users
        self.num_items = num_items
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        self.groups = groups
        self.user_to_shard = self._map_user_to_shard(groups)
        self.device = torch.device(self.config.device)

        self.shard_models = [None] * len(groups)
        self.shard_checkpoints = {}  # (shard, slice_idx) -> path

        self.train_user_items = self._build_user_items_map(self.train_df)
        self.test_user_items = self._build_user_items_map(self.test_df)

    def _map_user_to_shard(self, groups):
        mapping = {}
        for g_idx, users in enumerate(groups):
            for u in users:
                mapping[int(u)] = g_idx
        return mapping

    def _build_user_items_map(self, df):
        d = {}
        for _, row in df.iterrows():
            u = int(row['user_idx']); i = int(row['item_idx'])
            d.setdefault(u, set()).add(i)
        return d

    def _shard_dfs(self):
        shard_dfs = []
        for g_idx, users in enumerate(self.groups):
            mask = self.train_df['user_idx'].isin(users)
            shard_dfs.append(self.train_df[mask].reset_index(drop=True))
        return shard_dfs

    def _init_model(self):
        model = MatrixFactorization(self.num_users, self.num_items, self.config.num_factors)
        return model.to(self.device)

    def _train_on_df(self, model, df, epochs):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.reg)
        criterion = nn.BCEWithLogitsLoss()

        users = torch.LongTensor(df['user_idx'].values).to(self.device)
        items = torch.LongTensor(df['item_idx'].values).to(self.device)
        ratings = torch.FloatTensor(df['implicit'].values).to(self.device)

        n = len(users)
        bs = self.config.batch_size

        for ep in range(epochs):
            perm = torch.randperm(n, device=self.device)
            total_loss = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i+bs]
                u_b = users[idx]; it_b = items[idx]; r_b = ratings[idx]
                optimizer.zero_grad()
                preds = model(u_b, it_b)
                loss = criterion(preds, r_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)
            if self.config.verbose:
                print(f"    [Shard] Epoch {ep+1}/{epochs} - loss {total_loss / max(1, n):.4f}")
        return model

    def train_shards(self):
        print("=== SISA: training shard models with slices + checkpoints ===")
        shard_dfs = self._shard_dfs()

        for s_idx, shard_df in enumerate(shard_dfs):
            print(f"\n-- Shard {s_idx} | rows = {len(shard_df)} --")
            shard_df = shard_df.sample(frac=1, random_state=self.config.seed).reset_index(drop=True)
            slices = np.array_split(shard_df, self.config.num_slices)

            model = self._init_model()
            cumulative_df = None

            for slice_idx, slice_df in enumerate(slices):
                if cumulative_df is None:
                    cumulative_df = slice_df.copy()
                else:
                    cumulative_df = pd.concat([cumulative_df, slice_df]).reset_index(drop=True)

                print(f"Shard {s_idx}: training slice {slice_idx}, cumulative rows = {len(cumulative_df)}")
                model = self._train_on_df(model, cumulative_df, epochs=self.config.epochs_per_slice)

                ckpt_path = os.path.join(self.config.checkpoint_dir, f"shard_{s_idx}_slice_{slice_idx}.pt")
                torch.save(model.state_dict(), ckpt_path)
                self.shard_checkpoints[(s_idx, slice_idx)] = ckpt_path

            self.shard_models[s_idx] = model

        print("\n=== Finished SISA training for all shards ===")

    def _predict_topk_for_user(self, user_idx, k=10):
        shard = self.user_to_shard.get(user_idx, None)
        if shard is None:
            return []
        model = self.shard_models[shard]
        if model is None:
            return []
        model.eval()
        with torch.no_grad():
            scores = model.score_user_all_items(user_idx).cpu().numpy()
        train_items = self.train_user_items.get(user_idx, set())
        if train_items:
            scores[list(train_items)] = -np.inf
        topk_idx = np.argpartition(-scores, k)[:k]
        topk_sorted = topk_idx[np.argsort(-scores[topk_idx])]
        return topk_sorted.tolist()

    def evaluate(self, k=10, users_sample=None):
        if users_sample is None:
            users = list(self.test_user_items.keys())
        else:
            users = users_sample
        recs = {u: self._predict_topk_for_user(u, k=k) for u in users}
        p, r, n = precision_recall_ndcg_at_k(recs, self.test_user_items, k=k)
        return {
            f'precision@{k}': p,
            f'recall@{k}': r,
            f'ndcg@{k}': n,
        }

    # ---------- Unlearning ----------
    def _find_earliest_slice_for_users(self, shard_idx, users_to_delete):
        mask = self.train_df['user_idx'].isin(self.groups[shard_idx])
        shard_df = self.train_df[mask].sample(frac=1, random_state=self.config.seed).reset_index(drop=True)
        slices = np.array_split(shard_df, self.config.num_slices)
        for slice_idx, slice_df in enumerate(slices):
            if slice_df['user_idx'].isin(users_to_delete).any():
                return slice_idx
        return None

    def unlearn_shard(self, shard_idx, users_to_delete):
        print(f"\n=== SISA unlearning: shard {shard_idx}, users {len(users_to_delete)} ===")
        earliest = self._find_earliest_slice_for_users(shard_idx, users_to_delete)
        if earliest is None:
            print("No matching rows in shard; nothing to unlearn.")
            return 0.0

        start_slice = max(0, earliest - 1)
        ckpt_key = (shard_idx, start_slice)

        # load checkpoint or fresh model
        if ckpt_key in self.shard_checkpoints:
            model = self._init_model()
            model.load_state_dict(torch.load(self.shard_checkpoints[ckpt_key], map_location=self.device))
            print(f"Loaded checkpoint for shard {shard_idx}, slice {start_slice}")
        else:
            print("Checkpoint missing, starting from scratch for this shard.")
            model = self._init_model()

        # rebuild shard df and slices
        mask = self.train_df['user_idx'].isin(self.groups[shard_idx])
        shard_df = self.train_df[mask].sample(frac=1, random_state=self.config.seed).reset_index(drop=True)
        slices = np.array_split(shard_df, self.config.num_slices)

        cumulative = None
        for s_idx in range(start_slice + 1, len(slices)):
            slice_df = slices[s_idx]
            slice_df = slice_df[~slice_df['user_idx'].isin(users_to_delete)].reset_index(drop=True)
            if len(slice_df) == 0:
                continue
            if cumulative is None:
                cumulative = slice_df.copy()
            else:
                cumulative = pd.concat([cumulative, slice_df]).reset_index(drop=True)

        if cumulative is None or len(cumulative) == 0:
            print("No remaining data after unlearning; keeping checkpoint model.")
            self.shard_models[shard_idx] = model
            for u in users_to_delete:
                self.train_user_items.pop(u, None)
            return 0.0

        t0 = time.time()
        model = self._train_on_df(model, cumulative, epochs=self.config.epochs_per_slice)
        t1 = time.time()
        self.shard_models[shard_idx] = model

        for u in users_to_delete:
            self.train_user_items.pop(u, None)

        unlearn_time = t1 - t0
        print(f"SISA unlearning (shard {shard_idx}) took {unlearn_time:.2f}s")
        return unlearn_time

    def full_retrain_all_shards(self):
        """If you want: re-run train_shards and time it."""
        t0 = time.time()
        self.train_shards()
        t1 = time.time()
        return t1 - t0


# --------------------
# Main experiment flow / CLI
# --------------------
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="SISA-based unlearning on MovieLens 1M")
    parser.add_argument("--action", type=str,
                        choices=["baseline_full", "train_sisa", "compare_unlearn"],
                        default="compare_unlearn")
    parser.add_argument("--shard-method", type=str,
                        choices=["kmeans", "random", "stratified"],
                        default="kmeans")
    parser.add_argument("--num-groups", type=int, default=5)
    parser.add_argument("--num-slices", type=int, default=2)
    parser.add_argument("--k", type=int, default=10, help="K for Precision@K / NDCG@K")
    parser.add_argument("--shard", type=int, default=-1, help="Shard index for unlearning (default: random)")
    args = parser.parse_args()

    # apply config
    config.num_groups = args.num_groups
    config.num_slices = args.num_slices

    print(f"Using device: {config.device}")
    print(f"Shard method: {args.shard_method}, num_groups={args.num_groups}, num_slices={args.num_slices}")

    set_seeds(config.seed)

    # --- Load data ---
    loader = DataLoader(config)
    ratings, num_users, num_items = loader.load_data()
    train_df, test_df = loader.split_data(ratings, test_size=0.2)
    print(f"Users: {num_users}, Items: {num_items}")
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # interaction matrix for grouping
    interaction_matrix = loader.create_interaction_matrix(train_df, num_users, num_items)

    # --- Create groups ---
    grouper = DataGrouper(config)
    if args.shard_method == "kmeans":
        groups = grouper.create_groups_kmeans(interaction_matrix)
    elif args.shard_method == "random":
        groups = grouper.create_groups_random(num_users)
    else:
        groups = grouper.create_groups_stratified(interaction_matrix)

    print("Group sizes:", [len(g) for g in groups])

    # ========== ACTIONS ==========
    if args.action == "baseline_full":
        print("\n=== ACTION: baseline_full ===")
        global_rec = GlobalRecommender(config, num_users, num_items, train_df, test_df)
        t0 = time.time()
        global_rec.train()
        t1 = time.time()
        metrics = global_rec.evaluate(k=args.k)
        print(f"Baseline training time: {t1 - t0:.2f}s")
        print(f"Baseline metrics (k={args.k}): {metrics}")

    elif args.action == "train_sisa":
        print("\n=== ACTION: train_sisa ===")
        sisa = SISARecommender(config, num_users, num_items, train_df, test_df, groups)
        t0 = time.time()
        sisa.train_shards()
        t1 = time.time()
        metrics = sisa.evaluate(k=args.k)
        print(f"SISA training time: {t1 - t0:.2f}s")
        print(f"SISA metrics (k={args.k}): {metrics}")

    elif args.action == "compare_unlearn":
        print("\n=== ACTION: compare_unlearn (baseline vs SISA) ===")

        # select shard to delete
        if args.shard >= 0:
            shard_idx = args.shard
        else:
            shard_idx = random.randint(0, config.num_groups - 1)
        users_to_delete = list(groups[shard_idx])
        print(f"Selected shard {shard_idx} with {len(users_to_delete)} users to unlearn.")

        # ----- Baseline global: before & after full retrain -----
        print("\n[Baseline] Training original global model...")
        global_before = GlobalRecommender(config, num_users, num_items, train_df, test_df)
        tb0 = time.time()
        global_before.train()
        tb1 = time.time()
        base_metrics_before = global_before.evaluate(k=args.k)
        print(f"[Baseline] Training time (original): {tb1 - tb0:.2f}s")
        print(f"[Baseline] Metrics before unlearning: {base_metrics_before}")

        # naive unlearning = full retrain on filtered data
        print("[Baseline] Naive unlearning: full retrain after removing shard users...")
        mask = ~train_df['user_idx'].isin(users_to_delete)
        train_after = train_df[mask].reset_index(drop=True)
        global_after = GlobalRecommender(config, num_users, num_items, train_after, test_df)
        tb2 = time.time()
        global_after.train()
        tb3 = time.time()
        base_metrics_after = global_after.evaluate(k=args.k)
        naive_unlearn_time = tb3 - tb2
        print(f"[Baseline] Naive unlearning retrain time: {naive_unlearn_time:.2f}s")
        print(f"[Baseline] Metrics after unlearning: {base_metrics_after}")

        # ----- SISA: before & after shard-only retrain -----
        print("\n[SISA] Training shard models...")
        sisa = SISARecommender(config, num_users, num_items, train_df, test_df, groups)
        ts0 = time.time()
        sisa.train_shards()
        ts1 = time.time()
        sisa_metrics_before = sisa.evaluate(k=args.k)
        print(f"[SISA] Training time (all shards): {ts1 - ts0:.2f}s")
        print(f"[SISA] Metrics before unlearning: {sisa_metrics_before}")

        print("[SISA] Unlearning by retraining only affected shard...")
        sisa_unlearn_time = sisa.unlearn_shard(shard_idx, users_to_delete)
        sisa_metrics_after = sisa.evaluate(k=args.k)
        print(f"[SISA] Metrics after unlearning: {sisa_metrics_after}")

        print("\n=== Summary (k = {}) ===".format(args.k))
        print(f"Baseline naive-unlearn retrain time: {naive_unlearn_time:.2f}s")
        print(f"SISA shard-unlearn time: {sisa_unlearn_time:.2f}s")
        print(f"Baseline metrics before: {base_metrics_before}")
        print(f"Baseline metrics after:  {base_metrics_after}")
        print(f"SISA metrics before:    {sisa_metrics_before}")
        print(f"SISA metrics after:     {sisa_metrics_after}")


if __name__ == "__main__":
    main()
