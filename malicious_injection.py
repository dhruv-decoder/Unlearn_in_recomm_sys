# malicious_injection.py
import pandas as pd
import numpy as np
import random
import torch

from config import config
from read import DataLoader
from main import GlobalRecommender, SISARecommender
from group import DataGrouper

config.num_groups = 5
config.num_slices = 2

def rank_of_item_for_user_global(global_rec, user_idx, item_idx):
    """Return 1-based rank of item_idx among all items for a given user, using a GlobalRecommender."""
    model = global_rec.model
    model.eval()
    with torch.no_grad():
        scores = model.score_user_all_items(user_idx).detach().cpu().numpy()
    order = np.argsort(-scores)  # descending
    pos = np.where(order == item_idx)[0]
    return int(pos[0]) + 1 if len(pos) > 0 else None

def rank_of_item_for_user_sisa(sisa_rec, user_idx, item_idx):
    """Return 1-based rank of item_idx for a SISARecommender (use the shard model for this user)."""
    shard = sisa_rec.user_to_shard.get(user_idx)
    if shard is None:
        return None
    model = sisa_rec.shard_models[shard]
    model.eval()
    with torch.no_grad():
        scores = model.score_user_all_items(user_idx).detach().cpu().numpy()
    order = np.argsort(-scores)
    pos = np.where(order == item_idx)[0]
    return int(pos[0]) + 1 if len(pos) > 0 else None

# ---- Load base data ----
loader = DataLoader(config)
ratings, num_users, num_items = loader.load_data()
train_df, test_df = loader.split_data(ratings, test_size=0.2)

# choose target movie (by item_idx)
target_item = random.choice(train_df['item_idx'].unique().tolist())
print("Target item_idx:", target_item)

# sample some real users to inspect
sample_users = train_df['user_idx'].drop_duplicates().sample(5, random_state=42).tolist()
print("Sample users:", sample_users)

# ---- Baseline (clean) ----
global_clean = GlobalRecommender(config, num_users, num_items, train_df, test_df)
global_clean.train()

print("\nBaseline top-10 & target rank BEFORE injection:")
for u in sample_users:
    topk = global_clean._predict_topk_for_user(u, k=10)
    r = rank_of_item_for_user_global(global_clean, u, target_item)
    print(f"User {u}: top10={topk}, target_rank={r}")

# ---- Create malicious fake users ----
N = 200  # number of fake users (increase for stronger effect)
start_new = num_users
fake_users = list(range(start_new, start_new + N))

rows = []
for u in fake_users:
    for t in range(10):  # 10 positive interactions each with target
        rows.append({
            'user_id': 999999 + u,
            'item_id': target_item,
            'rating': 5.0,
            'timestamp': 0,
            'implicit': 1,
            'user_idx': u,
            'item_idx': int(target_item)
        })

fake_df = pd.DataFrame(rows)
aug_train = pd.concat([train_df, fake_df], ignore_index=True)
new_num_users = num_users + N
print("\nNew train rows:", len(aug_train), "new num users:", new_num_users)

# ---- Baseline after injection ----
global_inj = GlobalRecommender(config, new_num_users, num_items, aug_train, test_df)
global_inj.train()

print("\nBaseline top-10 & target rank AFTER injection:")
for u in sample_users:
    topk = global_inj._predict_topk_for_user(u, k=10)
    r = rank_of_item_for_user_global(global_inj, u, target_item)
    print(f"User {u}: top10={topk}, target_rank={r}")

# ---- Naive unlearning (remove fake users, full retrain) ----
mask = ~aug_train['user_idx'].isin(fake_users)
clean_train = aug_train[mask].reset_index(drop=True)
global_unlearn = GlobalRecommender(config, num_users, num_items, clean_train, test_df)
global_unlearn.train()

print("\nBaseline top-10 & target rank AFTER naive unlearn (retrain):")
for u in sample_users:
    topk = global_unlearn._predict_topk_for_user(u, k=10)
    r = rank_of_item_for_user_global(global_unlearn, u, target_item)
    print(f"User {u}: top10={topk}, target_rank={r}")

# ---- SISA: injection + SISA unlearning ----
grouper = DataGrouper(config)
interaction_matrix = loader.create_interaction_matrix(train_df, num_users, num_items)
groups = grouper.create_groups_kmeans(interaction_matrix)

sisa = SISARecommender(config, new_num_users, num_items, aug_train, test_df, groups)
sisa.train_shards()

print("\nSISA top-10 & target rank AFTER injection (before unlearning):")
for u in sample_users:
    topk = sisa._predict_topk_for_user(u, k=10)
    r = rank_of_item_for_user_sisa(sisa, u, target_item)
    print(f"User {u}: top10={topk}, target_rank={r}")

# identify shards with fake users and unlearn those shards
fake_shards = set([sisa.user_to_shard.get(u) for u in fake_users])
print("\nFake users belong to shards:", fake_shards)
for s in fake_shards:
    if s is not None:
        users_in_shard = [u for u in fake_users if sisa.user_to_shard.get(u) == s]
        sisa.unlearn_shard(s, users_in_shard)

print("\nSISA top-10 & target rank AFTER SISA unlearning:")
for u in sample_users:
    topk = sisa._predict_topk_for_user(u, k=10)
    r = rank_of_item_for_user_sisa(sisa, u, target_item)
    print(f"User {u}: top10={topk}, target_rank={r}")

