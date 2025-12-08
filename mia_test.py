# mia_test.py
import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from config import config
from read import DataLoader
from main import GlobalRecommender
from group import DataGrouper

def sample_negative_pairs(num_users, num_items, existing_pairs, n_samples, seed=42):
    rng = np.random.RandomState(seed)
    negs = []
    exist_set = set(existing_pairs)
    while len(negs) < n_samples:
        u = rng.randint(0, num_users)
        i = rng.randint(0, num_items)
        if (u, i) not in exist_set:
            negs.append((u, i))
    return negs

def collect_scores(global_rec, pairs):
    model = global_rec.model
    model.eval()
    users = torch.LongTensor([u for (u, _) in pairs]).to(config.device)
    items = torch.LongTensor([i for (_, i) in pairs]).to(config.device)
    with torch.no_grad():
        logits = model(users, items).detach().cpu().numpy()
    # you can also use sigmoid(logits); logits are okay as feature
    return logits.reshape(-1, 1)

if __name__ == "__main__":
    # load data
    loader = DataLoader(config)
    ratings, num_users, num_items = loader.load_data()
    train_df, test_df = loader.split_data(ratings, test_size=0.2)

    # create groups, pick one shard to delete
    grouper = DataGrouper(config)
    interaction_matrix = loader.create_interaction_matrix(train_df, num_users, num_items)
    groups = grouper.create_groups_kmeans(interaction_matrix)
    shard_idx = 0  # or pick random
    users_to_delete = list(groups[shard_idx])
    print("Using shard", shard_idx, "with", len(users_to_delete), "users for MIA/unlearning test")

    # restrict to some users (e.g., 200) to keep it lightweight
    target_users = users_to_delete[:200]

    # train global model BEFORE unlearning
    global_before = GlobalRecommender(config, num_users, num_items, train_df, test_df)
    global_before.train()

    # membership pairs = (user,item) that were in training and user in target_users
    member_rows = train_df[train_df['user_idx'].isin(target_users)]
    member_pairs_all = list(zip(member_rows['user_idx'], member_rows['item_idx']))
    # subsample to M pairs
    M = min(5000, len(member_pairs_all))
    member_pairs = member_pairs_all[:M]

    # non-member pairs: random user-item not in training for those users
    existing_pairs = set(member_pairs_all)
    neg_pairs = sample_negative_pairs(num_users, num_items, existing_pairs, M)

    print("Collected", len(member_pairs), "member pairs and", len(neg_pairs), "non-member pairs")

    # scores before unlearning
    scores_member_before = collect_scores(global_before, member_pairs)
    scores_nonmember_before = collect_scores(global_before, neg_pairs)

    X_before = np.vstack([scores_member_before, scores_nonmember_before])
    y_before = np.array([1]*len(scores_member_before) + [0]*len(scores_nonmember_before))

    # train a simple logistic regression attack model
    clf_before = LogisticRegression().fit(X_before, y_before)
    prob_before = clf_before.predict_proba(X_before)[:,1]
    auc_before = roc_auc_score(y_before, prob_before)
    print("MIA AUC BEFORE unlearning (baseline global model):", auc_before)

    # --------- Naive unlearning (full retrain without target shard users) ---------
    mask = ~train_df['user_idx'].isin(users_to_delete)
    train_after = train_df[mask].reset_index(drop=True)
    global_after = GlobalRecommender(config, num_users, num_items, train_after, test_df)
    global_after.train()

    # compute scores AFTER unlearning for the same candidate pairs
    scores_member_after = collect_scores(global_after, member_pairs)
    scores_nonmember_after = collect_scores(global_after, neg_pairs)
    X_after = np.vstack([scores_member_after, scores_nonmember_after])
    y_after = y_before  # same labels

    clf_after = LogisticRegression().fit(X_after, y_after)
    prob_after = clf_after.predict_proba(X_after)[:,1]
    auc_after = roc_auc_score(y_after, prob_after)
    print("MIA AUC AFTER naive unlearning (global retrain):", auc_after)
