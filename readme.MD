This repository implements a SISA-style training pipeline on the MovieLens 1M dataset.  

The data is partitioned into shards (groups); for each configuration we train a matrix factorization model and simulate targeted “forget” (unlearning) by removing all data from a chosen shard and retraining only on the remaining data.

> Note: This is a *SISA-style* implementation focused on sharding and unlearning experiments, not a full production SISA system with all optimizations from the original paper.

---

## Repository Structure

- `config.py` – configuration (paths, device, hyperparameters, number of groups, seed)  
- `read.py` – data loading and preprocessing (MovieLens 1M → implicit feedback)  
- `group.py` – creation of user groups/shards  
- `main.py` – training, evaluation, and SISA-style unlearning logic  
- `utils.py` – helper functions (metrics, seeding, saving, etc.)  
- `experiments/` – optional scripts for ablations and comparisons  
- `results/` – recommended folder to store logs/plots

---

How to Run

Basic pipeline (default config):

python config.py      # optional: check config
python read.py        # load and preprocess data
python group.py       # create groups/shards
python main.py        # train and run unlearning


Examples (if CLI arguments are enabled in main.py):

# Full model (no grouping)
python main.py --group 0 --epochs 50

# SISA-style with 5 groups
python main.py --group 5 --epochs 50

# Unlearning on a specific group 
python main.py --group 5 --epochs 30 --unlearn_group 1


Typical output looks like:

Loading data...
Number of users: 6040
Number of items: 3706
Training samples: 800167
Test samples: 200042

Creating 5 groups...

Training initial model...
Initial test accuracy: 0.4784

Performing SISA unlearning on group 1
Original training samples: 800167
Remaining training samples: 631171
Test accuracy after unlearning: 0.4727


This shows baseline performance and the effect of unlearning on a selected shard.

Demo we can show for now :

1. Full retrain vs SISA (S=5)
2. Effect of shards on accuracy: plot accuracy (S = 1,5,10)
3. Stratified vs Random sharding: stratified reduces accuracy drop on minority users.
4. A single unlearning demo : run python main.py --group 5 --epochs 10 --unlearn_group 2 and show the printed retrain time and accuracy drop.

# baseline
python main.py --group 0 --epochs 20 > results/baseline.log

# SISA (S=5)
python main.py --group 5 --epochs 20 > results/sisa5.log

# targeted unlearn group 2
python main.py --group 5 --epochs 20 --unlearn_group 2 > results/unlearn_g2.log

Ideas to work on : (GPT Just kept ; we can explore and include easy parts)

The shrad based comparison

Case studies exploration forgetting groups/users etc (to be figured out) . We need to add graphs and results 

Any original feature improvement /simple multi-agentic etc

Show a simple attacker trying to infer whether a user was in training. After unlearning, show reduced membership advantage. This or some attacks to show the difference in  movie recommendations after unlearning via metrics. (I think we should focus here)

Valuable metrics to compute & plot

Model quality: Accuracy (for your binary setup), but better: Precision@K, Recall@K, NDCG@K, HitRate@K. Use K = 5, 10.

Unlearning cost: wall-clock retrain time (seconds), CPU/GPU utilization, number of parameters retrained.

Robustness: accuracy drop after unlearning (init_acc − post_unlearn_acc).

Fairness / minority impact: per-user-group accuracy or distribution (e.g., low-activity users’ accuracy change).

Influence: how many recommendations changed per unlearn request (count of changed top-K lists).

Stability: variability across seeds (put error bars).

Good plots:

Accuracy vs Number of Shards (with error bars).

Retrain time vs Number of Shards.

Accuracy before vs after unlearning (bar chart).

Influence histogram (how many items changed when removing a user).

Per-shard performance heatmap.

Comparison table: Full retrain vs SISA vs SISA+checkpoints (if implemented).

Core / required improvements (makes your prototype solid)

Fix training loop & logging (use DataLoader, average loss, seed everything).

Per-shard models + ensemble: train a model per shard and combine predictions (avg or weighted). This is the canonical SISA flow.

Implement slicing + checkpoints: split each shard into R slices and save checkpoint after each slice. On unlearning of data in slice r, retrain only from checkpoint r. This is the main SISA optimization and your strongest demo point.

Measure retrain time precisely and log results to CSV — then make the plots.

High-value, differentiating improvements (original & demonstrative)

Stratified sharding: instead of random or purely KMeans, ensure class-balance / activity-balance per shard. Demonstrate that stratified shards reduce minority damage. (Good for report)

Prototype-based ensemble (ProtoSISA idea): compute class / item centroids from shard embeddings and use prototype distances to augment ensemble predictions to recover accuracy lost by sharding. This is a lightweight original contribution.

Targeted unlearning policy / multi-agent scheduler: implement a simple multi-agent system where each shard is an “agent” that reports a score (validation loss, time-to-retrain) and a coordinator agent decides which shards to retrain or whether to accept a forget request immediately or delay/aggregate it. This is novel & easy to present: you can show decisions & metrics in a dashboard.

Advanced but attractive ideas (show ML chops)

Transformer-based recommender option: add an experimental transformer sequential recommender (e.g., SASRec or BERT4Rec) as an alternative baseline. You can then explore whether SISA-like sharding works for sequence models. This is more work but impressive.

Influence estimation: implement an approximate influence score for users (e.g., change-in-loss heuristic) to prioritize retrains or to show which users are most harmful when deleted.

Adversarial / poisoning test: insert poisoned interactions from a malicious user and show how quickly SISA can remove the effect by targeted unlearning. Very strong demo linking privacy/security.

8) Multi-agent ideas (brief — make them visual & simple)

Shard agents: each shard exposes metrics (validation loss, last checkpoint time, size). A coordinator agent receives a forget request and decides which shard(s) to retrain first based on priority (influence + time cost). Show an interface where agents “vote” to retrain or defer.

Ensemble-weight agents: after training, each shard agent proposes a weight for ensemble predictions (based on validation performance). A small RL agent could learn weights that maximize validation accuracy — but you can start with greedy / softmax weighting.

Active unlearning agent: an agent that, given a budget (time/resources), chooses which forgotten requests to process first to minimize global harm. This is conceptual but great to present as future work.

This is something I want to include :

Compute top-K recommendations for each user before the event (baseline).

Trigger event: (a) forget a user (remove interactions) or (b) simulate attack (add malicious interactions).

Retrain (or unlearn via SISA) to get post-event model.

Compute top-K after.

Measure change with Jaccard@K between pre/post lists per user, and summarize (mean, histogram, fraction of users with Jaccard < threshold). Optionally compute rank changes or NDCG delta.
