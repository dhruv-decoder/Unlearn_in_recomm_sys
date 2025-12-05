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
