
This project investigated the problem of machine unlearning in recommendation systems,
focusing on how user data can be efficiently removed while preserving recommendation quality,
privacy, and overall system stability. Using the MovieLens-1M dataset, we compared exact
unlearning methods based on SISA-style retraining with approximate optimization-based
approaches such as Gradient Ascent (GA) and Gradient Difference (GradDiff) across both
Matrix Factorization (MF) and transformer-based SASRec recommendation models.

The experimental results reveal a clear trade-off between privacy guarantees, computational
efficiency, and recommendation stability. Exact unlearning methods such as full retraining and
SISA successfully remove the influence of forgotten data, but often introduce high
computational overhead and significant disruption to recommendation behavior. In particular,
user-level drift analysis showed that these methods frequently alter recommendation lists for a
large number of users, including users unrelated to the deletion request. The fairness analysis
further demonstrated that low-activity users are disproportionately affected by exact unlearning
approaches, indicating that data removal can create uneven impacts across different user groups.

Approximate unlearning methods produced more balanced results overall. Gradient Ascent
consistently achieved strong privacy performance while maintaining stable recommendation
quality and substantially lower unlearning time compared to exact retraining methods. The
results also showed that transformer-based sequential recommendation models respond
particularly well to approximate unlearning. Among all evaluated configurations, SASRec
combined with Gradient Ascent achieved the lowest Membership Inference Attack (MIA) AUC
while preserving competitive recommendation accuracy, suggesting that optimization-based
forgetting methods are well suited for modern deep recommendation architectures.
The experiments additionally highlight that evaluating unlearning solely through aggregate
recommendation accuracy is insufficient. Metrics such as Jaccard@10 similarity,
recommendation drift, and activity-group fairness provide important insight into how unlearning
affects real user experience. These analyses revealed that some methods can preserve overall
accuracy while still causing large behavioral changes at the user level.

Overall, the findings suggest that approximate unlearning methods currently offer the most
practical solution for recommendation systems, achieving strong privacy preservation with lower
computational cost and reduced recommendation disruption. While exact approaches such as
SISA remain theoretically appealing due to their stronger retraining guarantees, their scalability
limitations and user-level side effects present significant challenges for deployment in large-
scale recommendation platforms.

## How to Run

### Basic pipeline (default config):

```bash
python config.py      # optional: check config
python read.py        # load and preprocess data
python group.py       # create groups/shards
python main.py        # train and run unlearning
````

### Examples :

```bash
# Full model (no grouping)
python main.py --group 0 --epochs 50

# SISA-style with 5 groups
python main.py --group 5 --epochs 50

# Unlearning on a specific group 
python main.py --group 5 --epochs 30 --unlearn_group 1
```

### Output I got for baseline

```text
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
```

This shows baseline performance and the effect of unlearning on a selected shard.


### Run Commands for Demo

```bash
# baseline
python main.py --group 0 --epochs 20 > results/baseline.log

# SISA (S=5)
python main.py --group 5 --epochs 20 > results/sisa5.log

# targeted unlearn group 2
python main.py --group 5 --epochs 20 --unlearn_group 2 > results/unlearn_g2.log
```

Notebook attached for reference for GA method 

