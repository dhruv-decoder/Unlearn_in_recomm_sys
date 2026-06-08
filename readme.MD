# SISA-based Machine Unlearning for Recommender Systems

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

This repository explores machine unlearning in recommendation systems using the MovieLens-1M dataset. The main goal is to compare traditional full retraining against SISA-style shard-level unlearning for matrix-factorization-based recommenders.

## What is inside this repo

- Main experiment flow and MF recommendation model: [main.py](main.py)
- Dataset loading and preprocessing: [read.py](read.py)
- User sharding strategies: [group.py](group.py)
- Reproducible experiment configuration: [config.py](config.py)
- Structured experiment runner and JSON summary export: [experiment_runner.py](experiment_runner.py)
- Reusable plotting/reporting helpers: [reporting.py](reporting.py)
- Plot generation entry point: [plots.py](plots.py)
- Extra analysis scripts: [mia_test.py](mia_test.py), [malicious_injection.py](malicious_injection.py)
- Lightweight regression test: [tests/test_reporting.py](tests/test_reporting.py)

## Project goals

The project focuses on three practical questions:

1. How much time does full retraining take after forgetting a user group?
2. How much time does shard-only retraining take in a SISA-style setup?
3. How do recommendation quality and privacy-related indicators change before and after unlearning?

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run a baseline experiment

```bash
python main.py --action baseline_full --num-groups 2 --num-slices 1 --k 5
```

### 3. Run the SISA comparison experiment

```bash
python main.py --action compare_unlearn --shard-method kmeans --num-groups 5 --num-slices 2 --k 5
```

### 4. Run the full experiment sweep

```bash
python experiment_runner.py
```

This writes:

- `experiment_results.csv` for tabular summaries
- `experiment_summary.json` for structured metadata
- `logs/*.log` for raw run output
- `plots/*.png` for charts

## Plotting and reporting

The reporting flow now uses [reporting.py](reporting.py) and [plots.py](plots.py) to generate reusable charts from the saved experiment summaries.

```bash
python plots.py
```

## Testing

A small regression test is included to validate the reporting utility:

```bash
python -m unittest discover -s tests
```

## Notes on the current workflow

- The main recommendation model is a matrix factorization baseline implemented in [main.py](main.py).
- The sharding logic supports `kmeans`, `random`, and `stratified` grouping strategies in [group.py](group.py).
- The project now separates experiment execution, structured result export, and plotting into cleaner modules.

## Expected output

Running the experiment pipeline produces comparison metrics such as:

- baseline training time
- naive retrain time after deletion
- SISA train time
- SISA unlearning time
- Precision@5 / NDCG@5 before and after unlearning

This makes it easier to compare the cost and utility trade-offs of naive retraining versus shard-local unlearning.

