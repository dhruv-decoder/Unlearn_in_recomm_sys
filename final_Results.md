Here’s a clean, report-ready **Methodology** you can drop into your final year project (you can renumber headings to match your college format).

---

## 3. Methodology

### 3.1 Dataset and Pre-processing

We use the **MovieLens 1M** dataset, which contains user–movie ratings in the form `(user_id, item_id, rating, timestamp)`.
Explicit ratings are converted to **implicit feedback** by labelling an interaction as positive if `rating ≥ 3.5` and negative otherwise. Raw `user_id` and `item_id` values are mapped to continuous indices `user_idx` and `item_idx` for efficient embedding lookups. The data is then randomly split into **80% training** and **20% test** interactions using a fixed seed. For user grouping, we also construct a sparse user–item interaction matrix from the training data.

---

### 3.2 Baseline Recommendation Model

As the base recommender, we implement a **matrix factorization (MF)** model for implicit feedback. Each user ( u ) and item ( i ) is represented by a latent vector ( p_u, q_i ∈ ℝ^{20} ), and the relevance score is computed as the dot product ( \hat{r}_{ui} = p_u^\top q_i ). The model is trained in **PyTorch** using mini-batch gradient descent with the **binary cross-entropy with logits** loss (`BCEWithLogitsLoss`), the **Adam** optimizer (learning rate 0.01) and L2 regularization. Training is run for a fixed number of epochs over all training interactions to obtain a **single global MF model**.

---

### 3.3 Evaluation Metrics

We evaluate recommendation quality using **top-K ranking metrics** with ( K = 10 ). For each test user, scores are computed for all items, training items are filtered out, and the **top-10** items are recommended. We then compute **Precision@10**, **Recall@10** and **NDCG@10** by comparing the recommended items against the user’s held-out positive interactions in the test set. All metrics are averaged over users. In addition, we record total **training time** and, for unlearning experiments, **time required to perform unlearning**.

---

### 3.4 SISA-Style Sharding and Slicing

To enable efficient unlearning, we implement a **SISA-style training scheme** (“Sharded, Isolated, Sliced, Aggregated”). First, we derive low-dimensional user embeddings via truncated SVD on the interaction matrix and apply **k-means clustering** to partition users into **G = 5 shards**, grouping similar users together. For each shard, we then split its interactions into **S = 2 slices** ordered in time or randomly. A separate MF model is trained per shard in a **slice-wise cumulative** manner: the model is trained on slice 0 and a **checkpoint** is saved, then training continues on cumulative data (slice 0 + slice 1) and another checkpoint is stored. This produces multiple shard models and checkpoints that represent different prefixes of the shard’s data.

---

### 3.5 Unlearning Procedures

We compare two unlearning strategies when we are required to forget the data of a set of users (in our experiments, all users in one shard):

1. **Naive global unlearning**:
   All interactions of the target users are removed from the training data, and the **global MF model is retrained from scratch** on the remaining interactions. We measure the retraining time and the recommendation metrics after unlearning.

2. **SISA unlearning**:
   For the SISA setup, we **identify the shard** containing the users to be forgotten and locate the earliest slice where they appear. We then **rollback the shard model** to the checkpoint just before that slice and optionally retrain only on the remaining, unaffected interactions in subsequent slices. Only the **affected shard model** is updated; all other shard models remain unchanged. The updated SISA ensemble is then evaluated on the test set to measure the impact on utility and the time taken for shard-level unlearning.

---

### 3.6 Additional Privacy Verification (Optional)

To provide additional evidence of unlearning effectiveness, we implement two simple checks on the global model:

* A **membership inference attack (MIA)** that trains a classifier to distinguish training interactions from non-members using the model’s scores, and compares its AUC before and after unlearning.
* A **malicious data injection case study** where synthetic “fake” users are added to promote a target movie; we observe how the target’s rank in user recommendations changes after injection and after subsequent removal of the malicious data.

These analyses complement the main SISA vs naive unlearning comparison by illustrating how unlearning reduces both influence of specific users and membership leakage.


RESULTS!
We performed a simple membership inference attack on the global model, using a logistic regression classifier on model scores.

AUC before unlearning: ≈ 0.612

AUC after naive unlearning (removing shard-0 users and retraining): ≈ 0.518

Interpretation:

Before unlearning, the attack can distinguish member vs non-member pairs better than random (AUC > 0.5).

After unlearning, the AUC moves closer to 0.5, indicating that membership information for removed users becomes harder to recover.

Malicious injection - 

We select a target movie and add 200 synthetic users who heavily rate it positively.

After retraining with the injected data, the rank of this target movie improves for several real users (rank numbers get smaller).

After removing these fake users and retraining from scratch, the target movie’s rank moves back away from the top, showing that we can “forget” the influence of malicious profiles by unlearning.

