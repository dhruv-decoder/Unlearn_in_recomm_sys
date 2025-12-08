# experiment_runner.py
import subprocess
import csv
import time
import os
import re

methods = ["kmeans", "random", "stratified"]
num_groups_list = [2, 5, 10]
k = 5   # top-k for analysis (you asked top-5)

os.makedirs("logs", exist_ok=True)

def safe_search_float(pattern, text):
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except:
        return None

def safe_search_metric_dict(name, text, key):
    """
    Attempts to find a printed dict snippet like:
    "Baseline metrics (k=10): {'precision@10': 0.153... 'ndcg@10': 0.16}"
    or "Baseline metrics before: {...}"
    Returns float or None.
    """
    # find the dict-like line by searching for the line that contains name
    lines = [l for l in text.splitlines() if name in l]
    if not lines:
        return None
    line = lines[-1]
    # find "'precision@{k}': <number>"
    m = re.search(rf"'{key}':\s*([0-9]*\.?[0-9]+)", line)
    if m:
        return float(m.group(1))
    # try JSON-like numeric in parentheses or other formats
    m2 = re.search(rf"{key}[:=]\s*([0-9]*\.?[0-9]+)", line)
    if m2:
        return float(m2.group(1))
    return None

with open("experiment_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["method", "num_groups",
                     "baseline_train_time",
                     "baseline_prec@5", "baseline_ndcg@5",
                     "naive_unlearn_time",
                     "naive_prec_after@5", "naive_ndcg_after@5",
                     "sisa_train_time",
                     "sisa_prec_before@5", "sisa_ndcg_before@5",
                     "sisa_unlearn_time",
                     "sisa_prec_after@5", "sisa_ndcg_after@5"])
    for method in methods:
        for ng in num_groups_list:
            print(f"\n=== Running {method} with {ng} shards ===")
            cmd = ["python", "main.py", "--action", "compare_unlearn",
                   "--shard-method", method, "--num-groups", str(ng), "--num-slices", "2", "--k", str(k)]
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            t1 = time.time()
            out = proc.stdout + "\n" + proc.stderr

            # write full log
            logfile = f"logs/{method}_ng{ng}.log"
            with open(logfile, "w") as lf:
                lf.write(out)

            # parse with safe functions
            baseline_train_time = safe_search_float(r"Baseline training time: ([0-9\.]+)s", out)
            naive_unlearn_time = safe_search_float(r"Baseline naive-unlearn retrain time: ([0-9\.]+)s", out)
            sisa_train_time = safe_search_float(r"\[SISA\] Training time \(all shards\): ([0-9\.]+)s", out)
            sisa_unlearn_time = safe_search_float(r"SISA shard-unlearn time: ([0-9\.]+)s", out)

            baseline_prec = safe_search_metric_dict("Baseline metrics", out, f"precision@{k}")
            baseline_ndcg = safe_search_metric_dict("Baseline metrics", out, f"ndcg@{k}")

            naive_prec_after = safe_search_metric_dict("Baseline metrics after", out, f"precision@{k}")
            naive_ndcg_after = safe_search_metric_dict("Baseline metrics after", out, f"ndcg@{k}")

            sisa_prec_before = safe_search_metric_dict("SISA metrics before", out, f"precision@{k}")
            sisa_ndcg_before = safe_search_metric_dict("SISA metrics before", out, f"ndcg@{k}")
            sisa_prec_after = safe_search_metric_dict("SISA metrics after", out, f"precision@{k}")
            sisa_ndcg_after = safe_search_metric_dict("SISA metrics after", out, f"ndcg@{k}")

            # give a short parse report
            print("Parsed:", {
                "baseline_train_time": baseline_train_time,
                "naive_unlearn_time": naive_unlearn_time,
                "sisa_train_time": sisa_train_time,
                "sisa_unlearn_time": sisa_unlearn_time,
                "baseline_prec": baseline_prec
            })

            writer.writerow([method, ng,
                             baseline_train_time,
                             baseline_prec, baseline_ndcg,
                             naive_unlearn_time,
                             naive_prec_after, naive_ndcg_after,
                             sisa_train_time,
                             sisa_prec_before, sisa_ndcg_before,
                             sisa_unlearn_time,
                             sisa_prec_after, sisa_ndcg_after])
            f.flush()

print("Done. Results saved to experiment_results.csv and logs/")

