# plots.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

fn = "experiment_results.csv"
if not os.path.exists(fn):
    print("experiment_results.csv not found. Run experiment_runner.py first.")
    raise SystemExit(1)

df = pd.read_csv(fn)

# Convert "None"/strings to NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

os.makedirs("plots", exist_ok=True)

# 1) Precision@5 vs method (per num_groups)
for ng in sorted(df['num_groups'].dropna().unique()):
    sub = df[df['num_groups'] == ng].dropna(subset=['baseline_prec@5', 'sisa_prec_before@5'])
    if sub.empty:
        continue

    plt.figure(figsize=(6,4))
    plt.title(f"Precision@5 by sharding method (num_groups={int(ng)})")

    methods = sub['method']
    precs = sub['baseline_prec@5']
    sisa_prec = sub['sisa_prec_before@5']

    x = np.arange(len(methods))
    width = 0.35
    plt.bar(x - width/2, precs, width, label='Baseline prec@5')
    plt.bar(x + width/2, sisa_prec, width, label='SISA prec@5')
    plt.xticks(x, methods)
    plt.ylabel("Precision@5")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/prec_vs_method_ng{int(ng)}.png")
    plt.close()

# 2) Unlearning time: naive vs SISA
plt.figure(figsize=(8,5))
sub = df.dropna(subset=['naive_unlearn_time', 'sisa_unlearn_time'])
groups = sub.groupby(['method', 'num_groups'])

labels, naive_times, sisa_times = [], [], []
for (m, ng), g in groups:
    row = g.iloc[0]
    labels.append(f"{m}-{int(ng)}")
    naive_times.append(row['naive_unlearn_time'])
    sisa_times.append(row['sisa_unlearn_time'])

if labels:
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, naive_times, width, label='Naive retrain (s)')
    plt.bar(x + width/2, sisa_times, width, label='SISA unlearn (s)')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Time (s)')
    plt.title('Unlearning time: naive vs SISA')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/unlearning_time_comparison.png")
    plt.close()

# 3) num_groups vs Precision@5 (SISA)
plt.figure(figsize=(6,4))
for method in df['method'].dropna().unique():
    sub = df[df['method'] == method].dropna(subset=['num_groups','sisa_prec_before@5'])
    if sub.empty:
        continue
    sub = sub.sort_values('num_groups')
    plt.plot(sub['num_groups'], sub['sisa_prec_before@5'], marker='o', label=method)

plt.xlabel("num_groups")
plt.ylabel("SISA Precision@5")
plt.legend()
plt.title("num_groups vs Precision@5 (SISA)")
plt.tight_layout()
plt.savefig("plots/num_groups_vs_prec.png")
plt.close()

print("Saved plots to plots/*.png")
