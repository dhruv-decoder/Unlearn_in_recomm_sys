# plot_sharding_methods.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("experiment_results.csv")

# keep 'method' as string, convert others to numeric
for col in df.columns:
    if col != "method":
        df[col] = pd.to_numeric(df[col], errors="coerce")

num_groups_target = 5  # the plot you just showed
sub = df[df["num_groups"] == num_groups_target].dropna(
    subset=["baseline_prec@5", "sisa_prec_before@5"]
)

methods = sub["method"].tolist()
baseline = sub["baseline_prec@5"].tolist()
sisa = sub["sisa_prec_before@5"].tolist()

x = np.arange(len(methods))
width = 0.35

plt.figure(figsize=(6,4))
plt.title(f"Precision@5 by sharding method (num_groups={num_groups_target})")
plt.bar(x - width/2, baseline, width, label="Baseline prec@5")
plt.bar(x + width/2, sisa, width, label="SISA prec@5")
plt.xticks(x, methods)  # kmeans / random / stratified
plt.ylabel("Precision@5")
plt.legend()
plt.tight_layout()
plt.savefig("plots/prec5_sharding_ng5.png")
plt.close()
print("Saved plots/prec5_sharding_ng5.png")
