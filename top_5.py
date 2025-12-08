# plot_prec5_vs_numgroups.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("experiment_results.csv")

# Only keep numeric for numeric columns
for col in df.columns:
    if col != "method":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Choose one method to visualize (e.g. kmeans, since it's your main)
method = "kmeans"
sub = df[df["method"] == method].dropna(subset=["num_groups", "sisa_prec_before@5"])

sub = sub.sort_values("num_groups")
x = sub["num_groups"].values
y = sub["sisa_prec_before@5"].values

plt.figure(figsize=(6,4))
plt.plot(x, y, marker="o")
plt.xlabel("Number of shards (num_groups)")
plt.ylabel("Precision@5 (SISA)")
plt.title(f"Top-5 recommendation accuracy vs number of shards ({method})")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/prec5_vs_numgroups_kmeans.png")
plt.close()
print("Saved plots/prec5_vs_numgroups_kmeans.png")
