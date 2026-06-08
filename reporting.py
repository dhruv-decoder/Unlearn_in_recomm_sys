import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_experiment_summary(path='experiment_results.csv'):
    """Load experiment results and coerce numeric columns safely."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run experiment_runner.py first.")

    df = pd.read_csv(path)
    for col in df.columns:
        if col != 'method':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def save_plot(fig, filename, folder='plots'):
    os.makedirs(folder, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, filename), dpi=150)
    plt.close(fig)


def plot_precision_comparison(df, output_name='precision_comparison.png'):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ng in sorted(df['num_groups'].dropna().unique()):
        sub = df[df['num_groups'] == ng].dropna(subset=['baseline_prec@5', 'sisa_prec_before@5'])
        if sub.empty:
            continue
        x = np.arange(len(sub))
        ax.bar(x - 0.18, sub['baseline_prec@5'], width=0.18, label=f'Baseline ng={int(ng)}')
        ax.bar(x + 0.18, sub['sisa_prec_before@5'], width=0.18, label=f'SISA ng={int(ng)}')
        ax.set_xticks(x)
        ax.set_xticklabels(sub['method'].tolist(), rotation=30, ha='right')
    ax.set_ylabel('Precision@5')
    ax.set_title('Precision@5 by sharding method')
    ax.legend(loc='best')
    save_plot(fig, output_name)


def plot_unlearning_time(df, output_name='unlearning_time_comparison.png'):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sub = df.dropna(subset=['naive_unlearn_time', 'sisa_unlearn_time'])
    labels = [f"{m}-{int(ng)}" for m, ng in zip(sub['method'], sub['num_groups'])]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, sub['naive_unlearn_time'], width=0.18, label='Naive retrain')
    ax.bar(x + 0.18, sub['sisa_unlearn_time'], width=0.18, label='SISA unlearn')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Time (s)')
    ax.set_title('Unlearning time: naive vs SISA')
    ax.legend()
    save_plot(fig, output_name)


def plot_num_groups_vs_precision(df, output_name='num_groups_vs_precision.png'):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in sorted(df['method'].dropna().unique()):
        sub = df[df['method'] == method].dropna(subset=['num_groups', 'sisa_prec_before@5']).sort_values('num_groups')
        if sub.empty:
            continue
        ax.plot(sub['num_groups'], sub['sisa_prec_before@5'], marker='o', label=method)
    ax.set_xlabel('Number of shards')
    ax.set_ylabel('SISA Precision@5')
    ax.set_title('Number of groups vs SISA Precision@5')
    ax.legend()
    save_plot(fig, output_name)
