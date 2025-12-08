 # dashboard.py
import streamlit as st
import pandas as pd
import os
import glob

st.set_page_config(page_title="SISA Unlearning Dashboard", layout="wide")
st.title("SISA-based Machine Unlearning – MovieLens-1M")

st.markdown("""
This dashboard summarizes experiments comparing **naive full retraining** vs **SISA shard unlearning**  
for different sharding strategies (kmeans, random, stratified) and numbers of shards.
""")

# ---- Summary Table ----
if os.path.exists("experiment_results.csv"):
    df = pd.read_csv("experiment_results.csv")
    # show only key columns
    show_cols = [
        "method", "num_groups",
        "baseline_prec@5",
        "naive_unlearn_time",
        "sisa_train_time",
        "sisa_unlearn_time",
        "sisa_prec_before@5",
        "sisa_prec_after@5"
    ]
    st.subheader("Experiment summary (key metrics)")
    st.dataframe(df[show_cols].round(4).fillna("N/A"))

    st.markdown("**Interpretation:**")
    st.markdown("""
    - `baseline_prec@5`: global MF precision@5 **before** any unlearning.  
    - `naive_unlearn_time`: time to fully retrain the global model after deletion.  
    - `sisa_unlearn_time`: time to retrain only the affected shard in SISA (often 0s if checkpoint is enough).  
    - `sisa_prec_before@5` vs `sisa_prec_after@5`: SISA utility before/after unlearning.
    """)

else:
    st.warning("experiment_results.csv not found. Run `python experiment_runner.py` first.")

# ---- Plots ----
st.subheader("Plots")
plot_files = sorted(glob.glob("plots/*.png"))
if not plot_files:
    st.info("No plots found in /plots. Run `python plots.py` to generate them.")
else:
    cols = st.columns(2)
    for i, p in enumerate(plot_files):
        with cols[i % 2]:
            st.image(p, caption=os.path.basename(p), use_column_width=True)

# ---- Log viewer ----
st.subheader("Raw log viewer (for detailed run logs)")
logs = sorted(glob.glob("logs/*.log"))
if not logs:
    st.info("No log files found in /logs. Run experiments to populate logs.")
else:
    # create nicer labels
    def pretty_label(path):
        base = os.path.basename(path)
        # expect like "kmeans_ng5.log"
        if "_ng" in base:
            method = base.split("_ng")[0]
            ng = base.split("_ng")[1].split(".")[0]
            return f"{method}, {ng} shards"
        return base

    options = ["-- none --"] + [pretty_label(p) for p in logs]
    choice = st.selectbox("Select a run to inspect", options)
    if choice != "-- none --":
        # map back to path
        idx = options.index(choice) - 1
        log_path = logs[idx]
        st.caption(f"Showing log: {os.path.basename(log_path)}")
        with open(log_path, "r") as f:
            txt = f.read()
        st.code(txt[:8000])
