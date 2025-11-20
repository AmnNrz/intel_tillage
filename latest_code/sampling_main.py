# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os, logging, random
from datetime import datetime
import numpy as np
import pandas as pd
import sys

BASE_CODE = "/home/a.norouzikandelati/Projects/intel_tillage/latest_code/"

if str(BASE_CODE) not in sys.path:
    sys.path.append(str(BASE_CODE))

from sampling_utils import *          # must include load_and_prepare_data, CustomWeightedRF, summarize_one_method
from sampling_methods import *        # must include random_sampling, passive_dpp, active_dpp, qbc_dpp, qbc_pure, curriculum_guided_active_dpp, full_model

# -------------------------
# Repro & logging
# -------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"

os.makedirs("my_DPP_tillage/logs", exist_ok=True)
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"my_DPP_tillage/logs/run_{ts}.txt"
run_dir  = f"my_DPP_tillage/runs_csv/run_{ts}"
os.makedirs(run_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
logger = logging.getLogger("sampling")

# -------------------------
# Data / hyperparams
# -------------------------
data, labels, test_data, test_labels = load_and_prepare_data()
STEPS = 200
sub_sample_count = 200
num_runs = 10
batch_size = 10

# -------------------------
# Full model (wrapped as pseudo-batch "full")
# -------------------------
full_runs = []
for i in range(num_runs):
    m = CustomWeightedRF(n_estimators=30, max_depth=30, min_samples_split=4,
                         bootstrap=True, a=0.8, random_state=i)
    # full_model must return dict with key 'test_accuracy'
    full_dict = full_model(data, labels, test_data, test_labels, m, logger)
    full_runs.append({"full": {"test_accuracy": full_dict["test_accuracy"]}})

full_df = summarize_one_method(full_runs)  # index -> 'full'; cols -> mean, std
full_df.to_csv(os.path.join(run_dir, "full_accuracy.csv"), index=True)
logger.info(f"Saved full summary to {os.path.join(run_dir, 'full_accuracy.csv')}")

# -------------------------
# Sampling methods
# -------------------------
methods = {
    "uniform":     lambda m: random_sampling(data, labels, test_data, test_labels, m, logger, batch_size, sub_sample_count),
    "passive_dpp": lambda m: passive_dpp(data, labels, test_data, test_labels, m, logger, batch_size, sub_sample_count, steps=STEPS, alpha=4),
    #"active_dpp":  lambda m: active_dpp(data, labels, test_data, test_labels, m, logger, batch_size, sub_sample_count, steps=STEPS, gamma=5, alpha=4, num_models=10),
    "qbc_dpp":     lambda m: qbc_dpp(data, labels, test_data, test_labels, m, logger, batch_size=batch_size, sub_sample_count=sub_sample_count, steps=STEPS, gamma=5, alpha=4, num_models=10, eps=1e-8, rng_seed=None),
    #"qbc_pure":    lambda m: qbc_pure(data, labels, test_data, test_labels, m, logger, batch_size=batch_size, sub_sample_count=sub_sample_count, num_models=10, eps=1e-8, warm_start=None, rng_seed=None),
    #"cgad":        lambda m: curriculum_guided_active_dpp(data, labels, test_data, test_labels, m, logger, batch_size=batch_size, sub_sample_count=sub_sample_count, steps=STEPS),
}

results_runs = {name: [] for name in methods}

for i in range(num_runs):
    for name, runner in methods.items():
        model = CustomWeightedRF(n_estimators=30, max_depth=30, min_samples_split=4,
                                 bootstrap=True, a=0.8, random_state=i)
        results_runs[name].append(runner(model))

# -------------------------
# Save per-method summaries (mean & std, 4 d.p.)
# -------------------------
for name, runs in results_runs.items():
    df = summarize_one_method(runs)  # expects rounding internally (4 d.p.)
    out_path = os.path.join(run_dir, f"{name}_accuracy.csv")
    df.to_csv(out_path, index=True)
    logger.info(f"Saved {name} summary to {out_path}")

# -------------------------
# Combined means (including 'full' row)
# -------------------------
# --- combined means (including 'full' row) ---
# --- Combined means with FULL as a COLUMN (same value for every batch) ---
combined = []
for name, runs in results_runs.items():
    df = summarize_one_method(runs)          # has ['mean','std']
    out = df[['mean']].copy()
    out.columns = [name]                     # 'mean' -> method name
    combined.append(out)

# concat methods (no 'full' row added)
all_methods_mean = pd.concat(combined, axis=1)


# ensure numeric batch index & sort
all_methods_mean.index = all_methods_mean.index.astype(int)
all_methods_mean = all_methods_mean.sort_index()

# add FULL column (same mean for all rows)
full_mean = float(full_df.loc['full', 'mean'])   # from summarize_one_method(full_runs)
all_methods_mean['full'] = round(full_mean, 4)

# round and save
all_methods_mean = all_methods_mean.round(4)
all_methods_mean.index.name = "batch_size"
combined_path = os.path.join(run_dir, "all_methods_mean_accuracy.csv")
all_methods_mean.to_csv(combined_path)
logger.info(f"Saved combined means to {combined_path}")
# -

all_methods_mean
