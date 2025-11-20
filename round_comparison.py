# ---
# jupyter:
#   jupytext:
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
import pandas as pd 
import numpy as np  
import pickle
import matplotlib.pyplot as plt
import os
from pathlib import Path

BASE = Path("/home/a.norouzikandelati/Projects/data/tillage_mapping")
path_to_data = (
BASE
/ "sampling_methods_results"
)

random_path =  os.path.join(path_to_data, "random_cp_rounds")
with open(os.path.join(random_path, "test_acc_random_cp.pkl"), "rb") as f:
    test_acc_random = pickle.load(f)

full_path =  os.path.join(path_to_data, "full_cp_rounds")
with open(os.path.join(full_path, "test_acc_full_cp.pkl"), "rb") as f:
    test_acc_full = pickle.load(f)

# with open("online_cp_rounds/test_acc_online_cp.pkl", "rb") as f:
    # test_acc_online_cp = pickle.load(f)

active_path = os.path.join(path_to_data, "active_cp_rounds")
with open(os.path.join(active_path, "test_acc_active_cp.pkl"), "rb") as f:
    test_acc_active_cp = pickle.load(f)

query_path = os.path.join(path_to_data, "query_cp_rounds")
with open(os.path.join(query_path, "test_acc_query_cp.pkl"), "rb") as f:
    test_acc_query = pickle.load(f)[0]
    


def plot_all_metrics(results_dicts, method_names, k_values):
    """
    results_dicts: list of lists of result dicts, one per method
    method_names: list of method names, e.g. ["Random", "Passive DPP"]
    k_values: list of k values corresponding to subset sizes
    """

    #til_mapping = {"NoTill-DirectSeed": 0, "MinimumTill": 1, "ConventionalTill": 2}
    til_mapping = {"NT": 0, "MT": 1, "CT": 2}
    crop_mapping = {"Grain": 1, "Legumes": 2, "Canola": 3}

    # Reverse mappings: index -> name
    til_name = {v: k for k, v in til_mapping.items()}
    crop_name = {v: k for k, v in crop_mapping.items()}

    # Initialize storage
    test_acc = {}
    macro_tillage = {}
    macro_crop = {}
    tillage_per_class = {0: {}, 1: {}, 2: {}}
    crop_per_type = {1.0: {}, 2.0: {}, 3.0: {}}

    # Fill in metrics
    for method, name in zip(results_dicts, method_names):
        test_acc[name] = [res["test_accuracy"] for res in method]
        macro_tillage[name] = [res["macro_tillage_accuracy"] for res in method]
        macro_crop[name] = [
            np.mean(list(res["crop_type_accuracies"].values())) for res in method
        ]
        for cls in tillage_per_class:
            tillage_per_class[cls][name] = [
                res["tillage_class_accuracies"].get(cls, np.nan) for res in method
            ]
        for ctype in crop_per_type:
            crop_per_type[ctype][name] = [
                res["crop_type_accuracies"].get(ctype, np.nan) for res in method
            ]

    # === Plotting ===
    def plot_metric(metric_dict, title, ylabel, filename):
        plt.figure()
        for method, values in metric_dict.items():
            plt.plot(k_values, values, label=method, marker='o')
        plt.xlabel("Subset size (k)", fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.title(title, fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        # plt.savefig("../figures/" + filename, dpi=300)
        plt.show()

    # 1. General Test Accuracy
    plot_metric(test_acc, "Test Accuracy", "Accuracy", "test_accuracy.png")

    # 2. Macro-Averaged Tillage Accuracy
    plot_metric(macro_tillage, "Macro-Averaged Tillage Accuracy", "Accuracy", "test_macro_tillage_accuracy.png")

    # 3. Macro-Averaged Crop Accuracy
    plot_metric(macro_crop, "Macro-Averaged Crop Accuracy", "Accuracy", "test_macro_crop_accuracy.png")

    # 4–6. Tillage Class Accuracy with names
    for cls in tillage_per_class:
        plot_metric(
            tillage_per_class[cls],
            f"Tillage Class {til_name.get(cls, cls)} Accuracy",
            "Accuracy",
            f"test_{til_name.get(cls, cls)}_accuracy.png"
        )

    # 7–9. Crop Type Accuracy with names
    for ctype in crop_per_type:
        plot_metric(
            crop_per_type[ctype],
            f"Crop Type {crop_name.get(int(ctype), int(ctype))} Accuracy",
            "Accuracy",
            f"test_{crop_name.get(int(ctype), int(ctype))}_accuracy.png"
        )


# +
k_values = [i * 10 for i in range(1, 18)] #33

plot_all_metrics(
    results_dicts=[
        test_acc_random[:17],
        test_acc_full[:17],
        # test_acc_online_cp[:17],
        test_acc_active_cp[:17],
        test_acc_query[:17],
    ],
    method_names=[
        "Random",
        "Full",
        # "Online CP",
        "Active CP",
        "Query by Committee",
    ],
    k_values=k_values
)


# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ===================== CONFIG =====================
# Change BASE if your data root is different
BASE = Path("/home/a.norouzikandelati/Projects/data/tillage_mapping")
PATH_TO_DATA = BASE / "sampling_methods_results"

K_TRIM = 18          # show only the first 18 points
K_SCALE = 10         # subset sizes: 10, 20, 30, ...

# Pickle paths
P_RANDOM = PATH_TO_DATA / "random_cp_rounds" / "test_acc_iters_random_cp.pkl"
P_FULL   = PATH_TO_DATA / "full_cp_rounds"   / "test_acc_iters_full_cp.pkl"
P_ACTIVE = PATH_TO_DATA / "active_cp_rounds" / "test_acc_iters_active_cp.pkl"
P_QUERY  = PATH_TO_DATA / "query_cp_rounds"  / "test_acc_iters_query_cp.pkl"

OUT_PNG  = BASE / "test_accuracy_mean_sd_first18.png"
# ==================================================


# --------------- Load ----------------
def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)

test_acc_random = load_pickle(P_RANDOM)  # list of iters; each iter = list of dicts per k
test_acc_full   = load_pickle(P_FULL)
test_acc_active = load_pickle(P_ACTIVE)
test_acc_query_raw = load_pickle(P_QUERY)  # nested one level deeper


# --------------- Normalize shapes ----------------
def normalize_qbc(method):
    """
    Your QBC is: [ [ [dict_k1..kN] ],  [ [dict_k1..kN] ], ... ]
                  └─ iter ─┘  ^ extra list
    Convert to:   [ [dict_k1..kN], [dict_k1..kN], ... ]
    """
    if (
        isinstance(method, list) and len(method) > 0
        and isinstance(method[0], list) and len(method[0]) == 1
        and isinstance(method[0][0], list)
    ):
        return [it[0] for it in method]
    return method

test_acc_query = normalize_qbc(test_acc_query_raw)


def to_iters(method):
    """Ensure we always have a list of iterations."""
    return method if (isinstance(method, list) and isinstance(method[0], list)) else [method]


# --------------- Aggregation ----------------
def mean_std_per_k(method, accessor):
    """
    Compute per-k mean and std across iterations (ignoring NaNs).
    method: list of iterations; each iteration = list of per-k dicts
    accessor: function(dict) -> float
    Returns: means (L,), stds (L,)
    """
    iters = to_iters(method)
    max_k = max((len(it) for it in iters), default=0)

    means, stds = [], []
    for k_idx in range(max_k):
        vals = []
        for it in iters:
            if k_idx < len(it):
                try:
                    vals.append(accessor(it[k_idx]))
                except Exception:
                    vals.append(np.nan)
            else:
                vals.append(np.nan)
        vals = np.asarray(vals, dtype=float)
        finite = np.isfinite(vals)
        if finite.sum() == 0:
            means.append(np.nan); stds.append(np.nan)
        elif finite.sum() == 1:
            means.append(float(np.nanmean(vals))); stds.append(0.0)
        else:
            means.append(float(np.nanmean(vals))); stds.append(float(np.nanstd(vals, ddof=1)))
    return np.array(means), np.array(stds)


# --------------- Accessors ----------------
get_test_acc = lambda rec: rec["test_accuracy"] if isinstance(rec, dict) else np.nan


# --------------- Plot ----------------
def plot_test_accuracy_first18(methods, labels, k_scale=10, k_trim=18, out_png=None):
    """
    Plot per-k mean ± SD error bars for the first k_trim points.
    """
    x_full = np.arange(1, k_trim + 1) * k_scale

    plt.figure()
    for name, data in zip(labels, methods):
        means, stds = mean_std_per_k(data, get_test_acc)
        means = means[:k_trim]
        stds  = stds[:k_trim]

        if len(means) == 0 or np.all(~np.isfinite(means)):
            continue

        L = min(len(x_full), len(means))
        plt.errorbar(
            x_full[:L], means[:L], yerr=stds[:L],
            fmt='o-', capsize=6, capthick=2, elinewidth=2, label=name
        )

    plt.xlabel("Subset size (k)", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title(f"Test Accuracy — per-k Mean ± SD (first {k_trim} points)", fontsize=16)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"Saved figure to: {out_png}")
    plt.show()


if __name__ == "__main__":
    methods = [test_acc_random, test_acc_full, test_acc_active, test_acc_query]
    labels  = ["Random", "Full", "Active CP", "Query by Committee"]
    plot_test_accuracy_first18(methods, labels, k_scale=K_SCALE, k_trim=K_TRIM, out_png=OUT_PNG)


# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===================== CONFIG =====================
BASE = Path("/home/a.norouzikandelati/Projects/data/tillage_mapping")
PATH_TO_DATA = BASE / "sampling_methods_results"

# show only the first K_TRIM points (k = 10, 20, ..., K_TRIM*10)
K_TRIM  = 18
K_SCALE = 10

# Where to save figures (set to None to skip saving)
OUT_DIR = BASE / "figs_mean_sd_first18"
# ==================================================


# ---------------- I/O ----------------
def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def ensure_outdir(d):
    if d and not d.exists():
        d.mkdir(parents=True, exist_ok=True)

# ----- Load all methods -----
P_RANDOM = PATH_TO_DATA / "random_cp_rounds" / "test_acc_iters_random_cp.pkl"
P_FULL   = PATH_TO_DATA / "full_cp_rounds"   / "test_acc_iters_full_cp.pkl"
P_ACTIVE = PATH_TO_DATA / "active_cp_rounds" / "test_acc_iters_active_cp.pkl"
P_QUERY  = PATH_TO_DATA / "query_cp_rounds"  / "test_acc_iters_query_cp.pkl"

test_random = load_pickle(P_RANDOM)   # list of 20 iters; each iter=list of dicts per k
test_full   = load_pickle(P_FULL)
test_active = load_pickle(P_ACTIVE)
test_query_raw = load_pickle(P_QUERY) # nested one extra level in your files

# ---------------- Shape fixes ----------------
def normalize_qbc(method):
    """
    Your QBC shape: [ [ [dict_k1..kN] ], [ [dict_k1..kN] ], ... ]
    Convert to:     [ [dict_k1..kN],    [dict_k1..kN],    ... ]
    (If already flat like the others, leave as-is.)
    """
    if (
        isinstance(method, list) and len(method) > 0
        and isinstance(method[0], list) and len(method[0]) == 1
        and isinstance(method[0][0], list)
    ):
        return [it[0] for it in method]
    return method

test_query = normalize_qbc(test_query_raw)

def to_iters(method):
    """Ensure we always return a list of iterations."""
    return method if (isinstance(method, list) and isinstance(method[0], list)) else [method]


# ---------------- Metric extractors ----------------
def safe_mean_dict_values(d):
    """Mean over dict values; returns np.nan if dict is missing/empty."""
    if not isinstance(d, dict) or len(d) == 0:
        return np.nan
    vals = list(d.values())
    return float(np.nanmean(vals)) if len(vals) else np.nan

# accessors match your result dict keys
GET_TEST_ACC       = lambda r: r.get("test_accuracy", np.nan) if isinstance(r, dict) else np.nan
GET_MACRO_TILLAGE  = lambda r: r.get("macro_tillage_accuracy", np.nan) if isinstance(r, dict) else np.nan
GET_MACRO_CROP     = lambda r: safe_mean_dict_values(r.get("crop_type_accuracies", {})) if isinstance(r, dict) else np.nan

def GET_TILLAGE_CLASS(cls):
    return lambda r: r.get("tillage_class_accuracies", {}).get(cls, np.nan) if isinstance(r, dict) else np.nan

def GET_CROP_TYPE(ctype):
    return lambda r: r.get("crop_type_accuracies", {}).get(ctype, np.nan) if isinstance(r, dict) else np.nan


# ---------------- Aggregation ----------------
def mean_std_per_k(method, accessor):
    """
    Compute per-k mean and std across iterations (ignoring NaNs).
    method: list of iterations; each iteration=list of per-k dicts
    accessor: function(dict)->float
    Returns: means (L,), stds (L,)
    """
    iters = to_iters(method)
    max_k = max((len(it) for it in iters), default=0)
    means, stds = [], []
    for k_idx in range(max_k):
        vals = []
        for it in iters:
            if k_idx < len(it):
                try:
                    vals.append(accessor(it[k_idx]))
                except Exception:
                    vals.append(np.nan)
            else:
                vals.append(np.nan)
        vals = np.asarray(vals, dtype=float)
        finite = np.isfinite(vals)
        if finite.sum() == 0:
            means.append(np.nan); stds.append(np.nan)
        elif finite.sum() == 1:
            means.append(float(np.nanmean(vals))); stds.append(0.0)
        else:
            means.append(float(np.nanmean(vals))); stds.append(float(np.nanstd(vals, ddof=1)))
    return np.array(means), np.array(stds)


# ---------------- Plotting core ----------------
def plot_metric_first18(
    methods, labels, accessor, title, ylabel="Accuracy",
    k_trim=K_TRIM, k_scale=K_SCALE, out_dir=OUT_DIR, fname_stub=None
):
    """
    Draw per-k mean ± SD error bars for each method, trimmed to first k_trim points.
    """
    x_full = np.arange(1, k_trim + 1) * k_scale
    plt.figure()
    for name, data in zip(labels, methods):
        means, stds = mean_std_per_k(data, accessor)
        means = means[:k_trim]
        stds  = stds[:k_trim]
        if len(means) == 0 or np.all(~np.isfinite(means)):
            continue
        L = min(len(x_full), len(means))
        plt.errorbar(
            x_full[:L], means[:L], yerr=stds[:L],
            fmt='o-', capsize=6, capthick=2, elinewidth=2, label=name
        )

    plt.xlabel("Subset size (k)", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f"{title}", fontsize=16)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.tight_layout()

    if out_dir and fname_stub:
        ensure_outdir(out_dir)
        out_png = Path(out_dir) / f"{fname_stub}_first{k_trim}.png"
        plt.savefig(out_png, dpi=220)
        print(f"Saved: {out_png}")

    plt.show()


# ---------------- Run all metrics (like your old code) ----------------
if __name__ == "__main__":
    methods = [test_random, test_full, test_active, test_query]
    names   = ["Random", "Full", "Active CP", "Query by Committee"]

    # 1) Overall Test Accuracy
    plot_metric_first18(methods, names, GET_TEST_ACC,
                        title="Test Accuracy", ylabel="Accuracy",
                        fname_stub="test_accuracy")

    # 2) Macro-Averaged Tillage Accuracy
    plot_metric_first18(methods, names, GET_MACRO_TILLAGE,
                        title="Macro-Averaged Tillage Accuracy", ylabel="Accuracy",
                        fname_stub="macro_tillage_accuracy")

    # 3) Macro-Averaged Crop Accuracy (mean over crop_type_accuracies)
    plot_metric_first18(methods, names, GET_MACRO_CROP,
                        title="Macro-Averaged Crop Accuracy", ylabel="Accuracy",
                        fname_stub="macro_crop_accuracy")

    # 4–6) Tillage class accuracies (your mapping: NT=0, MT=1, CT=2)
    til_mapping = {"NT": 0, "MT": 1, "CT": 2}
    for label, cls in til_mapping.items():
        plot_metric_first18(methods, names, GET_TILLAGE_CLASS(cls),
                            title=f"Tillage Class {label} Accuracy", ylabel="Accuracy",
                            fname_stub=f"tillage_{label}_accuracy")

    # 7–9) Crop type accuracies (your mapping keys in results: 1,2,3)
    crop_mapping = {1: "Grain", 2: "Legumes", 3: "Canola"}
    for ctype, label in crop_mapping.items():
        plot_metric_first18(methods, names, GET_CROP_TYPE(ctype),
                            title=f"Crop Type {label} Accuracy", ylabel="Accuracy",
                            fname_stub=f"crop_{label}_accuracy")

