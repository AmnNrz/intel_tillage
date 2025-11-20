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

import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from pathlib import Path
import os

method = 'active' # 'active' or 'online'
# # Load the csv file from the folder active_CP_rounds 
# path1 = f'{method}_cp_rounds/{method}_cp_round_2.csv'
# df_1 = pd.read_csv(path1)
# #df_1[df_1['source'] == 'train']
# #df_1.shape[0]  # Total number of points in the first round

# +
def extract_accuracy_from_df(df):
    y_true = df["true_label"].values
    y_pred = df["top_pred"].values
    crop_types = df["cdl_cropType"].values

    # Overall test accuracy
    test_acc = accuracy_score(y_true, y_pred)

    # Accuracy per tillage class
    tillage_acc = {}
    for label in np.unique(y_true):
        idx = y_true == label
        tillage_acc[label] = accuracy_score(y_true[idx], y_pred[idx])

    macro_tillage_acc = np.mean(list(tillage_acc.values()))

    # Accuracy per crop type
    crop_acc = {}
    for crop in np.unique(crop_types):
        idx = crop_types == crop
        crop_acc[crop] = accuracy_score(y_true[idx], y_pred[idx])

    macro_crop_acc = np.mean(list(crop_acc.values()))

    return {
        "test_accuracy": test_acc,
        "macro_tillage_accuracy": macro_tillage_acc,
        "tillage_class_accuracies": tillage_acc,
        "crop_type_accuracies": crop_acc
    }

BASE = Path("/home/a.norouzikandelati/Projects/data/tillage_mapping")
path_to_data = (
BASE
/ "sampling_methods_results"
)

iter_dfs = []
for iter in np.arange(1, 21):
    test_acc_active_cp = []
    print(f"iter: {iter}")
    for i in range(1, 21):
        if i ==1:
            df = pd.read_csv(os.path.join(path_to_data, f"{method}_cp_rounds/{method}_cp_round_{i}.csv"))
            test_df = df[df["source"] == "test"]
            acc_dict = extract_accuracy_from_df(test_df)
            test_acc_active_cp.append(acc_dict)
        else:
            df = pd.read_csv(os.path.join(path_to_data, f"{method}_cp_rounds/{method}_cp_iter_{iter}_round_{i}.csv"))
            test_df = df[df["source"] == "test"]
            acc_dict = extract_accuracy_from_df(test_df)
            test_acc_active_cp.append(acc_dict)

    iter_dfs.append(test_acc_active_cp)


# +
# with open(os.path.join(path_to_data, f'{method}_cp_rounds/test_acc_{method}_cp.pkl'), "wb") as f:
#     pickle.dump(test_acc_active_cp, f)

with open(os.path.join(path_to_data, f'{method}_cp_rounds/test_acc_iters_{method}_cp.pkl'), "wb") as f:
    pickle.dump(iter_dfs, f)
