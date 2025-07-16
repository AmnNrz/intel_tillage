# ---
# jupyter:
#   jupytext:
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
import os
import numpy as np
from sklearn.metrics import confusion_matrix


path_to_data = "/home/a.norouzikandelati/Projects/data/tillage_mapping/"

round_num = 2  # Change this to the desired round number
file_name = f"active_cp_round_{round_num}.csv"

# Load the data
data = pd.read_csv(os.path.join(path_to_data, file_name))


pool = data.loc[data['source'] == 'pool'].copy()
data_to_concat = data.loc[~(data['source'] == 'pool')].copy()
data_to_concat['reliability'] = np.nan

# Rename tillage columns
rename_dict = {
    0: "CT", 
    1: "MT", 
    2: "NT"
}
pool['true_label'] = pool['true_label'].astype(int).map(rename_dict)
pool['top_pred'] = pool['top_pred'].astype(int).map(rename_dict)

# 2) Make sure labels are in a consistent order
all_classes = sorted(pool['true_label'].unique())

# 3) Compute producer’s accuracy for each class, at each pred_set_size
metrics = []
for size, group in pool.groupby('pred_set_size'):
    y_true = group['true_label']
    y_pred = group['top_pred']
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    for i, cls in enumerate(all_classes):
        row_sum = cm[i, :].sum()
        col_sum = cm[:, i].sum()
        prod_acc = cm[i, i] / row_sum if row_sum else 0.0
        usr_acc = cm[i, i] / col_sum if col_sum != 0 else 0.0
        metrics.append({
            'pred_set_size': size,
            'true_label': cls,
            'acc': usr_acc
        })

metrics_df = pd.DataFrame(metrics)

# Compute 1/3 and 2/3 accuracy quantiles  
q1 = metrics_df['acc'].quantile(1/3)
q2 = metrics_df['acc'].quantile(2/3)

# define ranges
bins = [0, q1, q2, 1]
labels = ['low', 'medium', 'high']
metrics_df['reliability'] = pd.cut(metrics_df['acc'], bins=bins, labels=labels, include_lowest=True)

# Apply the new reliability rules to the data in the pool
# Merge the rules from metrics_df with data_to_concat
merged = pd.merge(
    pool,
      metrics_df[['pred_set_size', 'true_label', 'reliability']], on=['pred_set_size', 'true_label'],
        how='left')

# Prepare data to save 
# Rename tillage columns
rename_dict = {
    "CT": 0, 
    "MT": 1, 
    "NT": 2
}
merged['true_label'] = merged['true_label'].map(rename_dict)
merged['top_pred'] = merged['top_pred'].map(rename_dict)

df = pd.concat([merged, data_to_concat], ignore_index=True)
df = df.reset_index(drop=True)

df.to_csv(
    path_to_data +
    f"reliability_round_{round_num}.csv", index=False
)
