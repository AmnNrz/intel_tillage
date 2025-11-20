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
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from time import time
import dpp_sampler as sampler
import random
import os
from pathlib import Path

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

#Load data
BASE = Path(
    "/home/a.norouzikandelati/Projects/data/tillage_mapping/"
)

lsat_data = pd.read_csv(os.path.join(BASE / "dpp_data.csv"))

if 'tillage_true' in lsat_data.columns:
    lsat_data = lsat_data.rename(columns={'tillage_true': 'Tillage'})

tillage_idx = lsat_data.columns.get_loc('Tillage')
X = lsat_data.iloc[:, :tillage_idx]
X.columns = X.columns.astype(str)
y = lsat_data["Tillage"]

groups = X["cdl_cropType"]
combo_counts = lsat_data.groupby(["Tillage", "cdl_cropType"]).size()
# Filter out combinations with fewer than 2 instances
valid_combos = combo_counts[combo_counts >= 2].index
stratify_column = pd.DataFrame({"y": y, "cdl_cropType": groups})
# Keep only rows where the combination of y and cdl_cropType is in the valid_combos
valid_mask = stratify_column.set_index(["y", "cdl_cropType"]).index.isin(valid_combos)
X_valid = X[valid_mask]
y_valid = y[valid_mask]
groups = X_valid["cdl_cropType"]
# Perform the stratified split with the valid data
stratify_column_valid = pd.DataFrame(
    {"y": y_valid, "cdl_cropType": X_valid["cdl_cropType"]}
)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
for train_index, test_index in sss.split(X_valid, stratify_column_valid):
    X_train, X_test = X_valid.iloc[train_index], X_valid.iloc[test_index]
    y_train, y_test = y_valid.iloc[train_index], y_valid.iloc[test_index]

# Define the mapping, Map and Convert to NumPy array
til_mapping = {"NoTill-DirectSeed": 0, "MinimumTill": 1, "ConventionalTill": 2}
y_train_mapped = y_train.map(til_mapping).to_numpy()
y_test_mapped = y_test.map(til_mapping).to_numpy()

# Reminder: feat_vector is [crop type, res cov, imagery features]
data = X_train
labels = y_train_mapped
test_data = X_test
test_labels = y_test_mapped

class CustomWeightedRF(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        a=0,
        min_samples_split=None,
        bootstrap=None,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.min_samples_split = min_samples_split
        self.a = a
        # self.sample_weight_mode = sample_weight_mode  # Store the mode
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=0, **kwargs
        )

    def fit(self, X, y, **kwargs):
        # Calculate the target weights based on 'a'
        target_weights_dict = self.calculate_custom_weights(y, self.a)
        target_weights = np.array([target_weights_dict[sample] for sample in y])

        if self.a == 0:
            X_mod = X.copy()
            X_mod["cdl_cropType"] = self.add_random_noise(
                X_mod["cdl_cropType"], noise_level=0.55
            )
            feature_cols = ["cdl_cropType"]
            feature_weights = np.zeros(X_mod.shape[0])
            for col in feature_cols:
                feature_weights_dict = self.calculate_custom_weights(
                    X_mod[col].values, self.a
                )
                feature_weights += X_mod[col].map(feature_weights_dict).values
            sample_weights = target_weights * feature_weights
        else:
            X_mod = X.copy()
            feature_cols = ["cdl_cropType"]
            feature_weights = np.zeros(X_mod.shape[0])
            for col in feature_cols:
                feature_weights_dict = self.calculate_custom_weights(
                    X_mod[col].values, self.a
                )
                feature_weights += X_mod[col].map(feature_weights_dict).values
        sample_weights = target_weights * feature_weights
        self.rf.fit(X_mod, y, sample_weight=sample_weights)
        # Set the classes_ attribute
        self.classes_ = self.rf.classes_
        return self
    
    # Custom weight formula function
    def calculate_custom_weights(self, y, a):
        unique_classes, class_counts = np.unique(y, return_counts=True)
        weight_dict = {}
        sum_weight = np.sum((1 / class_counts) ** a)
        for cls, cnt in zip(unique_classes, class_counts):
            weight_dict[cls] = (1 / cnt) ** a / sum_weight
        return weight_dict
    
    def add_random_noise(self, series, noise_level=0.35):
        np.random.seed(10)  # You can choose any integer value
        noisy_series = series.copy()
        categories = series.unique()
        mask = np.random.rand(len(series)) < noise_level
        noisy_series.loc[mask] = noisy_series[mask].apply(
            lambda x: np.random.choice(categories[categories != x])
        )
        return noisy_series

    def predict(self, X, **kwargs):
        X_mod = X.copy()
        return self.rf.predict(X_mod)

    def predict_proba(self, X, **kwargs):
        X_mod = X.copy()
        return self.rf.predict_proba(X_mod)

    @property
    def feature_importances_(self):
        return self.rf.feature_importances_
#best_model = CustomWeightedRF(n_estimators=30,max_depth=30,min_samples_split=4,bootstrap=True,a=0.8)

def train_and_test(X_train, y_train, X_test, y_test):
    model = CustomWeightedRF(n_estimators=30,max_depth=30,min_samples_split=4,bootstrap=True,a=0)   
    model.fit(X_train, y_train) # train
    y_train_pred = model.predict(X_train) # predict on train set
    y_test_pred = model.predict(X_test) # predict on test set
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    return model, test_acc

def active_cp_first_round(steps=1000, target_coverage=0.9, gamma=5, alpha=4, round=0, save_csv=True):
    """
    Active DPP using conformal prediction-based uncertainty with formal calibration.
    - steps: MCMC steps for DPP sampling
    - target_coverage: desired conformal coverage level (e.g., 0.9)
    - load_previous_round: if True, load train/pool/calib splits from CSV; else start fresh
    - csv_path: path to save/load CSV with round info
    """

    print(f"Active CP-based active learning (target coverage={target_coverage})")

    global data, labels, test_data, test_labels, sub_sample_count
    csv_load_path = os.path.join(
        BASE,
        f'sampling_methods_results/active_cp_rounds/pre_active_cp_round_{round}.csv'
    )

    if round == 0:
        # --- Initialization: split calibration and pool ---

        full_indices = np.arange(labels.shape[0])
        calib_size = int(0.1 * len(full_indices))

        calib_ids = np.random.choice(full_indices, size=calib_size, replace=False)
        calib_ids_sorted = np.sort(calib_ids)
        calib_data = data.iloc[calib_ids_sorted]
        calib_labels = labels[calib_ids_sorted]

        pool_ids = np.setdiff1d(full_indices, calib_ids)
        pool_ids_sorted = np.sort(pool_ids)
        pool_data = data.iloc[pool_ids_sorted]
        pool_labels = labels[pool_ids_sorted]

        # Initialize scores for pool data
        scores = np.ones(pool_labels.shape[0])
        train_ids = []

        batches = [sub_sample_count // 10] * 10

        # Select initial batch of points using DPP sampler
        k = int(np.ceil(2.0 * batches[0] / 3))
        new_train_ids = sampler.sample_ids_mc(pool_data.values, scores, k=k,
                                             alpha=alpha, gamma=gamma,
                                             cond_ids=train_ids, steps=steps)
        train_ids = np.append(train_ids, new_train_ids).astype(int)

        new_train_ids = sampler.sample_ids_mc(pool_data.values, scores, k=batches[0] - k,
                                             alpha=alpha, gamma=0.,
                                             cond_ids=train_ids, steps=steps)
        train_ids = np.append(train_ids, new_train_ids).astype(int)

        train_data = pool_data.iloc[train_ids]
        train_labels = pool_labels[train_ids]

        # Save calibration data info for later
        calib_dict = {
            "calib_id": calib_ids_sorted.tolist(),
            "calib_label": calib_labels.tolist()
        }

    else:
        # --- Load from CSV ---
        
        full_df = pd.read_csv(csv_load_path)

        # Extract calibration indices and labels (stored as strings, convert back)
        calib_ids_sorted = full_df.loc[full_df['source'] == 'calib'].index.to_numpy()
        #calib_ids_sorted = np.array(eval(full_df.loc[full_df['source'] == 'calib', 'index'].to_list()))
        calib_labels = labels[calib_ids_sorted]
        calib_data = data.iloc[calib_ids_sorted]

        # Identify train, pool sets from loaded CSV
        train_mask = full_df['source'] == 'train'
        pool_mask = full_df['source'] == 'pool'

        train_data = full_df.loc[train_mask, data.columns].reset_index(drop=True)
        train_labels = full_df.loc[train_mask, 'true_label'].values

        pool_data = full_df.loc[pool_mask, data.columns].reset_index(drop=True)
        pool_labels = full_df.loc[pool_mask, 'true_label'].values

        # Prepare variables for sampling
        scores = np.ones(len(pool_data))
        train_ids = np.arange(len(train_data))  # Local train indices for sampler cond_ids
        batches = [sub_sample_count // 10] * 10

    # Train model on current training set, evaluate on test set (fixed)
    model, _ = train_and_test(train_data, train_labels, test_data, test_labels)

    # Conformal calibration using calibration set
    calib_probs = model.predict_proba(calib_data)
    label_to_index = {label: idx for idx, label in enumerate(model.classes_)}
    calib_indices = np.array([label_to_index[lbl] for lbl in calib_labels])
    calib_scores = 1.0 - calib_probs[np.arange(len(calib_labels)), calib_indices]

    alpha = 1.0 - target_coverage
    threshold = np.quantile(calib_scores, 1.0 - alpha)

    # Compute uncertainty for pool data
    all_probs = model.predict_proba(pool_data)
    true_class_probs = np.max(all_probs, axis=1)
    cp_scores = 1.0 - true_class_probs

    prediction_sets = all_probs >= (1.0 - threshold)
    prediction_set_sizes = prediction_sets.sum(axis=1)
    top_preds = model.predict(pool_data)

    # Build combined dataframe with all points labeled by source
    # For pool and train, add predicted info
    pool_df = pool_data.copy()
    pool_df["true_label"] = pool_labels
    pool_df["top_pred"] = top_preds
    pool_df["pred_set_size"] = prediction_set_sizes

    source_col = np.full(pool_labels.shape[0], "pool", dtype=object)
    source_col[train_ids] = "train"
    pool_df["source"] = source_col

    # Calibration data frame
    calib_df = calib_data.copy()
    calib_df["true_label"] = calib_labels
    calib_df["top_pred"] = model.predict(calib_data)
    calib_df["pred_set_size"] = np.nan  # Not defined for calib
    calib_df["source"] = "calib"

    # Test dataframe - no predictions saved here but could add if desired
    test_df = test_data.copy()
    test_df["true_label"] = test_labels
    test_df["top_pred"] = model.predict(test_data)
    test_df["pred_set_size"] = np.nan
    test_df["source"] = "test"

    # Concatenate all for collaborator
    full_df = pd.concat([pool_df, calib_df, test_df], ignore_index=True)

    # Add original indices as a column so we can track rows properly after shuffling etc.
    #full_df["index"] = (list(pool_ids_sorted) + list(calib_ids_sorted) + list(range(-1, -1 - len(test_df), -1)))  # test indices negative to distinguish

    # Save to CSV for next round or collaborator
    csv_save_path = os.path.join(
        BASE,
        f'sampling_methods_results/active_cp_rounds/active_cp_round_{round}.csv')
    if save_csv:
        full_df.to_csv(csv_save_path, index=False)

    avg_set_size = prediction_set_sizes.mean()
    _, test_acc = train_and_test(train_data, train_labels, test_data, test_labels)
    print(f"Avg. prediction set size = {avg_set_size:.2f} | Test accuracy = {test_acc:.4f}")

    return full_df

def active_cp(steps=1000, target_coverage=0.9, gamma=5, alpha=4, round=0, save_csv=True):
    """
    Active DPP using conformal prediction-based uncertainty with formal calibration.
    - steps: MCMC steps for DPP sampling
    - target_coverage: desired conformal coverage level (e.g., 0.9)
    - load_previous_round: if True, load train/pool/calib splits from CSV; else start fresh
    - csv_path: path to save/load CSV with round info
    """

    print(f"Active CP-based active learning (target coverage={target_coverage})")

    global data, labels, test_data, test_labels, sub_sample_count
    # Define the path to load/save CSV
    csv_load_path = os.path.join(
        BASE,
        f'sampling_methods_results/active_cp_rounds/pre_active_cp_round_{round}.csv'
    )

    if round == 0:
        # --- Initialization: split calibration and pool ---

        full_indices = np.arange(labels.shape[0])
        calib_size = int(0.1 * len(full_indices))

        calib_ids = np.random.choice(full_indices, size=calib_size, replace=False)
        calib_ids_sorted = np.sort(calib_ids)
        calib_data = data.iloc[calib_ids_sorted]
        calib_labels = labels[calib_ids_sorted]

        pool_ids = np.setdiff1d(full_indices, calib_ids)
        pool_ids_sorted = np.sort(pool_ids)
        pool_data = data.iloc[pool_ids_sorted]
        pool_labels = labels[pool_ids_sorted]

        # Initialize scores for pool data
        scores = np.ones(pool_labels.shape[0])
        train_ids = []

        batches = [sub_sample_count // 10] * 10

        # Select initial batch of points using DPP sampler
        k = int(np.ceil(2.0 * batches[0] / 3))
        new_train_ids = sampler.sample_ids_mc(pool_data.values, scores, k=k,
                                             alpha=alpha, gamma=gamma,
                                             cond_ids=train_ids, steps=steps)
        train_ids = np.append(train_ids, new_train_ids).astype(int)

        new_train_ids = sampler.sample_ids_mc(pool_data.values, scores, k=batches[0] - k,
                                             alpha=alpha, gamma=0.,
                                             cond_ids=train_ids, steps=steps)
        train_ids = np.append(train_ids, new_train_ids).astype(int)

        train_data = pool_data.iloc[train_ids]
        train_labels = pool_labels[train_ids]

        # Save calibration data info for later
        calib_dict = {
            "calib_id": calib_ids_sorted.tolist(),
            "calib_label": calib_labels.tolist()
        }

    else:
        # --- Load from CSV ---
        
        full_df = pd.read_csv(csv_load_path)

        # Extract calibration indices and labels (stored as strings, convert back)
        calib_ids_sorted = full_df.loc[full_df['source'] == 'calib'].index.to_numpy()
        #calib_ids_sorted = np.array(eval(full_df.loc[full_df['source'] == 'calib', 'index'].to_list()))
        calib_labels = labels[calib_ids_sorted]
        calib_data = data.iloc[calib_ids_sorted]

        # Identify train, pool sets from loaded CSV
        train_mask = full_df['source'] == 'train'
        pool_mask = full_df['source'] == 'pool'

        train_data = full_df.loc[train_mask, data.columns].reset_index(drop=True)
        train_labels = full_df.loc[train_mask, 'true_label'].values

        pool_data = full_df.loc[pool_mask, data.columns].reset_index(drop=True)
        pool_labels = full_df.loc[pool_mask, 'true_label'].values

        # Prepare variables for sampling
        scores = np.ones(len(pool_data))
        train_ids = np.arange(len(train_data))  # Local train indices for sampler cond_ids
        batches = [sub_sample_count // 10] * 10

    # Train model on current training set, evaluate on test set (fixed)
    model, _ = train_and_test(train_data, train_labels, test_data, test_labels)

    # Conformal calibration using calibration set
    calib_probs = model.predict_proba(calib_data)
    label_to_index = {label: idx for idx, label in enumerate(model.classes_)}
    calib_indices = np.array([label_to_index[lbl] for lbl in calib_labels])
    calib_scores = 1.0 - calib_probs[np.arange(len(calib_labels)), calib_indices]

    alpha = 1.0 - target_coverage
    threshold = np.quantile(calib_scores, 1.0 - alpha)

    # Compute uncertainty for pool data
    all_probs = model.predict_proba(pool_data)
    true_class_probs = np.max(all_probs, axis=1)
    cp_scores = 1.0 - true_class_probs

    prediction_sets = all_probs >= (1.0 - threshold)
    prediction_set_sizes = prediction_sets.sum(axis=1)
    top_preds = model.predict(pool_data)

    # Build combined dataframe with all points labeled by source
    # For pool and train, add predicted info
    pool_df = pool_data.copy()
    pool_df["true_label"] = pool_labels
    pool_df["top_pred"] = top_preds
    pool_df["pred_set_size"] = prediction_set_sizes
    pool_df["source"] = "pool"

    # For train
    train_df = train_data.copy()
    train_df["true_label"] = train_labels
    train_df["top_pred"] = model.predict(train_data)
    train_df["pred_set_size"] = np.nan
    train_df["source"] = "train"

    # Calibration data frame
    calib_df = calib_data.copy()
    calib_df["true_label"] = calib_labels
    calib_df["top_pred"] = model.predict(calib_data)
    calib_df["pred_set_size"] = np.nan  # Not defined for calib
    calib_df["source"] = "calib"

    # Test dataframe - no predictions saved here but could add if desired
    test_df = test_data.copy()
    test_df["true_label"] = test_labels
    test_df["top_pred"] = model.predict(test_data)
    test_df["pred_set_size"] = np.nan
    test_df["source"] = "test"

    # Concatenate all for collaborator
    full_df = pd.concat([train_df, pool_df, calib_df, test_df], ignore_index=True)

    # Add original indices as a column so we can track rows properly after shuffling etc.
    #full_df["index"] = (list(pool_ids_sorted) + list(calib_ids_sorted) + list(range(-1, -1 - len(test_df), -1)))  # test indices negative to distinguish

    # Save to CSV for next round or collaborator
    csv_save_path = os.path.join(
        BASE,
        f'sampling_methods_results/active_cp_rounds/active_cp_round_{round}.csv'
    )

    if save_csv:
        full_df.to_csv(csv_save_path, index=False)

    avg_set_size = prediction_set_sizes.mean()
    _, test_acc = train_and_test(train_data, train_labels, test_data, test_labels)
    print(f"Avg. prediction set size = {avg_set_size:.2f} | Test accuracy = {test_acc:.4f}")

    return full_df

def get_next_points(steps=1000, gamma=5, alpha=4, round=1, k=10, save_csv=False):
    # Load current data
    current_df = pd.read_csv(os.path.join(
        BASE,
        f'sampling_methods_results/active_cp_rounds/reliability_round_{round}.csv')
    )
    #filtered_df = current_df[(current_df['reliability'] == 'low') & (current_df['source'] == 'pool')]
    filtered_df = current_df[(current_df['reliability'].isin(['low', 'medium'])) & (current_df['source'] == 'pool')]
    feature_df = filtered_df[['cdl_cropType', 'fr_pred', '0', '1', '2', '3', '4', '5', '6', '7']]
    train_ids = sampler.sample_ids_mc(feature_df.values,np.ones(len(feature_df)),k=k, alpha=alpha,gamma=0.,steps=steps)
    # Map selected indices back to their true index in current_df
    selected_indices = filtered_df.index[train_ids]
    # Update source column to 'train' for selected rows
    current_df.loc[selected_indices, 'source'] = 'train'
    current_df = current_df.drop(columns=['reliability'])
    if save_csv:
        current_df.to_csv(
        os.path.join(
        BASE,
        f'sampling_methods_results/active_cp_rounds/pre_active_cp_round_{round+1}.csv'),
                index=False)
    return current_df

sub_sample_count = 100
#first_df = active_cp_first_round(round=0, save_csv=False)

current_round = 20
current_df = get_next_points(round =current_round-1, save_csv=True)
round_df = active_cp(round=current_round, save_csv=True)
