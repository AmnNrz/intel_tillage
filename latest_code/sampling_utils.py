import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
import os, re, csv

from sampling_methods import *

def load_and_prepare_dual_datasets(csv_name='dataset'):
    # -------------------------
    # CONFIG
    # -------------------------
    PATH_TO_DATA = "my_DPP_tillage/training_data/"
    FILENAME = csv_name + ".csv"
    RANDOM_STATE = 0
    TEST_SIZE = 0.30
    N_PCS = 8  # legacy '0'..'7'

    # -------------------------
    # LOAD
    # -------------------------
    lsat = pd.read_csv(PATH_TO_DATA + FILENAME)

    # -------------------------
    # ENCODE TARGETS + CATEGORICALS (maps match your original)
    # -------------------------
    til_mapping  = {"NoTill-DirectSeed": 0, "MinimumTill": 1, "ConventionalTill": 2}
    crop_mapping = {"Grain": 1, "Legume": 2, "Canola": 3}
    res_mapping  = {"0-15%": 1, "16-30%": 2, ">30%": 3}

    # Keep originals for targets before dropping
    y_til_raw = lsat['Tillage'].map(til_mapping).astype('int64')
    y_res_raw = lsat['ResidueCov'].map(res_mapping).astype('int64')

    # Drop purely non-feature columns (same as your code)
    lsat = lsat.drop(columns=['year', 'County'])

    # -------------------------
    # PCA FEATURES (exclude IDs, labels, and residue)
    # -------------------------
    num = lsat.select_dtypes(include=[np.number]).copy()
    # Never feed IDs or targets or ResidueCov into PCA
    num = num.drop(columns=['pointID'], errors='ignore')
    num = num.drop(columns=['Tillage'], errors='ignore')
    num = num.drop(columns=['ResidueCov'], errors='ignore')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(num)

    n_pcs = min(N_PCS, X_scaled.shape[1])  # up to 8
    pca = PCA(n_components=n_pcs, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X_scaled)

    pc_cols = [str(i) for i in range(n_pcs)]
    pcs_df = pd.DataFrame(pcs, columns=pc_cols, index=lsat.index)
    for i in range(N_PCS):
        name = str(i)
        if name not in pcs_df.columns:
            pcs_df[name] = 0.0
    pcs_df = pcs_df[[str(i) for i in range(N_PCS)]]

    # -------------------------
    # SHARED FEATURE MATRIX (NO ResidueCov feature)
    # -------------------------
    # Keep crop type as encoded numeric feature
    X_feat = pd.DataFrame(index=lsat.index)
    X_feat['cdl_cropType'] = lsat['cdl_cropType'].map(crop_mapping).astype('int64')
    # Concatenate PCA columns
    X_feat = pd.concat([X_feat, pcs_df], axis=1)

    # Helper to stratify & split per-target
    def stratified_split_per_target(X, y_target, groups, test_size, random_state):
        # Ensure each (target, crop) combo has >=2 samples
        combo_counts = pd.DataFrame({"y": y_target, "cdl_cropType": groups}).groupby(
            ["y", "cdl_cropType"]
        ).size()
        valid_combos = combo_counts[combo_counts >= 2].index

        stratify_df = pd.DataFrame({"y": y_target, "cdl_cropType": groups})
        valid_mask = stratify_df.set_index(["y", "cdl_cropType"]).index.isin(valid_combos)

        X_valid = X[valid_mask].reset_index(drop=True)
        y_valid = y_target[valid_mask].reset_index(drop=True)
        stratify_valid = pd.DataFrame({"y": y_valid, "cdl_cropType": X_valid["cdl_cropType"]})

        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for tr_idx, te_idx in sss.split(X_valid, stratify_valid):
            X_tr = X_valid.iloc[tr_idx]
            X_te = X_valid.iloc[te_idx]
            y_tr = y_valid.iloc[tr_idx]
            y_te = y_valid.iloc[te_idx]
        return X_tr, X_te, y_tr, y_te

    # Build both targets using the SAME shared features
    groups = X_feat["cdl_cropType"]

    # --- Dataset A: target = Tillage ---
    Xtr_til, Xte_til, ytr_til, yte_til = stratified_split_per_target(
        X_feat, y_til_raw, groups, TEST_SIZE, RANDOM_STATE
    )
    data_tillage       = Xtr_til[['cdl_cropType','0','1','2','3','4','5','6','7']].copy()
    test_data_tillage  = Xte_til[['cdl_cropType','0','1','2','3','4','5','6','7']].copy()
    labels_tillage     = ytr_til.to_numpy()
    test_labels_tillage= yte_til.to_numpy()

    # --- Dataset B: target = ResidueCov ---
    Xtr_res, Xte_res, ytr_res, yte_res = stratified_split_per_target(
        X_feat, y_res_raw, groups, TEST_SIZE, RANDOM_STATE
    )
    data_residue       = Xtr_res[['cdl_cropType','0','1','2','3','4','5','6','7']].copy()
    test_data_residue  = Xte_res[['cdl_cropType','0','1','2','3','4','5','6','7']].copy()
    labels_residue     = ytr_res.to_numpy()
    test_labels_residue= yte_res.to_numpy()

    # -------------------------
    # INFO
    # -------------------------
    print("Shared feature columns:", list(data_tillage.columns))
    print("[Tillage]    Train:", data_tillage.shape, "Test:", test_data_tillage.shape)
    print("  Class balance (train):", pd.Series(labels_tillage).value_counts().to_dict())
    print("[ResidueCov] Train:", data_residue.shape, "Test:", test_data_residue.shape)
    print("  Class balance (train):", pd.Series(labels_residue).value_counts().to_dict())

    return {
        "tillage": {
            "data": data_tillage,
            "labels": labels_tillage,
            "test_data": test_data_tillage,
            "test_labels": test_labels_tillage
        },
        "residue": {
            "data": data_residue,
            "labels": labels_residue,
            "test_data": test_data_residue,
            "test_labels": test_labels_residue
        },
        "feature_names": ['cdl_cropType'] + [str(i) for i in range(N_PCS)]
    }

def load_and_prepare_data(csv_name='dataset'):
    # -------------------------
    # CONFIG
    # -------------------------
    PATH_TO_DATA = "/home/a.norouzikandelati/Projects/data/tillage_mapping/"
    FILENAME = csv_name + ".csv"
    RANDOM_STATE = 0
    TEST_SIZE = 0.30
    N_PCS = 8  # legacy '0'..'7'

    # -------------------------
    # LOAD
    # -------------------------
    lsat_data = pd.read_csv(PATH_TO_DATA + FILENAME)

    # -------------------------
    # TARGET (encode later) and BASE COLS
    # -------------------------
    Y = lsat_data[['Tillage']].copy()
    lsat_data = lsat_data.drop(columns=['year', 'County'])
    base = lsat_data[['pointID', 'ResidueCov', 'cdl_cropType']].copy()

    # -------------------------
    # PCA ON ALL OTHER NUMERIC FEATURES (exclude IDs/label/residue)
    # -------------------------
    num = lsat_data.select_dtypes(include=[np.number]).copy()
    num = num.drop(columns=['pointID', 'ResidueCov'], errors='ignore')
    num = num.drop(columns=['Tillage'], errors='ignore')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(num)

    n_pcs = min(N_PCS, X_scaled.shape[1])  # up to 8
    pca = PCA(n_components=n_pcs, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X_scaled)

    pc_cols = [str(i) for i in range(n_pcs)]
    pcs_df = pd.DataFrame(pcs, columns=pc_cols, index=lsat_data.index)

    # pad to always have '0'..'7'
    for i in range(N_PCS):
        name = str(i)
        if name not in pcs_df.columns:
            pcs_df[name] = 0.0
    pcs_df = pcs_df[[str(i) for i in range(N_PCS)]]
    X = pd.concat([base, pcs_df], axis=1)

    # -------------------------
    # ENCODING (overwrite text cols, keep names)
    # -------------------------
    til_mapping  = {"NoTill-DirectSeed": 0, "MinimumTill": 1, "ConventionalTill": 2}
    crop_mapping = {"Grain": 1, "Legume": 2, "Canola": 3}
    res_mapping  = {"0-15%": 1, "16-30%": 2, ">30%": 3}
    X["cdl_cropType"] = X["cdl_cropType"].map(crop_mapping).astype("int64")
    X["ResidueCov"]   = X["ResidueCov"].map(res_mapping).astype("int64")
    Y["Tillage"] = Y["Tillage"].map(til_mapping).astype("int64")

    # STRATIFIED SPLIT
    y = Y["Tillage"]
    groups = X["cdl_cropType"]
    combo_counts = pd.DataFrame({"Tillage": y, "cdl_cropType": groups}).groupby(
        ["Tillage", "cdl_cropType"]
    ).size()
    valid_combos = combo_counts[combo_counts >= 2].index
    stratify_column = pd.DataFrame({"y": y, "cdl_cropType": groups})
    valid_mask = stratify_column.set_index(["y", "cdl_cropType"]).index.isin(valid_combos)
    X_valid = X[valid_mask].reset_index(drop=True)
    y_valid = y[valid_mask].reset_index(drop=True)
    stratify_column_valid = pd.DataFrame(
        {"y": y_valid, "cdl_cropType": X_valid["cdl_cropType"]}
    )
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    for train_index, test_index in sss.split(X_valid, stratify_column_valid):
        X_train = X_valid.iloc[train_index]
        X_test  = X_valid.iloc[test_index]
        y_train = y_valid.iloc[train_index]
        y_test  = y_valid.iloc[test_index]

    # -------------------------
    # LEGACY-LIKE FEATURE VECTORS
    # -------------------------
    data       = X_train[['cdl_cropType', 'ResidueCov', '0','1','2','3','4','5','6','7']]
    labels     = y_train.to_numpy()
    test_data  = X_test [['cdl_cropType', 'ResidueCov', '0','1','2','3','4','5','6','7']]
    test_labels= y_test.to_numpy()

    # -------------------------
    # INFO
    # -------------------------
    print("Train legacy shape:", data.shape, "| Test legacy shape:", test_data.shape)
    print("Columns:", list(data.columns))
    print("Class balance (train):", pd.Series(labels).value_counts().to_dict())
    print("Class balance (test):",  pd.Series(test_labels).value_counts().to_dict())
    return data, labels, test_data, test_labels


class CustomWeightedRF(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        a=0,
        min_samples_split=None,
        bootstrap=None,
        random_state=None,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.min_samples_split = min_samples_split
        self.a = a
        self.random_state = random_state
        # self.sample_weight_mode = sample_weight_mode  # Store the mode
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, **kwargs
        )

    def fit(self, X, y, **kwargs):
        # Calculate the target weights based on 'a'
        target_weights_dict = self.calculate_custom_weights(y, self.a)
        target_weights = np.array([target_weights_dict[sample] for sample in y])
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

    def predict(self, X, **kwargs):
        X_mod = X.copy()
        return self.rf.predict(X_mod)

    def predict_proba(self, X, **kwargs):
        X_mod = X.copy()
        return self.rf.predict_proba(X_mod)

    @property
    def feature_importances_(self):
        return self.rf.feature_importances_
    
def train_and_test(X_train, y_train, X_test, y_test, model, logger, print_results=False):
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    # Accuracy per tillage class
    #tillage_accs = [
    #    accuracy_score(y_test[y_test == c], y_test_pred[y_test == c])
    #    for c in np.unique(y_test)
    #]
    #macro_tillage_acc = np.mean(tillage_accs)
    # Accuracy per crop type (first column of X_test)
    #crop_col = X_test.iloc[:, 0] if hasattr(X_test, "iloc") else X_test[:, 0]
    #crop_types = np.unique(crop_col)
    #crop_accs = {
    #    c: accuracy_score(y_test[crop_col == c], y_test_pred[crop_col == c])
    #    for c in crop_types
    #}
    # Print results
    if print_results:
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        #logger.info(f"Macro-Averaged Tillage Accuracy: {macro_tillage_acc:.4f}")
        #logger.info("Tillage Class Accuracies: " + " | ".join(
        #    [f"{c}: {a:.4f}" for c, a in zip(np.unique(y_test), tillage_accs)]
        #))
        #logger.info("Crop Type Accuracies: " + " | ".join(
        #    [f"{int(c)}: {a:.4f}" for c, a in crop_accs.items()]
        #))
    return model, {
    "test_accuracy": test_acc,
    #"macro_tillage_accuracy": macro_tillage_acc,
    #"tillage_class_accuracies": dict(zip(np.unique(y_test), tillage_accs)),
    #"crop_type_accuracies": crop_accs
}


def summarize_one_method(results_runs):
    """
    results_runs: list of dicts, each like {batch_k: {'test_accuracy': float, ...}, ...}
    returns: DataFrame indexed by batch_size with columns ['mean', 'std'] (4 decimal places)
    """
    def _mean_std(a):
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return {"mean": np.nan, "std": np.nan}
        ddof = 1 if a.size > 1 else 0
        mean_val = np.mean(a)
        std_val = np.std(a, ddof=ddof)
        return {"mean": round(float(mean_val), 4), "std": round(float(std_val), 4)}

    acc = defaultdict(list)

    for run in results_runs:
        for k, d in run.items():
            acc[k].append(d["test_accuracy"])

    acc_df = pd.DataFrame({k: _mean_std(v) for k, v in acc.items()}).T.sort_index()
    acc_df.index.name = "batch_size"
    return acc_df
