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
# Import libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold

import joblib
import sys



# Set the path to the data
path_to_data = ("/home/a.norouzikandelati/Projects/data/data_tillage_mapping/")

lsat_data = pd.read_csv(path_to_data + "dataset.csv")
# Encode crop type
to_replace = {"Grain": 1, "Legume": 2, "Canola": 3}
lsat_data["cdl_cropType"] = lsat_data["cdl_cropType"].replace(to_replace)
lsat_data = lsat_data.set_index("pointID")

# Get split number
split_num = int(sys.argv[1])


# Initialize the scaler
scaler = StandardScaler()

# Apply PCA
x_imagery = lsat_data.loc[:, "B_S0_p0":]
x_imagery_scaled = scaler.fit_transform(x_imagery)

pca = PCA(n_components=0.7)
x_imagery_pca = pca.fit_transform(x_imagery_scaled)

# # Save the fitted scaler
joblib.dump(scaler, path_to_data + "best_models/multi_splits/tillage_scaler_" + f"split_{split_num}.pkl")
# Save the fitted PCA object
joblib.dump(pca, path_to_data + "best_models/multi_splits/tillage_pca_" + f"split_{split_num}.pkl")

x_imagery_pca = pd.DataFrame(x_imagery_pca)
x_imagery_pca.set_index(x_imagery.index, inplace=True)

X = pd.concat(
    [
        lsat_data["cdl_cropType"],
        lsat_data["fr_pred"],
        x_imagery_pca,
    ],
    axis=1,
)

to_replace = {"0-15%": 1, "16-30%": 2, ">30%": 3}
X["fr_pred"] = X["fr_pred"].replace(to_replace)

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
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=split_num)

for train_index, test_index in sss.split(X_valid, stratify_column_valid):
    X_train, X_test = X_valid.iloc[train_index], X_valid.iloc[test_index]
    y_train, y_test = y_valid.iloc[train_index], y_valid.iloc[test_index]


# Custom weight formula function
def calculate_custom_weights(y, a):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    weight_dict = {}
    sum_weight = np.sum((1 / class_counts) ** a)
    for cls, cnt in zip(unique_classes, class_counts):
        weight_dict[cls] = (1 / cnt) ** a / sum_weight
    return weight_dict

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
            n_estimators=self.n_estimators, max_depth=self.max_depth, **kwargs
        )

    def fit(self, X, y, **kwargs):
        # Calculate the target weights based on 'a'
        target_weights_dict = calculate_custom_weights(y, self.a)
        target_weights = np.array([target_weights_dict[sample] for sample in y])

        X_mod = X.copy()
        feature_cols = ["cdl_cropType"]
        feature_weights = np.zeros(X_mod.shape[0])
        for col in feature_cols:
            feature_weights_dict = calculate_custom_weights(
                X_mod[col].values, self.a
            )
            feature_weights += X_mod[col].map(feature_weights_dict).values

        sample_weights = target_weights * feature_weights

        self.rf.fit(X_mod, y, sample_weight=sample_weights)
        # Set the classes_ attribute
        self.classes_ = self.rf.classes_

        return self

    def predict(self, X, **kwargs):
        X_mod = X.copy()
        return self.rf.predict(X_mod)

    def predict_proba(self, X, **kwargs):
        X_mod = X.copy()
        return self.rf.predict_proba(X_mod)

    @property
    def feature_importances_(self):
        return self.rf.feature_importances_
    

# Define the cross-validation splitter with a fixed random state
cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


# Define micro and macro scoring metrics
scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}

param_grid = {
    "n_estimators": [30],
    "max_depth": [30],
    "min_samples_split": [4],
    "bootstrap": [True],
    "a": [0.8],
}


# Perform grid search with cross-validation
grid_search = GridSearchCV(
    CustomWeightedRF(),
    param_grid,
    cv = cv_splitter,
    scoring=scoring,
    verbose=0,
    refit="f1_macro",
    return_train_score=True,
)
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_

# Save the best model
joblib.dump(best_model, path_to_data + "best_models/multi_splits/tillage_best_model_" + f"split_{split_num}.pkl")

# Save the grid search object
joblib.dump(grid_search, path_to_data + "best_models/multi_splits/tillage_grid_search_" + f"split_{split_num}.pkl")

# Save X_train and y_train
X_train.to_csv(path_to_data + "best_models/train_test_splits/tillage_X_train_" + f"split_{split_num}.csv")
y_train.to_csv(path_to_data + "best_models/train_test_splits/tillage_y_train_" + f"split_{split_num}.csv")

# Save X_test and y_test
X_test.to_csv(path_to_data + "best_models/train_test_splits/tillage_X_test_" + f"split_{split_num}.csv")
y_test.to_csv(path_to_data + "best_models/train_test_splits/tillage_y_test_" + f"split_{split_num}.csv")

print("Done")

