# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# +
"""  Reliability classifier training script
This script trains a predictive model to classify tillage reliability groups
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import joblib


# Read data
path_to_data = (
    "/home/a.norouzikandelati/Projects/data/tillage_mapping/"
)

full_data = pd.read_csv(
    path_to_data + "map_reliability_data.csv"
)

# Filter out unnecesary columns
dataset = full_data.loc[:, ['pointID', 'reliability', 'cdl_cropType', 'fr_pred'] +
                list(full_data.loc[:, 'B_S0_p0':'sti_S1_prom_p100'].columns)]

# Move pointID column as index
dataset = dataset.set_index('pointID')

# +
# Identify feature types
categorical = ['cdl_cropType', 'fr_pred']
numeric = [c for c in dataset.columns if c not in categorical + ['reliability']]

X = dataset.drop(columns=['reliability'])
y = dataset['reliability']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Encode categorical features
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical)
], remainder='passthrough')

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# Remove features with zero variance
low_var_filter = VarianceThreshold(threshold=1e-3)
X_train_var  = low_var_filter.fit_transform(X_train_prep)
X_test_var  = low_var_filter.transform(X_test_prep)

# Rank importance using an embedded feature selection 
selector = SelectFromModel(
    estimator=RandomForestClassifier(
        n_estimators=100,       # keep this small for speed
        max_depth=7,            # shallow
        n_jobs=-1,
        random_state=42
    ),
    threshold='median'          # keep only features whose importance ≥ median
)

selector.fit(X_train_var, y_train)
X_train_sel  = selector.transform(X_train_var)
X_test_sel  = selector.transform(X_test_var)

# --- Now tune just the RF on X_train_sel (much smaller) ---
rf = RandomForestClassifier(n_jobs=1, random_state=42)  # n_jobs=1 often uses less RAM

param_grid = {
    "n_estimators":      [50, 100, 200],
    "max_depth":         [10, 20, 30],
    "min_samples_split": [3, 4],
    "bootstrap":         [True, False],
    "max_features":      ['sqrt','log2']
}

search = GridSearchCV(
    rf,
    param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=1,      # run sequentially to cap peak memory
    verbose=2
)

search.fit(X_train_sel, y_train)
print("Best CV:", search.best_score_, search.best_params_)


# Save all the steps in one dictionary
model_artifacts = {
    'preprocessor': preprocessor,
    'low_var_filter': low_var_filter,
    'selector': selector,
    'grid_search': search
}


# Save to a file
joblib.dump(model_artifacts, path_to_data + 'rf_pipeline_with_gridsearch.joblib')
print("Model and pipeline saved successfully.")


# final eval
y_pred = search.predict(X_test_sel)
print(classification_report(y_test, y_pred))
