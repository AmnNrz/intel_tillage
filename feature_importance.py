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
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
import shap
from sklearn.utils import resample

# --- Helper function ---
def get_feature_names(preprocessor: ColumnTransformer) -> np.ndarray:
    output_features = []

    for name, transformer, cols in preprocessor.transformers_:
        if transformer == 'drop':
            continue
        elif transformer == 'passthrough':
            output_features.extend(cols)
        else:
            try:
                output_features.extend(transformer.get_feature_names_out())
            except:
                output_features.extend(cols)

    return np.array(output_features)


# --- Path to saved model ---
path_to_data = "/home/a.norouzikandelati/Projects/data/tillage_mapping/"

# --- Load saved components ---
model_artifacts = joblib.load(path_to_data + 'rf_pipeline_with_gridsearch.joblib')
preprocessor = model_artifacts['preprocessor']
low_var_filter = model_artifacts['low_var_filter']
selector = model_artifacts['selector']
grid_search = model_artifacts['grid_search']
best_rf = grid_search.best_estimator_

# --- Step 1: Get all feature names after preprocessing ---
all_features = get_feature_names(preprocessor)

# --- Step 2: Apply masks sequentially ---
features_after_lowvar = all_features[low_var_filter.get_support()]
features_after_selector = features_after_lowvar[selector.get_support()]

# --- Step 3: Get feature importances ---
importances = best_rf.feature_importances_

# --- Step 4: Create DataFrame ---
feat_imp_df = pd.DataFrame({
    'Feature': features_after_selector,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# --- Step 5: Plot top 20 ---
plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['Feature'][:20][::-1], feat_imp_df['Importance'][:20][::-1])
plt.xlabel("Importance")
plt.title("Top 20 Most Important Features")
plt.tight_layout()
plt.show()


# +
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

# # +
# Identify feature types
categorical = ['cdl_cropType', 'fr_pred']
numeric = [c for c in dataset.columns if c not in categorical + ['reliability']]

X = dataset.drop(columns=['reliability'])
y = dataset['reliability']

# Step 1: Sample the original X (before transformation)
X_sample_orig = resample(X, n_samples=200, random_state=42)

# Step 2: Fully transform it
X_sample_transformed = selector.transform(
    low_var_filter.transform(
        preprocessor.transform(X_sample_orig)
    )
)

# Step 3: Build DataFrame with correct feature names
X_sample_df = pd.DataFrame(X_sample_transformed, columns=features_after_selector)

# Step 4: Re-create explainer using the model and transformed DataFrame
explainer = shap.TreeExplainer(best_rf)

# Step 5: Compute SHAP values
shap_values = explainer.shap_values(X_sample_df)  # should now be shape (200, 477)

# Step 6: Plot summary for one class
shap.summary_plot(shap_values[0], X_sample_df)

# -

print("shap_values[0].shape:", shap_values[0].shape)
print("X_sample_df.shape:", X_sample_df.shape)
print("len(features_after_selector):", len(features_after_selector))


# +
# say you want the summary for class index 0
class_idx = 2

# this will be shape (200, 477)
shap_2d = shap_values[:, :, class_idx]

# now this matches X_sample_df.shape = (200, 477)
shap.summary_plot(shap_2d, X_sample_df)

