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
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# --- Path to saved model ---
path_to_data = "/home/a.norouzikandelati/Projects/data/tillage_mapping/"

# Load model artifacts
model_artifacts = joblib.load(path_to_data + 'rf_pipeline_with_gridsearch.joblib')

preprocessor = model_artifacts['preprocessor']
low_var_filter = model_artifacts['low_var_filter']
selector = model_artifacts['selector']
grid_search = model_artifacts['grid_search']  # (this is your trained RF model)

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

# Step-by-step transformation
X_prep = preprocessor.transform(X)
X_var = low_var_filter.transform(X_prep)
X_sel = selector.transform(X_var)

# Train a shallow decision tree for interpretability
print("Training a shallow decision tree for interpretability...")
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)

tree_clf.fit(X_sel, y)

# Get final feature names
# Extract feature names from selector after all transformations
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical)
all_feature_names = list(ohe_feature_names) + numeric
low_var_feature_names = low_var_filter.get_feature_names_out(all_feature_names)
selected_feature_names = selector.get_feature_names_out(low_var_feature_names)

# Print human-readable rules
rules = export_text(tree_clf, feature_names=list(selected_feature_names))
print(rules)

X_sel_df = pd.DataFrame(X_sel, columns=selected_feature_names)
X_sel_df['reliability'] = y.values
X_sel_df.to_csv(path_to_data + 'selected_features.csv', index=False)



# 1) Subset your two categorical columns
X_small = X[['cdl_cropType', 'fr_pred']]

# 2) One‑hot encode with the new arg name
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_ohe = ohe.fit_transform(X_small)

# 3) Get feature names
feat_names = ohe.get_feature_names_out(['cdl_cropType', 'fr_pred'])

# 4) Fit your shallow tree
tree_clf = DecisionTreeClassifier(max_depth=None, random_state=42)
tree_clf.fit(X_ohe, y)

# 5) Print the rules
print(export_text(tree_clf, feature_names=list(feat_names)))

# How many High/Medium/Low you have by (crop, fr_pred)
counts = dataset.groupby(['cdl_cropType','fr_pred','reliability']) \
                .size() \
                .unstack(fill_value=0)
print(counts)


# +
import re
import pandas as pd

path_to_data = "/home/a.norouzikandelati/Projects/data/tillage_mapping/"
path_to_results_buffer = path_to_data + 'res_buffer_1.txt'

# Define regex pattern to parse rules
# Format: (conditions) => reliability=Label (X.0/Y.0)
pat = re.compile(r'^(.*)=> reliability=(\w+) \(([\d.]+)/([\d.]+)\)')

# Store parsed rules
rules = []

# Load and parse rules from JRip output file
with open(path_to_results_buffer) as f:
    for line in f:
        line = line.strip()
        match = pat.match(line)
        if match:
            conds, label, covered, errors = match.groups()
            try:
                attrs = sorted({
                    c.strip().split()[0]
                    for c in re.split(r' and ', conds)
                    if c.strip()
                })
            except IndexError:
                print(f"Skipping rule due to parsing issue: {conds}")
                continue
            rules.append({
                'label': label,
                'cover': int(float(covered)),
                'errors': int(float(errors)),
                'accuracy': round((float(covered) - float(errors)) / float(covered), 3),
                'num_conditions': len(attrs),
                'attrs': attrs,
                'conditions': conds.strip()
            })
        else:
            print(f"Line didn't match rule format: {line}")



# Create DataFrame
df = pd.DataFrame(rules)

# Sort rules by accuracy or coverage
df = df.sort_values(by=["label", "accuracy", "cover"], ascending=[True, False, False]).reset_index(drop=True)

# Display top few rules
pd.set_option('display.max_colwidth', None)  # Optional: allows viewing full rule condition
print(df.head(10))

# -

df[df.accuracy >= 0.95]

df.sort_values(by="num_conditions").head(10)


df.groupby("label").head(5)


# +
# Step 0: Convert 'attrs' to tuples (hashable)
df['attrs'] = df['attrs'].apply(tuple)

# Step 1: Group and summarize
grouped = (
    df.groupby(['attrs', 'label'])[['cover', 'errors']]
      .sum()
      .assign(
          acc=lambda d: 1 - d['errors'] / d['cover'],
          support_minus_errors=lambda d: d['cover'] - d['errors']
      )
)

# # Step 2: Sort and select top rules
# top_rules = grouped.sort_values('support_minus_errors', ascending=False).reset_index().head(20)

# Step 1: Get top 5 rules per label
top_rules = (
    grouped
    .sort_values(['label', 'support_minus_errors'], ascending=[True, False])
    .groupby('label')
    .head(5)
    .reset_index()
)

# Display
import pandas as pd
pd.set_option("display.max_colwidth", None)
print(top_rules)


# +
import itertools
import re
import pandas as pd
import numpy as np

# Ensure 'attrs' column is tuple for matching
df['attrs'] = df['attrs'].apply(tuple)

# Make sure your original condition string is available
# If not already, rename 'conditions' column as 'txt'
df['txt'] = df['conditions']

def repr_rule(sub):
    all_conditions = list(itertools.chain.from_iterable(r['txt'].split(' and ') for _, r in sub.iterrows()))
    conds_by_attr = {}

    for cond in all_conditions:
        cond = cond.strip()
        # Extract attribute, operator, threshold
        match = re.match(r'\(?([\w_>%-]+)\s*(<=|>=|<|>)\s*([-+]?[0-9]*\.?[0-9]+)', cond)
        if not match:
            continue
        attr, op, val = match.groups()
        val = float(val)
        if attr not in conds_by_attr:
            conds_by_attr[attr] = {'vals': [], 'op': op}
        conds_by_attr[attr]['vals'].append(val)

    final_conds = []
    # Define custom priority
    def attr_priority(attr):
        if attr.startswith('fr_pred'):
            return (0, attr)
        elif attr.startswith('cdl_cropType'):
            return (1, attr)
        else:
            return (2, attr)

    for attr, info in sorted(conds_by_attr.items(), key=lambda x: attr_priority(x[0])):
        vals = info['vals']
        op = info['op']
        median = np.median(vals)
        final_conds.append(f"{attr} {op} {median:.3f}")

    return ' and '.join(final_conds)

# Now apply to each row of top_rules
simplified = []
for _, row in top_rules.iterrows():
    subset = df[(df['attrs'] == row['attrs']) & (df['label'] == row['label'])]
    simplified.append({
        'label': row['label'],
        'cover': row['cover'],
        'acc': round(row['acc'], 3),
        'rule': repr_rule(subset)
    })

# Create simplified summary DataFrame
simplified_df = pd.DataFrame(simplified)
pd.set_option('display.max_colwidth', None)
print(simplified_df)

# -

simplified_df
