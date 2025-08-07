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

#Load data
path_to_data = ("my_DPP_tillage/training_data/")
lsat_data = pd.read_csv(path_to_data + "dpp_data.csv")

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
data = X_train[['cdl_cropType', 'fr_pred', '0', '1', '2', '3', '4', '5', '6', '7']]
labels = y_train_mapped
test_data = X_test[['cdl_cropType', 'fr_pred', '0', '1', '2', '3', '4', '5', '6', '7']]
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

#best_model = CustomWeightedRF(n_estimators=30,max_depth=30,min_samples_split=4,bootstrap=True,a=0.8)

def train_and_test_old(X_train, y_train, X_test, y_test):
    model = CustomWeightedRF(n_estimators=30,max_depth=30,min_samples_split=4,bootstrap=True,a=0.8)   
    model.fit(X_train, y_train) # train
    y_train_pred = model.predict(X_train) # predict on train set
    y_test_pred = model.predict(X_test) # predict on test set
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    return model, test_acc

def train_and_test(X_train, y_train, X_test, y_test, print_results=True):
    model = CustomWeightedRF(n_estimators=30, max_depth=30, min_samples_split=4, bootstrap=True, a=0.8)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    # Accuracy per tillage class
    tillage_accs = [
        accuracy_score(y_test[y_test == c], y_test_pred[y_test == c])
        for c in np.unique(y_test)
    ]
    macro_tillage_acc = np.mean(tillage_accs)
    # Accuracy per crop type (first column of X_test)
    crop_col = X_test.iloc[:, 0] if hasattr(X_test, "iloc") else X_test[:, 0]
    crop_types = np.unique(crop_col)
    crop_accs = {
        c: accuracy_score(y_test[crop_col == c], y_test_pred[crop_col == c])
        for c in crop_types
    }
    # Print results
    if print_results:
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Macro-Averaged Tillage Accuracy: {macro_tillage_acc:.4f}")
        print("Tillage Class Accuracies: " + " | ".join(
            [f"{c}: {a:.4f}" for c, a in zip(np.unique(y_test), tillage_accs)]
        ))
        print("Crop Type Accuracies: " + " | ".join(
            [f"{int(c)}: {a:.4f}" for c, a in crop_accs.items()]
        ))
    return model, {
    "test_accuracy": test_acc,
    "macro_tillage_accuracy": macro_tillage_acc,
    "tillage_class_accuracies": dict(zip(np.unique(y_test), tillage_accs)),
    "crop_type_accuracies": crop_accs
}

# Defines a method for training on the entire training set.
def full_model():
    print("Full")
    model, test_acc = train_and_test(data, labels, test_data, test_labels)
    #print(f'Accuracy of the model on the test images:{round(test_acc,4)}')
    return test_acc

# Defines a method for Uniform Sampling.
def random_subset(k=10):
    print(f'\033[93mUniform with {k} samples \033[0m')
    sub_ids = np.random.permutation(labels.shape[0])
    train_ids = sub_ids[:k]
    model, test_acc = train_and_test(data.iloc[train_ids], labels[train_ids], test_data, test_labels)
    #print(f'Accuracy of the model on the test images:{round(test_acc,4)}')
    #draw(data[train_ids].numpy())
    return test_acc

# Defines a method for Passive DPP (steps = 1000) and Passive Greedy-DPP (steps = 0), where steps is the 
# number of Monte Carlo steps.
def passive_dpp(steps, gamma=5, alpha=4, k=10):
    print(f'\033[93mPassive DPP with {k} samples \033[0m')
    train_ids = sampler.sample_ids_mc(data.values, np.ones(labels.shape[0]), k=k, alpha=alpha, gamma=0., steps=steps)
    model, test_acc = train_and_test(data.iloc[train_ids], labels[train_ids], test_data, test_labels)
    #print(f'Accuracy of the model on the test images:{round(test_acc,4)}')
    #draw(data[train_ids].numpy())
    return test_acc

# Defines a method for Active DPP (steps = 1000) and Active Greedy-DPP (steps = 0), where steps is the 
# number of Monte Carlo steps.
def active_dpp(steps, gamma=5, alpha=4):
    if steps == 0:
        print("Active Greedy-DPP")
    else:
        print("Active DPP")
    scores = np.ones(labels.shape[0])
    train_ids = []
    
    # Select batches of training points in ten iterations.
    batches = [sub_sample_count//10]*10

    for i in range(len(batches)):
        # Select 2/3 of batch for exploitation and 1/3 of batch for exploration.
        k = int(np.ceil(2.0*batches[i]/3))
        # select 2/3 for exploitation i.e. pts with High uncertainty 
        new_train_ids = sampler.sample_ids_mc(data.values, scores, k=k, alpha=alpha, gamma=gamma, cond_ids=train_ids, steps=steps)
        train_ids = np.append(train_ids, new_train_ids).astype(int)
        # select 1/3 for exploration i.e., pts with least similarity => diversity maximization
        new_train_ids = sampler.sample_ids_mc(data.values, scores, k=batches[i]-k, alpha=alpha, gamma=0., cond_ids=train_ids, steps=steps)
        train_ids = np.append(train_ids, new_train_ids).astype(int)

        mask = np.full(labels.shape[0], True, dtype=bool)
        mask[train_ids] = False
        remaining_ids = np.arange(labels.shape[0])[mask]

        # Estimate uncertainty of each training point with an ensemble of ten neural networks.
        num_models = 10
        out = np.zeros((num_models,data.shape[0],num_classes))
        models = []
        for j in range(num_models):
            print(f'\033[93mActive DPP model {j+1} of {num_models} in round {i+1} of {len(batches)} \033[0m')
            m, test_acc = train_and_test(data.iloc[train_ids], labels[train_ids], data.iloc[remaining_ids], labels[remaining_ids], print_results=False)
            models.append(m)
            out[j,:,:] = models[j].predict_proba(data)
            a = 2
        
        # Generate Shannon information entropy for each training point.
        out = np.mean(out, axis=0)
        scores = np.sum(-out*np.log(out),axis=1) # Shannon entropy applied across all points to quantify model uncertainty
        scores[scores!=scores] = 0 # because 0*log(0)=0
        scores = scores + 1e-8 # just to make sure condition is not zero-probability

    model, test_acc = train_and_test(data.iloc[train_ids], labels[train_ids], test_data, test_labels)
    print(f'Accuracy of the model on the test images:{round(test_acc,4)}')
    #draw(data[train_ids].numpy())
    return test_acc

def query_by_committee(steps, gamma=5, alpha=4):
    print("Queery by Committee ")

    scores = np.ones(labels.shape[0])
    train_ids = []
    
    batch_size = 10
    batches = [batch_size] * (sub_sample_count // batch_size)
    # If sub_sample_count is not divisible by 10, add the remainder as an extra batch
    remainder = sub_sample_count % batch_size
    if remainder:
        batches.append(remainder)
    query_accs = []

    for i in range(len(batches)):
        # select all batch for exploitation i.e. pts with High uncertainty 
        new_train_ids = sampler.sample_ids_mc(data.values, scores, k=batches[i], alpha=alpha, gamma=gamma, cond_ids=train_ids, steps=steps)
        train_ids = np.append(train_ids, new_train_ids).astype(int)
        mask = np.full(labels.shape[0], True, dtype=bool)
        mask[train_ids] = False
        remaining_ids = np.arange(labels.shape[0])[mask]
        # Estimate uncertainty of each training point with an ensemble of ten neural networks.
        num_models = 10
        out = np.zeros((num_models,data.shape[0],num_classes))
        models = []
        for j in range(num_models):
            print(f'Active DPP model {j+1} of {num_models} in round {i+1} of {len(batches)}')
            m, test_acc = train_and_test(data.iloc[train_ids], labels[train_ids], data.iloc[remaining_ids], labels[remaining_ids], print_results=False)
            models.append(m)
            out[j,:,:] = models[j].predict_proba(data)
        # Generate Shannon information entropy for each training point.
        out = np.mean(out, axis=0)
        scores = np.sum(-out*np.log(out),axis=1) # Shannon entropy applied across all points to quantify model uncertainty
        scores[scores!=scores] = 0 # because 0*log(0)=0
        scores = scores + 1e-8 # just to make sure condition is not zero-probability
        _, test_acc = train_and_test(data.iloc[train_ids], labels[train_ids], test_data, test_labels)
        round_acc = test_acc['test_accuracy']
        print(f'\033[93mTest accuracy in round {i+1} of {len(batches)} is {round_acc} \033[0m')
        query_accs.append(test_acc)
    print(f'active query by committee accuracy done.')
    return query_accs


# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"

# Set hyperparameters
input_size = data.shape[1]
num_classes = 3
max_sample_count = data.shape[0]
sub_sample_count = 330
num_trials = 2 # 33

# store results
test_acc_full = []
test_acc_random = []
test_acc_passive = []
test_acc_active = []
test_acc_query = []

for i in range(1,num_trials):
    '''test_acc_full.append(full_model())
    print('-'*50)
    test_acc_random.append(random_subset(k=i*10))
    print('-'*50)
    # passive dpp
    test_acc_passive.append(passive_dpp(1000, k=i*10))
    print('-'*50)
    # active dpp
    test_acc_active.append(active_dpp(1000))'''
    # query by committee
    test_acc_query.append(query_by_committee(1000))
aew = 3 