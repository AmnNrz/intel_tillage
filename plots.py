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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path_to_data = "/home/a.norouzikandelati/Projects/data/tillage_mapping/"

# +
# Load data
all_years_tsne_results = []
for year in [2012, 2017, 2022]:
    tsne_results = pd.read_csv(path_to_data + f"map_groundt_tsne_results_{year}.csv")
    all_years_tsne_results.append(tsne_results)
all_years_tsne_results = pd.concat(all_years_tsne_results, axis=0, ignore_index=True)
master_map_df = pd.read_csv(path_to_data + "master_map_df_with_tsne.csv")
df = pd.read_csv(path_to_data + "density_barplot_data.csv")

# Separate data to ground-truth, whitman and columbia counties, and other counties
master_map_df_whitman_columbia = master_map_df.loc[master_map_df['County'].isin(['Whitman', 'Columbia'])].copy()
master_map_df_other = master_map_df.loc[~master_map_df['County'].isin(['Whitman', 'Columbia'])].copy()
tsne_year = all_years_tsne_results.loc[all_years_tsne_results['year'] == 2012].copy()
tsne_groundt_only = tsne_year.loc[tsne_year['Source'] == 'ground-truth'].copy()

# Filter for split 11 and tsne year 2022
df_split = df.loc[df['split_number'] == 11].copy()
df_split['prediction_set_size'].value_counts()
df_split_2022 = df_split.loc[df_split['tsne_year_space'] == 2022].copy()

# Merge with tsne_groundt_only
gt_split_tsne2022 = pd.merge(tsne_groundt_only, df_split_2022[['pointID', 'prediction_set_size', 'tillage_pred']], on='pointID', how='left')

# Rename prediction set size columns
gt_split_2022 = gt_split_tsne2022.rename(columns={'prediction_set_size': '|ps|'})
master_map_df_whitman_columbia = master_map_df_whitman_columbia.rename(columns={'max_ps_vote': '|ps|'})
master_map_df_other = master_map_df_other.rename(columns={'max_ps_vote': '|ps|'})

# convert |ps| value types to string
gt_split_2022['|ps|'] = gt_split_2022['|ps|'].astype(str)
master_map_df_whitman_columbia['|ps|'] = master_map_df_whitman_columbia['|ps|'].astype(str)
master_map_df_other['|ps|'] = master_map_df_other['|ps|'].astype(str)

# Filter tnse space year
year = 2022
df = df[df["tsne_year_space"] == year]

# Check if for each split the pointIDs are the same
split_till = pd.DataFrame()
for split in df.split_number.unique():
    split_data = df[df.split_number == split]
    split_data = split_data.sort_values(by='pointID')
    split_data = split_data.reset_index()
    split_till[split] = split_data['tillage_pred']

split_till['pointID'] = split_data['pointID']

# Assume last column is 'pointID', so tillage data is all columns except the last one
tillage_cols = split_till.columns[:-1]

# Apply mode across rows
split_till['tillage_pred_vote'] = split_till[tillage_cols].mode(axis=1)[0]

# Merge with ground-truth data
gt_split_2022 = pd.merge(gt_split_2022, split_till[['pointID', 'tillage_pred_vote']], on='pointID', how='left')


# +
# 4. Create reliability groups
def reliability_class(row):
    ps = row["|ps|"]
    till = row["tillage_pred_vote"]
    if ps == '1':
        return "High"
    elif ps in ('2', '3') and till in ("ConventionalTill", "NoTill-DirectSeed"):
        return "Medium"
    elif ps in ('2', '3') and till == "MinimumTill":
        return "Low"
    else:
        return "Unclassified"

gt_split_2022["reliability"] = gt_split_2022.apply(reliability_class, axis=1)
master_map_df_whitman_columbia["reliability"] = master_map_df_whitman_columbia.apply(reliability_class, axis=1)
master_map_df_other["reliability"] = master_map_df_other.apply(reliability_class, axis=1)

# +
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Example data: replace these with your actual DataFrames
datasets = [gt_split_2022,
            master_map_df_whitman_columbia,
            master_map_df_other]

titles = ['Ground-truth',
        'Whitman & Columbia',
        'Other Counties']

# Set all font sizes globally
plt.rcParams.update({'font.size': 16})

# Create a 1x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

unique_reliability = gt_split_2022['reliability'].unique()

for row_idx, relibility in enumerate(unique_reliability):
    for col_idx, (data, title) in enumerate(zip(datasets, titles)):

        relibility_subset = data.loc[data['reliability'] == relibility].copy()
    
        ax = axes[row_idx, col_idx]
        
        # KDE plot
        kde = sns.kdeplot(
            data=relibility_subset,
            x='tsne_1',
            y='tsne_2',
            fill=True,
            cmap='viridis',
            thresh=0.01,
            ax=ax
        )
        
        # Apply logarithmic normalization to contours
        for coll in kde.collections:
            density_array = coll.get_array()
            base_vmin = density_array.min() if density_array.min() > 0 else 1e-3
            vmin = base_vmin * 0.5
            vmax = density_array.max()
            coll.set_norm(mcolors.LogNorm(vmin=vmin, vmax=vmax))
        
        # Axis formatting
        if row_idx == 0:
            ax.set_title(title, fontsize=14)  # Column titles (dataset names)

        if col_idx == 0:
            # Put |relibility| = ... in the standard y-axis label spot
            ax.set_ylabel(f"{relibility}", fontsize=14)

            # Manually place t-SNE 2 slightly below it
            ax.text(-0.15, 0.5, "t-SNE 2", transform=ax.transAxes,
                    fontsize=14, va='center', ha='right', rotation=90)

        if row_idx == 2:
            ax.set_xlabel('t-SNE 1', fontsize=14)  # X-axis label only on bottom row

        # if col_idx == 0:
        #     ax.set_ylabel('t-SNE 2', fontsize=14)  # Y-axis label only on first column

        ax.tick_params(axis='both', labelsize=12)


# plt.suptitle("t-SNE KDE Plots with Log-Scaled Densities", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# your three DataFrames
datasets = [
    gt_split_2022,
    master_map_df_whitman_columbia,
    master_map_df_other
]
titles = ['Ground-truth', 'Whitman & Columbia', 'Other Counties']

# define the reliability groups in the order you like
reliability_groups = ['High', 'Medium', 'Low']

# mapping of crop codes → names
crop_map = {1: 'grain', 2: 'legume', 3: 'canola'}
crop_order = ['grain', 'legume', 'canola']

# global font size
plt.rcParams.update({'font.size': 16})

fig, axes = plt.subplots(
    nrows=3, ncols=3,
    figsize=(12, 12),
    subplot_kw={'aspect': 'equal'}  # make pies circular
)

for row_idx, rel in enumerate(reliability_groups):
    for col_idx, (data, title) in enumerate(zip(datasets, titles)):
        ax = axes[row_idx, col_idx]
        
        # subset to this reliability group
        sub = data.loc[data['reliability'] == rel, 'cdl_cropType']
        # map codes → names, then count
        names = sub.map(crop_map)
        counts = names.value_counts().reindex(crop_order, fill_value=0)
        
        # if there's nothing in this group, skip plotting
        if counts.sum() == 0:
            ax.text(0.5, 0.5, 'no data',
                    ha='center', va='center')
        else:
            # plot pie
            wedges, texts, autotexts = ax.pie(
                counts,
                labels=crop_order,
                autopct=lambda pct: f"{pct:.0f}%",
                startangle=45,
                rotatelabels=True,             # ← rotate each label to match its wedge
                labeldistance=1.1, 
                pctdistance=0.8,             # ← move labels a bit farther out
                textprops={
                    'rotation_mode': 'anchor'  # ← ensure rotation pivots at the label’s anchor
                }
            )

            for txt in autotexts:
                txt.set_fontsize(14)

        # only set the column titles on the top row
        if row_idx == 0:
            ax.set_title(title)
        
        # only set the row labels on the first column
        if col_idx == 0:
            ax.set_ylabel(rel, rotation=0, labelpad=60, va='center')
        
        # remove any default axes
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()


# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# your three DataFrames
datasets = [
    gt_split_2022,
    master_map_df_whitman_columbia,
    master_map_df_other
]
titles = ['GT', 'Wtmn & Col', 'Others']

# define the reliability groups in the order you like
reliability_groups = ['High', 'Medium', 'Low']

# mapping of crop codes → names
crop_map = {1: 'grain', 2: 'legume', 3: 'canola'}
crop_order = ['grain', 'legume', 'canola']

# global font size
plt.rcParams.update({'font.size': 16})

# create figure and axes
fig, axes = plt.subplots(
    nrows=3, ncols=3,
    figsize=(9, 6),
    subplot_kw={'aspect': 'equal'}
)

# store a color map
colors = plt.get_cmap('Set2')(np.arange(len(crop_order)))

for row_idx, rel in enumerate(reliability_groups):
    for col_idx, (data, title) in enumerate(zip(datasets, titles)):
        ax = axes[row_idx, col_idx]
        
        # subset data
        sub = data.loc[data['reliability'] == rel, 'cdl_cropType']
        names = sub.map(crop_map)
        counts = names.value_counts().reindex(crop_order, fill_value=0)

        if counts.sum() == 0:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center')
        else:
            wedges, texts, autotexts = ax.pie(
                counts,
                colors=colors,
                autopct=lambda pct: f"{pct:.0f}%",
                startangle=45,
                pctdistance=0.9
            )

        if row_idx == 0:
            ax.set_title(title)
        if col_idx == 0:
            ax.set_ylabel(rel, rotation=0, labelpad=60, va='center')

        ax.set_xticks([])
        ax.set_yticks([])

# Add legend outside the plot
legend_handles = [Patch(facecolor=colors[i], label=crop_order[i]) for i in range(len(crop_order))]
fig.legend(handles=legend_handles, loc='center right', title='Crop Type')

plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave space on the right for legend
plt.show()


# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

dataset = pd.concat(
    [master_map_df_whitman_columbia,
    master_map_df_other]
    )

crop_map = {1: 'grain', 2: 'legume', 3: 'canola'}
# map crop codes to names
dataset['cdl_cropType'] = dataset['cdl_cropType'].map(crop_map)

tillage_map = {
    'ConventionalTill': 'CT',
    'NoTill-DirectSeed': 'NT',
    'MinimumTill': 'MT'
}
# map tillage codes to names
master_map_df_whitman_columbia['tillage_pred_vote'] = master_map_df_whitman_columbia['tillage_pred_vote'].map(tillage_map)
master_map_df_other['tillage_pred_vote'] = master_map_df_other['tillage_pred_vote'].map(tillage_map)

titles = ['grain', 'legume', 'canola']

# define the reliability groups in the order you like
fr_cats = ['0-15%', '16-30%', '>30%']

# mapping of crop codes → names
rel_order = ['High', 'Medium', 'Low']

# global font size
plt.rcParams.update({'font.size': 16})

# create figure and axes
fig, axes = plt.subplots(
    nrows=3, ncols=3,
    figsize=(9, 6),
    subplot_kw={'aspect': 'equal'}
)

# store a color map
colors = plt.get_cmap('Set2')(np.arange(len(rel_order)))

for row_idx, fr in enumerate(fr_cats):
    for col_idx, title in enumerate(titles):
        ax = axes[row_idx, col_idx]
        
        # subset data
        sub = dataset.loc[(dataset['fr_pred'] == fr) & (dataset['cdl_cropType'] == title), 'reliability']
        counts = sub.value_counts().reindex(rel_order, fill_value=0)
        total = counts.sum()

        if counts.sum() == 0:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center')
        else:
            wedges, texts, autotexts = ax.pie(
                counts,
                colors=colors,
                autopct=lambda pct: f"{pct:.0f}%",
                startangle=45,
                pctdistance=0.9
            )
        # add total below each pie
        ax.text(
            0.5, -0.1, f"n={total}",
            ha='center', va='top',
            transform=ax.transAxes
        )

        if row_idx == 0:
            ax.set_title(title)
        if col_idx == 0:
            ax.set_ylabel(fr, rotation=0, labelpad=60, va='center')

        ax.set_xticks([])
        ax.set_yticks([])

# Add legend outside the plot
legend_handles = [Patch(facecolor=colors[i], label=rel_order[i]) for i in range(len(rel_order))]
fig.legend(handles=legend_handles, loc='center right', title='reliability')

plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave space on the right for legend
plt.show()


# +
dataset = pd.concat([master_map_df_whitman_columbia.copy(),
                     master_map_df_other].copy())

dataset['tillage_pred_vote'].value_counts()

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

dataset = pd.concat([master_map_df_whitman_columbia.copy(),
                     master_map_df_other].copy())

# maps
crop_map = {1: 'grain', 2: 'legume', 3: 'canola'}
dataset['cdl_cropType'] = dataset['cdl_cropType'].replace(crop_map)


tillage_map = {
    'ConventionalTill': 'CT',
    'NoTill-DirectSeed': 'NT',
    'MinimumTill': 'MT'
}
for df in (master_map_df_whitman_columbia, master_map_df_other):
    df['tillage_pred_vote'] = df['tillage_pred_vote'].replace(tillage_map)

# orders
titles     = ['grain', 'legume', 'canola']
fr_cats    = ['0-15%', '16-30%', '>30%']
rel_order  = ['High','Medium','Low']
till_order = ['CT','NT','MT']

# build all 9 labels and colors
combos = [(r,t) for r in rel_order for t in till_order]
labels = [f"{r} {t}" for r,t in combos]
# you can pick any colormap with ≥9 distinct colors
colors = plt.get_cmap('tab20')(np.arange(len(labels)))

# start plotting
plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(3,3, figsize=(9,6), subplot_kw={'aspect':'equal'})

for i, fr in enumerate(fr_cats):
    for j, crop in enumerate(titles):
        ax = axes[i,j]
        sub = dataset.loc[
            (dataset['fr_pred']==fr)&
            (dataset['cdl_cropType']==crop),
            ['reliability','tillage_pred_vote']
        ]
        # count every combo, even if zero
        idx = pd.MultiIndex.from_product([rel_order,till_order],
                                         names=['reliability','tillage_pred_vote'])
        counts = sub.groupby(['reliability','tillage_pred_vote']).size().reindex(idx, fill_value=0)
        vals = counts.values
        total = vals.sum()

        if total==0:
            ax.text(0.5,0.5,'no data', ha='center', va='center')
        else:
            ax.pie(vals,
                   colors=colors,
                   startangle=45,
                   labels=None,           # hide slice labels
                   wedgeprops=dict(width=0.7))

        ax.text(.5, -0.1, f"n={total}",
                transform=ax.transAxes, ha='center', va='top')
        if i==0: ax.set_title(crop)
        if j==0: ax.set_ylabel(fr, rotation=0, labelpad=60, va='center')
        ax.set_xticks([]); ax.set_yticks([])

# 5) Reserve room on the right for the legend
fig.subplots_adjust(left=0.05, right=0.75, bottom=0.05, top=0.95,
                    wspace=0.3, hspace=0.3)

# 6) Add legend
handles = [Patch(facecolor=colors[k], label=labels[k]) for k in range(len(labels))]
fig.legend(handles=handles, loc='center right', title='Reliability + Tillage')

plt.show()


# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ─── 1) Prepare your dataset ───────────────────────────────────────────────────
# (apply the same mapping you had)
crop_map = {1: 'grain', 2: 'legume', 3: 'canola'}
tillage_map = {
    'ConventionalTill': 'CT',
    'NoTill-DirectSeed': 'NT',
    'MinimumTill': 'MT'
}

# make copies so we don’t mutate the originals
df1 = master_map_df_whitman_columbia.copy()
df2 = master_map_df_other.copy()

# map codes → names
df1['cdl_cropType']      = df1['cdl_cropType'].replace(crop_map)
df2['cdl_cropType']      = df2['cdl_cropType'].replace(crop_map)
df1['tillage_pred_vote'] = df1['tillage_pred_vote'].replace(tillage_map)
df2['tillage_pred_vote'] = df2['tillage_pred_vote'].replace(tillage_map)

# concat
dataset = pd.concat([df1, df2], ignore_index=True)

# ─── 2) Define categories & colors ─────────────────────────────────────────────
rel_order  = ['High', 'Medium', 'Low']       # the 3 columns
fr_cats    = ['0-15%', '16-30%', '>30%']
crop_types = ['grain', 'legume', 'canola']

# legend labels are all combinations of fr_pred × cdl_cropType
labels = [f"{fr} {crop}" for fr in fr_cats for crop in crop_types]
colors = plt.get_cmap('tab20')(np.arange(len(labels)))

# ─── 3) Plot 1×3 row of pies ───────────────────────────────────────────────────
plt.rcParams.update({'font.size': 14})
fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'aspect': 'equal'})

for j, rel in enumerate(rel_order):
    ax = axes[j]
    # filter to this reliability
    sub = dataset.loc[
        dataset['reliability'] == rel,
        ['fr_pred', 'cdl_cropType']
    ]
    # ensure all 9 combos appear
    idx    = pd.MultiIndex.from_product([fr_cats, crop_types],
                                        names=['fr_pred', 'cdl_cropType'])
    counts = sub.groupby(['fr_pred', 'cdl_cropType']).size() \
                .reindex(idx, fill_value=0)
    vals   = counts.values
    total  = vals.sum()

    if total == 0:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center')
    else:
        ax.pie(vals,
               colors=colors,
               startangle=45,
               labels=None,
               wedgeprops=dict(width=0.7))

    ax.set_title(f"{rel} Reliability")
    ax.text(0.5, -0.1, f"n={total}",
            transform=ax.transAxes, ha='center', va='top')
    ax.set_xticks([]); ax.set_yticks([])

# ─── 4) Adjust layout & add legend on the right ────────────────────────────────
fig.subplots_adjust(left=0.05, right=0.75, wspace=0.4)

handles = [Patch(facecolor=colors[i], label=labels[i]) for i in range(len(labels))]
fig.legend(handles=handles,
           loc='center right',
           title='fr % + Crop type',
           bbox_to_anchor=(0.95, 0.5))

plt.show()


# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ─── 1) Prepare your dataset ───────────────────────────────────────────────────
# (apply the same mapping you had)
crop_map = {1: 'grain', 2: 'legume', 3: 'canola'}
tillage_map = {
    'ConventionalTill': 'CT',
    'NoTill-DirectSeed': 'NT',
    'MinimumTill': 'MT'
}

# make copies so we don’t mutate the originals
df1 = master_map_df_whitman_columbia.copy()
df2 = master_map_df_other.copy()

# map codes → names
df1['cdl_cropType']      = df1['cdl_cropType'].replace(crop_map)
df2['cdl_cropType']      = df2['cdl_cropType'].replace(crop_map)
df1['tillage_pred_vote'] = df1['tillage_pred_vote'].replace(tillage_map)
df2['tillage_pred_vote'] = df2['tillage_pred_vote'].replace(tillage_map)

# concat
dataset = pd.concat([df1, df2], ignore_index=True)

# ─── 2) Define categories & colors ─────────────────────────────────────────────
rel_order  = ['High', 'Medium', 'Low']       # the 3 columns
fr_cats    = ['0-15%', '16-30%', '>30%']
crop_types = ['grain', 'legume', 'canola']
till_order = ['CT', 'MT', 'NT']

# legend labels are all combinations of fr_pred × cdl_cropType
labels = [f"{fr} {crop}" for fr in fr_cats for crop in crop_types]
colors = plt.get_cmap('tab20')(np.arange(len(labels)))

# ─── 3) Plot 1×3 row of pies ───────────────────────────────────────────────────
plt.rcParams.update({'font.size': 14})
fig, axes = plt.subplots(3, 3, figsize=(12, 8), subplot_kw={'aspect': 'equal'})

for i, rel in enumerate(rel_order):
    for j, till in enumerate(till_order):
        ax = axes[j, i]
        # filter to this reliability
        sub = dataset.loc[
            (dataset['reliability'] == rel) &
            (dataset['tillage_pred_vote'] == till),
            ['fr_pred', 'cdl_cropType']
        ]
        # ensure all 9 combos appear
        idx    = pd.MultiIndex.from_product([fr_cats, crop_types],
                                            names=['fr_pred', 'cdl_cropType'])
        counts = sub.groupby(['fr_pred', 'cdl_cropType']).size() \
                    .reindex(idx, fill_value=0)
        vals   = counts.values
        total  = vals.sum()

        if total == 0:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center')
        else:
            ax.pie(vals,
                colors=colors,
                startangle=45,
                labels=None,
                wedgeprops=dict(width=0.7))

        # — now rel are column titles —
        if j == 0:
            ax.set_title(rel)
        # — and till are row labels on the leftmost col —
        if i == 0:
            ax.set_ylabel(f"{till}", rotation=0,
                          labelpad=50, va='center')

        ax.text(0.5, -0.1, f"n={total}",
                transform=ax.transAxes,
                ha='center', va='top')
        ax.set_xticks([]); ax.set_yticks([])

fig.subplots_adjust(left=0.15, right=0.75, wspace=0.4)
handles = [Patch(facecolor=colors[k], label=labels[k])
           for k in range(len(labels))]
fig.legend(handles=handles,
           loc='center right',
           title='fr % + crop type',
           bbox_to_anchor=(0.95, 0.5))
plt.show()


# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# your three DataFrames
datasets = [
    gt_split_2022,
    master_map_df_whitman_columbia,
    master_map_df_other
]
titles = ['Ground-truth', 'Whitman & Columbia', 'Other Counties']

# 1) global t-SNE bounds
all_data = pd.concat(datasets, ignore_index=True)
xmin, xmax = all_data['tsne_1'].min(), all_data['tsne_1'].max()
ymin, ymax = all_data['tsne_2'].min(), all_data['tsne_2'].max()

# 2) bin edges for a 10×10 grid
x_edges = np.linspace(xmin, xmax, 11)
y_edges = np.linspace(ymin, ymax, 11)

# 3) define your reliability groups and colormap
#    (make sure this matches exactly your actual group names/order)
reliability_groups = ['High', 'Medium', 'Low']
group_to_int = {g: i for i, g in enumerate(reliability_groups)}
cmap = mcolors.ListedColormap(['#d73027','#fee08b','#1a9850'])  # e.g. red, yellow, green

# Set all font sizes globally
plt.rcParams.update({'font.size': 16})

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

for ax, data, title in zip(axes, datasets, titles):
    # initialize an array of “empty” for each of the 10×10 cells
    grid_codes = np.full((10, 10), np.nan)
    x = data['tsne_1'].values
    y = data['tsne_2'].values
    labels = data['reliability']

    # 4) loop over all cells, pick majority label
    for i in range(10):
        for j in range(10):
            in_cell = (
                (x >= x_edges[i]) & (x < x_edges[i+1]) &
                (y >= y_edges[j]) & (y < y_edges[j+1])
            )
            if in_cell.any():
                maj = labels[in_cell].value_counts().idxmax()
                grid_codes[j, i] = group_to_int[maj]

    # 5) plot it
    X, Y = np.meshgrid(x_edges, y_edges)
    ax.pcolormesh(X, Y, grid_codes, cmap=cmap, shading='auto')
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.tick_params()

# add a single legend off to the side
legend_patches = [
    Patch(facecolor=cmap(i), label=grp)
    for i, grp in enumerate(reliability_groups)
]
fig.legend(
    handles=legend_patches,
    title="Reliability Group",
    loc='center right',
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.1
)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()


# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
rename_dict = {
    1:"grain",
    2:"legume",
    3:"canola"
}

gt_split_2022['cdl_cropType'] = gt_split_2022['cdl_cropType'].replace(rename_dict)
master_map_df_whitman_columbia['cdl_cropType'] = master_map_df_whitman_columbia['cdl_cropType'].replace(rename_dict)
master_map_df_other['cdl_cropType'] = master_map_df_other['cdl_cropType'].replace(rename_dict)


# your three DataFrames
datasets = [
    gt_split_2022,
    master_map_df_whitman_columbia,
    master_map_df_other
]
titles = ['Ground-truth', 'Whitman & Columbia', 'Other Counties']

# 1) global t-SNE bounds
all_data = pd.concat(datasets, ignore_index=True)


xmin, xmax = all_data['tsne_1'].min(), all_data['tsne_1'].max()
ymin, ymax = all_data['tsne_2'].min(), all_data['tsne_2'].max()

# 2) bin edges for a 10×10 grid
x_edges = np.linspace(xmin, xmax, 11)
y_edges = np.linspace(ymin, ymax, 11)

# 3) define your reliability groups and colormap
#    (make sure this matches exactly your actual group names/order)
fr_cls = ['grain', 'legume', 'canola']
group_to_int = {g: i+1 for i, g in enumerate(fr_cls)}
cmap = mcolors.ListedColormap(['#d73027','#fee08b','#1a9850'])  # e.g. red, yellow, green

# Set all font sizes globally
plt.rcParams.update({'font.size': 16})

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

for ax, data, title in zip(axes, datasets, titles):
    # initialize an array of “empty” for each of the 10×10 cells
    grid_codes = np.full((10, 10), np.nan)
    x = data['tsne_1'].values
    y = data['tsne_2'].values
    labels = data['cdl_cropType']

    # 4) loop over all cells, pick majority label
    for i in range(10):
        for j in range(10):
            in_cell = (
                (x >= x_edges[i]) & (x < x_edges[i+1]) &
                (y >= y_edges[j]) & (y < y_edges[j+1])
            )
            if in_cell.any():
                maj = labels[in_cell].value_counts().idxmax()
                grid_codes[j, i] = group_to_int[maj]

    # 5) plot it
    X, Y = np.meshgrid(x_edges, y_edges)
    ax.pcolormesh(X, Y, grid_codes, cmap=cmap, shading='auto')
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.tick_params()

# add a single legend off to the side
legend_patches = [
    Patch(facecolor=cmap(i), label=grp)
    for i, grp in enumerate(fr_cls)
]
fig.legend(
    handles=legend_patches,
    title="crop",
    loc='center right',
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.1
)

plt.tight_layout(rect=[0,0,0.9,1])
plt.show()


# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
rename_dict = {
    1:"0-15%",
    2:"16-30%",
    3:">30%"
}

gt_split_2022['fr_pred'] = gt_split_2022['fr_pred'].replace(rename_dict)
master_map_df_whitman_columbia['fr_pred'] = master_map_df_whitman_columbia['fr_pred'].replace(rename_dict)
master_map_df_other['fr_pred'] = master_map_df_other['fr_pred'].replace(rename_dict)


# your three DataFrames
datasets = [
    gt_split_2022,
    master_map_df_whitman_columbia,
    master_map_df_other
]
titles = ['Ground-truth', 'Whitman & Columbia', 'Other Counties']

# 1) global t-SNE bounds
all_data = pd.concat(datasets, ignore_index=True)


xmin, xmax = all_data['tsne_1'].min(), all_data['tsne_1'].max()
ymin, ymax = all_data['tsne_2'].min(), all_data['tsne_2'].max()

# 2) bin edges for a 10×10 grid
x_edges = np.linspace(xmin, xmax, 11)
y_edges = np.linspace(ymin, ymax, 11)

# 3) define your reliability groups and colormap
#    (make sure this matches exactly your actual group names/order)
fr_cls = ['0-15%', '16-30%', '>30%']
group_to_int = {g: i+1 for i, g in enumerate(fr_cls)}
cmap = mcolors.ListedColormap(['#d73027','#fee08b','#1a9850'])  # e.g. red, yellow, green

# Set all font sizes globally
plt.rcParams.update({'font.size': 16})

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

for ax, data, title in zip(axes, datasets, titles):
    # initialize an array of “empty” for each of the 10×10 cells
    grid_codes = np.full((10, 10), np.nan)
    x = data['tsne_1'].values
    y = data['tsne_2'].values
    labels = data['fr_pred']

    # 4) loop over all cells, pick majority label
    for i in range(10):
        for j in range(10):
            in_cell = (
                (x >= x_edges[i]) & (x < x_edges[i+1]) &
                (y >= y_edges[j]) & (y < y_edges[j+1])
            )
            if in_cell.any():
                maj = labels[in_cell].value_counts().idxmax()
                grid_codes[j, i] = group_to_int[maj]

    # 5) plot it
    X, Y = np.meshgrid(x_edges, y_edges)
    ax.pcolormesh(X, Y, grid_codes, cmap=cmap, shading='auto')
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.tick_params()

# add a single legend off to the side
legend_patches = [
    Patch(facecolor=cmap(i), label=grp)
    for i, grp in enumerate(fr_cls)
]
fig.legend(
    handles=legend_patches,
    title="fr class",
    loc='center right',
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.1
)

plt.tight_layout(rect=[0,0,0.9,1])
plt.show()


# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
rename_dict = {
    1:"0-15%",
    2:"16-30%",
    3:">30%"
}

gt_split_2022['fr_pred'] = gt_split_2022['fr_pred'].replace(rename_dict)
master_map_df_whitman_columbia['fr_pred'] = master_map_df_whitman_columbia['fr_pred'].replace(rename_dict)
master_map_df_other['fr_pred'] = master_map_df_other['fr_pred'].replace(rename_dict)


# your three DataFrames
datasets = [
    gt_split_2022,
    master_map_df_whitman_columbia,
    master_map_df_other
]
titles = ['Ground-truth', 'Whitman & Columbia', 'Other Counties']

# 1) global t-SNE bounds
all_data = pd.concat(datasets, ignore_index=True)


xmin, xmax = all_data['tsne_1'].min(), all_data['tsne_1'].max()
ymin, ymax = all_data['tsne_2'].min(), all_data['tsne_2'].max()

# 2) bin edges for a 10×10 grid
x_edges = np.linspace(xmin, xmax, 11)
y_edges = np.linspace(ymin, ymax, 11)

# 3) define your reliability groups and colormap
#    (make sure this matches exactly your actual group names/order)
fr_cls = ['0-15%', '16-30%', '>30%']
group_to_int = {g: i+1 for i, g in enumerate(fr_cls)}
cmap = mcolors.ListedColormap(['#d73027','#fee08b','#1a9850'])  # e.g. red, yellow, green

# Set all font sizes globally
plt.rcParams.update({'font.size': 16})

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

for ax, data, title in zip(axes, datasets, titles):
    # initialize an array of “empty” for each of the 10×10 cells
    grid_codes = np.full((10, 10), np.nan)
    x = data['tsne_1'].values
    y = data['tsne_2'].values
    labels = data['fr_pred']

    # 4) loop over all cells, pick majority label
    for i in range(10):
        for j in range(10):
            in_cell = (
                (x >= x_edges[i]) & (x < x_edges[i+1]) &
                (y >= y_edges[j]) & (y < y_edges[j+1])
            )
            if in_cell.any():
                maj = labels[in_cell].value_counts().idxmax()
                grid_codes[j, i] = group_to_int[maj]

    # 5) plot it
    X, Y = np.meshgrid(x_edges, y_edges)
    ax.pcolormesh(X, Y, grid_codes, cmap=cmap, shading='auto')
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.tick_params()

# add a single legend off to the side
legend_patches = [
    Patch(facecolor=cmap(i), label=grp)
    for i, grp in enumerate(fr_cls)
]
fig.legend(
    handles=legend_patches,
    title="fr class",
    loc='center right',
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.1
)

plt.tight_layout(rect=[0,0,0.9,1])
plt.show()


# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# 0) your raw dfs
datasets = [
    gt_split_2022,
    master_map_df_whitman_columbia,
    master_map_df_other
]
titles = ['Ground-truth','Whitman & Columbia','Other Counties']

# 1) define the int→string maps
fr_map   = {1: "0-15%", 2: "16-30%", 3: ">30%"}
crop_map = {1: 'grain',   2: 'legume',  3: 'canola'}

# 2) build safe string columns
for df in datasets:
    df['fr_str']   = df['fr_pred'].apply(lambda x: fr_map.get(x, x))
    df['crop_str'] = df['cdl_cropType'].apply(lambda x: crop_map.get(x, x))

# 3) all 9 combos & an int code for each one
crop_types  = ['grain','legume','canola']
fr_classes  = ['0-15%','16-30%','>30%']
comb_labels = [f"{c}-{f}" for c in crop_types for f in fr_classes]
combo_to_int = {lab: i+1 for i, lab in enumerate(comb_labels)}

# 4) a 9-color palette
base   = plt.get_cmap('tab20')
colors = [base(i*2) for i in range(9)]
cmap9  = mcolors.ListedColormap(colors)

# 5) global t-SNE bounds & grid edges
all_data = pd.concat(datasets, ignore_index=True)
xmin, xmax = all_data['tsne_1'].min(), all_data['tsne_1'].max()
ymin, ymax = all_data['tsne_2'].min(), all_data['tsne_2'].max()
x_edges = np.linspace(xmin, xmax, 11)
y_edges = np.linspace(ymin, ymax, 11)

# 6) plot
plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(1, 3, figsize=(15,5), sharex=True, sharey=True)

for ax, df, title in zip(axes, datasets, titles):
    x      = df['tsne_1'].values
    y      = df['tsne_2'].values
    labels = df['crop_str'] + '-' + df['fr_str']

    grid = np.full((10,10), np.nan)
    for i in range(10):
        for j in range(10):
            m = (
                (x >= x_edges[i]) & (x < x_edges[i+1]) &
                (y >= y_edges[j]) & (y < y_edges[j+1])
            )
            if not m.any(): continue
            sub = labels[m].dropna()
            if sub.empty:   continue
            maj = sub.value_counts().idxmax()
            grid[j,i] = combo_to_int[maj]

    X, Y = np.meshgrid(x_edges, y_edges)
    ax.pcolormesh(X, Y, grid, cmap=cmap9, shading='auto')
    ax.set(title=title, xlabel='t-SNE 1', ylabel='t-SNE 2')

# 7) nine-entry legend
patches = [Patch(facecolor=cmap9(i), label=lab)
           for i, lab in enumerate(comb_labels)]
fig.legend(
    handles=patches,
    title="Crop – fr class",
    loc='center right',
    bbox_to_anchor=(1.02,0.5),
    borderaxespad=0.1
)

plt.tight_layout(rect=[0,0,0.85,1])
plt.show()


# +
all_data = pd.concat(datasets, ignore_index=True)
print(all_data['fr_pred'].value_counts())

rename_dict = {
    1:"0-15%",
    2:"16-30%",
    3:">30%"
}

all_data['fr_pred'] = all_data['fr_pred'].replace(rename_dict)

all_data['fr_pred'].value_counts()

print(all_data['fr_pred'].value_counts())

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# your three DataFrames
datasets = [
    gt_split_2022,
    master_map_df_whitman_columbia,
    master_map_df_other
]
titles = ['Ground-truth', 'Whitman & Columbia', 'Other Counties']

# 1) global t-SNE bounds
all_data = pd.concat(datasets, ignore_index=True)
xmin, xmax = all_data['tsne_1'].min(), all_data['tsne_1'].max()
ymin, ymax = all_data['tsne_2'].min(), all_data['tsne_2'].max()

# 2) bin edges for a 10×10 grid
x_edges = np.linspace(xmin, xmax, 11)
y_edges = np.linspace(ymin, ymax, 11)

# global font size
plt.rcParams.update({'font.size': 16})

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

# choose a continuous colormap
cmap = cm.get_cmap('viridis')
norm = Normalize(vmin=0.5, vmax=1.0)  
# (you could use vmin=0, vmax=1 if you want the full 0–100% scale)

for ax, data, title in zip(axes, datasets, titles):
    # 3) compute dominance fraction matrix
    grid_frac = np.full((10, 10), np.nan)
    x = data['tsne_1'].values
    y = data['tsne_2'].values
    labels = data['reliability']

    for i in range(10):
        for j in range(10):
            mask = (
                (x >= x_edges[i]) & (x < x_edges[i+1]) &
                (y >= y_edges[j]) & (y < y_edges[j+1])
            )
            if mask.any():
                counts = labels[mask].value_counts()
                grid_frac[j, i] = counts.max() / counts.sum()

    # 4) plot with continuous shading
    X, Y = np.meshgrid(x_edges, y_edges)
    pcm = ax.pcolormesh(
        X, Y, grid_frac,
        cmap=cmap,
        norm=norm,
        shading='auto'
    )
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.tick_params()

# 5) add a colorbar to the right of all subplots
# create a new axis on the right for the colorbar
cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(
    pcm,
    cax=cbar_ax,
    label='% of the dominant reliability group'
)
cbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
cbar.set_ticklabels(['50%', '60%', '70%', '80%', '90%', '100%'])


plt.tight_layout()
plt.show()

