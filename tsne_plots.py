import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path_to_data = "/home/a.norouzikandelati/Projects/data/data_tillage_mapping/"

# Read master file
master_file = pd.read_csv(path_to_data + "to_share/master_map_df_with_tsne.csv")