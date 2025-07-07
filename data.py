import pandas as pd
import numpy as np



path_to_data = ("/home/a.norouzikandelati/Projects/data/data_tillage_mapping/")

gt = pd.read_csv(path_to_data + "gt_n_data.csv")
gt # type: ignore
# ps = pd.read_csv(path_to_data + "to_share/all_map_data_with_vote_and_ps.csv")
print(gt.shape)