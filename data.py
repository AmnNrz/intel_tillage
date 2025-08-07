import pandas as pd
import numpy as np
import debugpy


# listen on all interfaces, port 5678
debugpy.listen(("0.0.0.0", 5678))
print("⏳ waiting for VS Code to attach on port 5678…")
debugpy.wait_for_client()

# …the rest of your code…

path_to_data = ("/home/a.norouzikandelati/Projects/data/data_tillage_mapping/")

gt = pd.read_csv(path_to_data + "gt_n_data.csv")
gt # type: ignore
# ps = pd.read_csv(path_to_data + "to_share/all_map_data_with_vote_and_ps.csv")
print(gt.shape)