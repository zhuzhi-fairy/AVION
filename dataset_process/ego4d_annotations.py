# %%
import os
import pickle

import numpy as np

# %%
files = [
    "Ego4D/ego4d_train.narrator_63690737.return_10.pkl",
    "Ego4D/ego4d_train.pkl",
    "Ego4D/ego4d_train.rephraser.no_punkt_top3.pkl",
    "Ego4D/ego4d_val.pkl",
]
n = 1
file = files[n]
with open(file, "rb") as f:
    data = pickle.load(f)
# %%
duration = np.array([d[2] - d[1] for d in data])
# %%
