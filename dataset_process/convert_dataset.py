# %%
import os

import decord
from tqdm import tqdm

source_file = "/mnt/data/zhu/hktk-maest/data/processed/hktk555_pt/train.csv"
with open(source_file) as f:
    lines = f.readlines()
target_file = "hktk555_pt/train.txt"
# %%
nfls = []
avion_hktk555_data = []
line = lines[0]
for line in tqdm(lines):
    file_path = line.split()[0][33:]
    vr = decord.VideoReader(f"hktk555_pt/sss/annofab-data/{file_path}")
    num_frames = len(vr) - 64
    avion_hktk555_data.append(
        " ".join([file_path, str(num_frames), "-1"]) + "\n"
    )
    nfls.append(num_frames)
# %%
with open(target_file, "w") as f:
    f.writelines(avion_hktk555_data)
# %%
