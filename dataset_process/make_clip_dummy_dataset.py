# %%
import os
import pickle

import pandas as pd

# %%
root = "clip_dummy"
videos = os.listdir(os.path.join(root, "video_320p_15sec"))
metadata_ls = []
vidx = 0
for vidx in range(len(videos)):
    video = videos[vidx]
    video_path = os.path.join(root, "video_320p_15sec", video)
    clips = os.listdir(video_path)
    clips = [int(clip[:-4]) for clip in clips]
    clips.sort()
    for cidx in range(clips[-2]):
        metadata = (
            video[:-4],
            cidx,
            cidx + 1,
            "#C C attention is all you need",
        )
        metadata_ls.append(metadata)
thr = int(len(metadata_ls) * 0.8)
train_metadata_ls = metadata_ls[:thr]
val_metadata_ls = metadata_ls[thr:]
with open(os.path.join(root, "train.pkl"), "wb") as f:
    pickle.dump(train_metadata_ls, f)
with open(os.path.join(root, "val.pkl"), "wb") as f:
    pickle.dump(val_metadata_ls, f)
# %%
dfv0 = pd.read_csv(
    "EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv"
)
# %%
