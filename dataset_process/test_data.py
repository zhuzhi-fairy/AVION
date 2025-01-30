# %%
import os

import decord

print(decord.__path__)
root = "datasets/clip_dummy/video_320p_15sec/"
folders = os.listdir(root)
# folder = folders[0]
for folder in folders:
    files = os.listdir(os.path.join(root, folder))
    # file = files[0]
    for file in files:
        video_path = os.path.join(root, folder, file)
        try:
            vr = decord.VideoReader(video_path)
        except:
            print(video_path)
# vr = decord.VideoReader(
#     "datasets/clip_dummy/video_320p_15sec/archive-2EKXFEXK495R1DDSEQJT5H8PN0.mp4/3045.mp4"
# )
# %%
