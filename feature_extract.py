# %%
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
import torchvision
from einops import rearrange
from pytorchvideo.transforms import Normalize

import avion.models.model_videomae as model_videomae
from avion.data.classification_dataset import VideoClsDataset
from avion.data.transforms import AdaptiveTemporalCrop, Permute, SpatialCrop


# %%
class VideoClsDataset_feature_extraction(VideoClsDataset):
    def __init__(
        self,
        file_idx,
        root="datasets/hktk/sss/annofab-data",
        metadata="data/features/hktk247_meta.csv",
        clip_length=16,
        clip_stride=4,
        threads=1,
        crop_size=224,
        shorter_side_size=224,
        new_height=256,
        new_width=340,
        keep_aspect_ratio=True,
        fast_rrc=False,
        rrc_params=(224, (0.5, 1.0)),
        fast_cc=False,
        cc_params=(224,),
        hflip_prob=0.5,
        vflip_prob=0.0,
        num_segment=1,
        num_crop=1,
        test_num_crop=3,
        args=None,
    ):
        # meta data of video data
        dfv = pd.read_csv(metadata, index_col=[0])
        self.vid = dfv.video.iloc[file_idx][:-4]
        self.samples = [[self.vid, -1, -1]]
        self.test_num_segment = dfv.num_segments.iloc[file_idx]
        # other parameters
        self.root = root
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.crop_size = crop_size
        self.shorter_side_size = shorter_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_cc = fast_cc
        self.cc_params = cc_params
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.num_segment = num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        self.mode = "test"
        # openai normalization
        mean = [108.3272985, 116.7460125, 104.09373615000001]
        std = [68.5005327, 66.6321579, 70.32316305]
        self.data_transform = torchvision.transforms.Compose(
            [
                Permute([3, 0, 1, 2]),
                torchvision.transforms.Resize(self.shorter_side_size),
                Normalize(mean=mean, std=std),
                AdaptiveTemporalCrop(
                    self.clip_length,
                    self.test_num_segment,
                    self.clip_stride,
                ),
                SpatialCrop(
                    crop_size=self.shorter_side_size,
                    num_crops=self.test_num_crop,
                ),
            ]
        )


def load_model(ckpt_file, load_weights=True):
    ckpt = torch.load(ckpt_file, map_location="cpu")
    args = ckpt["args"]
    model = getattr(model_videomae, "VIDEOMAE_CLIP_VITL14_FEATURE")(
        pretrained=False,
        drop_path_rate=args.drop_path_rate,
        decoder_depth=args.decoder_depth,
        use_flash_attn_at_encoder=args.use_flash_attn_at_encoder,
        use_flash_attn_at_decoder=args.use_flash_attn_at_decoder,
        use_checkpoint=args.use_grad_checkpointing,
        channel_last=args.channel_last,
        clip_ckpt_path=args.clip_ckpt,
    )
    if load_weights:
        print("load pretained weights from {}".format(ckpt_file))
        state_dict = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            if "module." in k:
                state_dict[k.replace("module.", "")] = v
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    device = f"cuda:{sys.argv[1]}"
    file_idx_start = int(sys.argv[2])
    file_idx_end = int(sys.argv[3])
    batch_size = 128
    num_workers = 10
    # ckpt_file = "experiments/trial1/checkpoint_00200.pt"
    # target_folder = "data/features/avion_LaViLa_vitl_14/features"
    # ckpt_file = "experiments/videomae_pretrain_clip2mae_vitl_hktk555_1200e/checkpoint_00800.pt"
    ckpt_file = "experiments/videomae_pretrain_clip2mae_vitl_hktk555_1200e_2/checkpoint_00670.pt"
    # target_folder = (
    #     "data/features/videomae_pretrain_clip2mae_vitl_hktk555_1200e/features"
    # )
    target_folder = "data/features/videomae_pretrain_clip2mae_vitl_hktk555_1200e_2-670e/features"
    os.makedirs(target_folder, exist_ok=True)
    for n in range(file_idx_start, file_idx_end):
        ds = VideoClsDataset_feature_extraction(n)
        data = next(iter(ds))
        target_file = os.path.join(
            target_folder, f"{ds.vid.split('/')[-1]}.npy"
        )
        model = load_model(ckpt_file, load_weights=True)
        t_patches = model.encoder.num_frames
        s_patches = model.encoder.grid_size
        vit_dim = model.encoder.width
        batch_start_ls = range(0, data[0].shape[0], batch_size)
        features = []
        for batch_start in batch_start_ls:
            x = data[0][batch_start : batch_start + batch_size]
            batch_size_r = x.shape[0]
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                with amp.autocast():
                    last_hidden_state = model(x)
            last_hidden_state = last_hidden_state.cpu().detach()
            last_hidden_state = last_hidden_state.view(
                [
                    batch_size_r,
                    t_patches // 2,  # number of time patches
                    # the real time patch of ViT in CLIP is 1
                    # however to compare with VideoMAE like models fairly
                    # the embedding of every 2 frames are averaged here
                    2,  # time patch size
                    s_patches[0],  # spatial patch size
                    s_patches[1],  # spatial patch size
                    vit_dim,  # ViT embedding dimension
                ]
            )
            last_hidden_state = torch.mean(last_hidden_state, (2, 3, 4))
            features.append(last_hidden_state)
        features = torch.concat(features)
        features = features.view(
            int(ds.test_num_segment),
            ds.test_num_crop,
            *features.shape[1:],
        )
        np.save(target_file, features)
# %%
