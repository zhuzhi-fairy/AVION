# %%
import argparse
import datetime
import json
import math
import os
import sys
import time
from collections import OrderedDict

import kornia as K
import torch
import torch.cuda.amp as amp
import torchvision
from einops import rearrange
from timm.data.loader import MultiEpochsDataLoader
from torch.distributed.optim import ZeroRedundancyOptimizer

import avion.models.model_videomae as model_videomae
import avion.utils.distributed as dist_utils
from avion.data.kinetics_dataset import KineticsDataset
from avion.data.transforms import (
    GroupMultiScaleCrop,
    Permute,
    TubeMaskingGeneratorGPU,
)
from avion.optim.lion import Lion
from avion.optim.schedulers import cosine_scheduler
from avion.utils.meters import AverageMeter, ProgressMeter
from avion.utils.misc import check_loss_nan

torch.backends.cudnn.benchmark = True
# %%
output_dir = "experiments/videomae_pretrain_clip2mae_vitl_hktk555_1200e"
epoch = 800
ckpt_file = os.path.join(output_dir, f"checkpoint_{epoch:05d}.pt")
ckpt = torch.load(ckpt_file, map_location="cpu")
args = ckpt["args"]
# %% model
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
model.cuda(args.gpu)
patch_size = model.encoder.patch_size
args.window_size = (
    args.clip_length,
    args.input_size // patch_size[0],
    args.input_size // patch_size[1],
)
args.patch_size = patch_size
state_dict = OrderedDict()
for k, v in ckpt["state_dict"].items():
    if "module." in k:
        state_dict[k.replace("module.", "")] = v
model.load_state_dict(state_dict, strict=False)
model.eval()
# %%
if args.norm_style == "openai":
    mean, std = [108.3272985, 116.7460125, 104.09373615000001], [
        68.5005327,
        66.6321579,
        70.32316305,
    ]
elif args.norm_style == "timm":
    mean, std = [0.485 * 255, 0.456 * 255, 0.406 * 255], [
        0.229 * 255,
        0.224 * 255,
        0.225 * 255,
    ]
else:
    raise ValueError('--norm-style should be in ["openai", "timm"]!')
normalize = K.enhance.Normalize(mean=mean, std=std)
