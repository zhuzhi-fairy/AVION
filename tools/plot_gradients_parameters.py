# %%
import os
from collections import OrderedDict

import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.cuda.amp as amp
import torchvision
from einops import rearrange
from timm.data.loader import MultiEpochsDataLoader

import avion.models.model_videomae as model_videomae
from avion.data.kinetics_dataset import KineticsDataset
from avion.data.transforms import (
    GroupMultiScaleCrop,
    Permute,
    TubeMaskingGeneratorGPU,
)
from avion.optim.lion import Lion

torch.backends.cudnn.benchmark = True


def load_ckpt(output_dir, epoch, gpu=None):
    if epoch == "err":
        ckpt_file = output_dir + "/checkpoint_err.pt"
    elif epoch == "last":
        ckpt_file = output_dir + "/checkpoint.pt"
    else:
        ckpt_file = output_dir + f"/checkpoint_{epoch:05d}.pt"
    ckpt = torch.load(ckpt_file)
    args = ckpt["args"]
    if gpu is not None:
        args.gpu = gpu
    # make model
    model = getattr(model_videomae, args.model)(
        pretrained=False,
        drop_path_rate=args.drop_path_rate,
        decoder_depth=args.decoder_depth,
        use_flash_attn_at_encoder=args.use_flash_attn_at_encoder,
        use_flash_attn_at_decoder=args.use_flash_attn_at_decoder,
        use_checkpoint=args.use_grad_checkpointing,
        channel_last=args.channel_last,
        clip_ckpt_path=args.clip_ckpt,
    )
    # load model state_dict
    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        if "module." in k:
            state_dict[k.replace("module.", "")] = v
    model.load_state_dict(state_dict)
    # load optimizer
    n_wd, n_non_wd = [], []
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if (
            p.ndim < 2
            or "bias" in n
            or "ln" in n
            or "bn" in n
            or "pos_embed" in n
            or "positional_embedding" in n
        ):
            n_non_wd.append(n)
            p_non_wd.append(p)
        else:
            n_wd.append(n)
            p_wd.append(p)

    print("parameters without wd:", n_non_wd)
    print("parameters with wd:", n_wd)
    optim_params = [
        {"params": p_wd, "weight_decay": args.wd},
        {"params": p_non_wd, "weight_decay": 0},
    ]
    if args.optimizer == "adamw":
        opt_fn = torch.optim.AdamW
    elif args.optimizer == "lion":
        opt_fn = Lion
    else:
        raise ValueError
    optimizer = opt_fn(
        optim_params,
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.wd,
    )
    optimizer.load_state_dict(ckpt["optimizer"])
    # scaler
    scaler = amp.GradScaler(enabled=not args.disable_amp)
    scaler.load_state_dict(ckpt["scaler"])
    return model, args, scaler, optimizer


def load_random_data(args):
    if args.fused_decode_crop:
        train_transform = None
    else:
        train_transform_ls = [
            Permute([3, 0, 1, 2]),
            GroupMultiScaleCrop(224, [1, 0.875, 0.75, 0.66]),
            torchvision.transforms.RandomHorizontalFlip(0.5),
        ]
        train_transform = torchvision.transforms.Compose(train_transform_ls)
    train_dataset = KineticsDataset(
        args.root,
        args.train_metadata,
        transform=train_transform,
        is_training=True,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        threads=args.decode_threads,
        fast_rrc=False,
        rrc_params=(224, (0.5, 1.0)),
        fast_msc=args.fused_decode_crop,
        msc_params=(224,),
        fast_cc=False,
        cc_params=(224,),
        hflip_prob=0.5,
        vflip_prob=0.0,
        mask_type="later",  # do masking in batches
        window_size=args.window_size,
        mask_ratio=args.mask_ratio,
        verbose=args.verbose,
        num_samples=args.num_samples,
    )
    train_sampler = None
    if args.use_multi_epochs_loader:
        train_loader = MultiEpochsDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=args.use_pin_memory,
            sampler=train_sampler,
            drop_last=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=args.use_pin_memory,
            sampler=train_sampler,
            drop_last=True,
        )
    data = next(iter(train_loader))
    return data


def load_data(output_dir, args):
    try:
        data = torch.load(output_dir + "/data_err.pt")
        print("loaded the data when error happened")
    except FileNotFoundError:
        data = load_random_data(args)
        print("loaded random data from training dataset")
    inputs, video_ids, frame_ids = data
    mean, std = [108.3272985, 116.7460125, 104.09373615000001], [
        68.5005327,
        66.6321579,
        70.32316305,
    ]
    normalize = K.enhance.Normalize(mean=mean, std=std)
    videos = inputs.cuda(args.gpu, non_blocking=True)
    if args.fused_decode_crop:
        videos = videos.permute(0, 4, 1, 2, 3)
    bool_masked_pos = (
        TubeMaskingGeneratorGPU(
            videos.shape[0],
            args.window_size,
            args.mask_ratio,
            device=args.gpu,
        )()
        .flatten(1)
        .to(torch.bool)
    )
    cls_masked_pos = (
        torch.zeros((args.batch_size, 1))
        .to(bool_masked_pos.dtype)
        .to(bool_masked_pos.device)
    )
    bool_masked_cls_pos = torch.concat([cls_masked_pos, bool_masked_pos], -1)

    if args.normalize_target:
        videos_squeeze = rearrange(
            videos,
            "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c",
            p0=1,
            p1=args.patch_size[0],
            p2=args.patch_size[1],
        )
        videos_norm = (
            videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
        ) / (
            videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt()
            + 1e-6
        )
        videos_patch = rearrange(videos_norm, "b n p c -> b n (p c)")
    else:
        videos_patch = rearrange(
            videos,
            "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)",
            p0=1,
            p1=args.patch_size[0],
            p2=args.patch_size[1],
        )

    B, _, C = videos_patch.shape
    targets = videos_patch[bool_masked_pos].reshape(B, -1, C)

    videos = normalize(videos)
    return videos, video_ids, frame_ids, bool_masked_cls_pos, targets


def get_hist(history, layer):
    history_ls = history[layer]
    value_range = (
        np.concatenate(history_ls).min(),
        np.concatenate(history_ls).max(),
    )
    hist = []
    for history_epoch in history_ls:
        histogram = np.histogram(history_epoch, bins=100, range=value_range)
        hist.append(histogram[0])
    hist = np.stack(hist)
    return hist, histogram[1][1:]


def get_gradient_parameter_history(
    output_dir, model, args, videos, bool_masked_cls_pos, targets
):
    gradients = {}
    parameters = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradients[name] = []
            parameters[name] = []
    criterion = torch.nn.MSELoss().cuda(args.gpu)
    epoch = 10
    while True:
        try:
            model, _, scaler, optimizer = load_ckpt(
                output_dir, epoch=epoch, gpu=args.gpu
            )
            print(f"loaded epoch {epoch} checkpoint")
            epoch += 10
        except FileNotFoundError:
            break
        model = model.to(f"cuda:{args.gpu}")
        model.train()
        optimizer.zero_grad()
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(videos, bool_masked_cls_pos)
            loss = criterion(outputs, target=targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip_norm, norm_type=2.0
        )
        for name, param in model.named_parameters():
            if param.requires_grad:
                # print(f"Layer: {name}, Gradient: {param.grad}")
                parameters[name].append(param.detach().cpu().squeeze().numpy())
                gradients[name].append(
                    param.grad.detach().cpu().squeeze().numpy()
                )
        model.zero_grad(set_to_none=True)
    return gradients, parameters, epoch


def plot_history(history, output_dir, epochs, set_name):
    os.makedirs(os.path.join(output_dir, set_name), exist_ok=True)
    layer = list(history.keys())[0]
    for layer in history:
        hist, bins = get_hist(history, layer)
        plt.figure()
        plt.pcolor(bins, np.arange(10, epochs - 1, 10), hist)
        plt.colorbar()
        plt.xlabel(set_name)
        plt.ylabel("epoch")
        plt.title(layer)
        plt.savefig(os.path.join(output_dir, set_name, f"{layer}.png"))
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()
    print("load checkpoint")
    model, train_args, scaler, optimizer = load_ckpt(
        args.output_dir, epoch="last"
    )
    print("load data")
    videos, video_ids, frame_ids, bool_masked_cls_pos, targets = load_data(
        args.output_dir, train_args
    )
    print("get gradients and parameters")
    gradients, parameters, epochs = get_gradient_parameter_history(
        args.output_dir,
        model,
        train_args,
        videos,
        bool_masked_cls_pos,
        targets,
    )
    print("plot results")
    plot_history(gradients, args.output_dir, epochs, "gradients")
    plot_history(parameters, args.output_dir, epochs, "parameters")


# %%
# output_dir = "experiments/trial1"
# # %%
# epoch = "err"
# model, args, scaler, optimizer = load_ckpt(output_dir, epoch=epoch)
#
# # %%
# videos, video_ids, frame_ids, bool_masked_cls_pos, targets = load_data(
#     output_dir, args
# )
# # %%

# # %%
# layer = list(gradients.keys())[0]
# for layer in gradients:
#     hist, bins = get_hist(gradients, layer)
#     plt.figure()
#     plt.pcolor(bins, np.arange(10, 61, 10), hist)
#     plt.colorbar()
#     plt.xlabel("gradient")
#     plt.ylabel("epoch")
#     plt.title(layer)
#     plt.savefig(
#         os.path.join(output_dir, "gradient_distribution", f"{layer}.png")
#     )
#     plt.close()
# %%
