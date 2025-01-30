# %%
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from avion.data.transforms import (
    GroupMultiScaleCrop,
    Permute,
    TubeMaskingGeneratorGPU,
)
from avion.models import model_clip, model_videomae, transformer, utils
from avion.models.utils import inflate_positional_embeds


class PretrainVisionTransformerEncoderCLIP2MAE(transformer.VisionTransformer):

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        num_frames: int = 1,
        ls_init_value: float = None,
        global_average_pool: bool = False,
        output_dim: int = None,
        patch_dropout: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        ln_pre: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
        use_fast_conv1: bool = False,
        use_flash_attn: bool = False,
    ):
        super().__init__(
            image_size,
            patch_size,
            width,
            layers,
            heads,
            mlp_ratio,
            num_frames,
            ls_init_value,
            global_average_pool,
            output_dim,
            patch_dropout,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            ln_pre,
            act_layer,
            norm_layer,
            use_fast_conv1,
            use_flash_attn,
        )
        self.num_patches = (
            num_frames
            * (image_size // patch_size)
            * (image_size // patch_size)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if self.use_fast_conv1:
            if self.num_frames == 1:
                x = rearrange(
                    x,
                    "b c (hh sh) (ww sw) -> b (hh ww) (c sh sw)",
                    sh=self.patch_size[0],
                    sw=self.patch_size[1],
                )
                x = self.conv1(x)
                x = torch.cat(
                    [
                        self.class_embedding.to(x.dtype)
                        + torch.zeros(
                            x.shape[0],
                            1,
                            x.shape[-1],
                            dtype=x.dtype,
                            device=x.device,
                        ),
                        x,
                    ],
                    dim=1,
                )  # shape = [*, grid ** 2 + 1, width]
                x = x + self.positional_embedding.to(x.dtype)
            else:
                x = rearrange(
                    x,
                    "b c t (hh sh) (ww sw) -> b (t hh ww) (c sh sw)",
                    sh=self.patch_size[0],
                    sw=self.patch_size[1],
                )
                x = self.conv1(x)
                x = torch.cat(
                    [
                        self.class_embedding.to(x.dtype)
                        + torch.zeros(
                            x.shape[0],
                            1,
                            x.shape[-1],
                            dtype=x.dtype,
                            device=x.device,
                        ),
                        x,
                    ],
                    dim=1,
                )  # shape = [*, grid ** 2 + 1, width]
                cls_embed = self.positional_embedding[0, :].unsqueeze(0)
                tile_pos_embed = self.positional_embedding[1:, :].repeat(
                    self.num_frames, 1
                )
                tile_temporal_embed = (
                    self.temporal_embedding.repeat_interleave(
                        self.patches_per_frame, 0
                    )
                )
                total_pos_embed = tile_pos_embed + tile_temporal_embed
                total_pos_embed = torch.cat(
                    [cls_embed, total_pos_embed], dim=0
                )
                x = x + total_pos_embed.to(x.dtype).unsqueeze(0)
        else:
            if self.num_frames == 1:
                x = self.conv1(x)  # shape = [*, width, grid, grid]
                x = x.reshape(
                    x.shape[0], x.shape[1], -1
                )  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                x = torch.cat(
                    [
                        self.class_embedding.to(x.dtype)
                        + torch.zeros(
                            x.shape[0],
                            1,
                            x.shape[-1],
                            dtype=x.dtype,
                            device=x.device,
                        ),
                        x,
                    ],
                    dim=1,
                )  # shape = [*, grid ** 2 + 1, width]
                x = x + self.positional_embedding.to(x.dtype)
            else:
                x = x.permute(
                    0, 2, 1, 3, 4
                ).contiguous()  # B, C, T, H, W =>  B, T, C, H, W
                B, F, C, H, W = x.shape
                x = x.view(-1, C, H, W)
                x = self.conv1(x)
                x = x.flatten(2).transpose(2, 1)  # BT, C', H, W => BT, HW, C'
                x = x.reshape(B, -1, self.width)
                x = torch.cat(
                    [
                        self.class_embedding.to(x.dtype)
                        + torch.zeros(
                            x.shape[0],
                            1,
                            x.shape[-1],
                            dtype=x.dtype,
                            device=x.device,
                        ),
                        x,
                    ],
                    dim=1,
                )  # shape = [*, grid ** 2 + 1, width]
                cls_embed = self.positional_embedding[0, :].unsqueeze(0)
                tile_pos_embed = self.positional_embedding[1:, :].repeat(
                    self.num_frames, 1
                )
                tile_temporal_embed = (
                    self.temporal_embedding.repeat_interleave(
                        self.patches_per_frame, 0
                    )
                )
                total_pos_embed = tile_pos_embed + tile_temporal_embed
                total_pos_embed = torch.cat(
                    [cls_embed, total_pos_embed], dim=0
                )
                x = x + total_pos_embed.to(x.dtype).unsqueeze(0)
        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.pos_drop(x)
        # masking
        B, _, C = x.shape
        x = x[~mask].reshape(B, -1, C)
        if not self.use_flash_attn:
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        else:
            x = self.transformer(x)
        return x


class PretrainVisionTransformerCLIP2MAE(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_num_classes=1536,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0,
        norm_layer=nn.LayerNorm,
        init_values=0,
        use_learnable_pos_emb=False,
        use_checkpoint=False,
        use_flash_attn_at_encoder=False,
        use_flash_attn_at_decoder=False,
        tubelet_size=1,
        channel_last=False,
        num_classes=0,  # avoid the error from create_fn in timm
        in_chans=0,  # avoid the error from create_fn in timm
        encoder_in_chans=3,  # avoid the error from create_fn in timm
        encoder_num_classes=0,  # avoid the error from create_fn in timm
        num_frames=16,
        ckpt_path=None,
    ):
        super().__init__()
        # load CLIP checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        old_args = ckpt["args"]
        state_dict = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            if "module.visual" in k:
                state_dict[k.replace("module.visual.", "")] = v
        # make ViT encoder
        self.encoder = PretrainVisionTransformerEncoderCLIP2MAE(
            image_size=img_size,
            patch_size=patch_size,
            width=encoder_embed_dim,
            layers=encoder_depth,
            heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            output_dim=old_args.project_embed_dim,
            patch_dropout=old_args.patch_dropout,
            use_fast_conv1=old_args.use_fast_conv1,
            use_flash_attn=old_args.use_flash_attn,
            num_frames=num_frames,
        )
        # bilinear interpolation of temporal positional embedding
        state_dict = inflate_positional_embeds(
            self.encoder.state_dict(),
            state_dict,
            num_frames=num_frames,
            load_temporal_fix="bilinear",
            temporal_embedding_key="temporal_embedding",
        )
        # load CLIP checkpoint
        self.encoder.load_state_dict(state_dict, strict=True)
        # make ViT decoder
        self.decoder = model_videomae.PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_flash_attn=use_flash_attn_at_decoder,
        )
        self.ln_post = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed = model_videomae.get_sinusoid_encoding_table(
            self.encoder.num_patches, decoder_embed_dim
        )
        trunc_normal_(self.mask_token, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "mask_token"}

    def forward(self, x, mask):
        _, _, T, _, _ = x.shape
        x_vis = self.encoder(x, mask)  # [B, N_vis, C_e]
        x_vis = x_vis[:, 1:, :]  # remove cls token
        mask = mask[:, 1:]  # remove cls token
        x_vis = self.ln_post(x_vis)
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = (
            self.pos_embed.expand(B, -1, -1)
            .type_as(x)
            .to(x.device)
            .clone()
            .detach()
        )
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat(
            [x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1
        )  # [B, N, C_d]
        x = self.decoder(
            x_full, pos_emd_mask.shape[1]
        )  # [B, N_mask, 3 * 16 * 16]

        return x


# %%
ckpt_path = "checkpoints/avion_pretrain_lavila_vitl_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")
old_args = ckpt["args"]
state_dict = OrderedDict()
for k, v in ckpt["state_dict"].items():
    if "module.visual" in k:
        state_dict[k.replace("module.visual.", "")] = v
# %%
clip_length = 16
encoder = PretrainVisionTransformerEncoderCLIP2MAE(
    image_size=224,
    patch_size=14,
    width=1024,
    layers=24,
    heads=16,
    mlp_ratio=4,
    output_dim=old_args.project_embed_dim,
    patch_dropout=old_args.patch_dropout,
    num_frames=clip_length,
    use_fast_conv1=old_args.use_fast_conv1,
    use_flash_attn=old_args.use_flash_attn,
)
state_dict = inflate_positional_embeds(
    encoder.state_dict(),
    state_dict,
    num_frames=clip_length,
    load_temporal_fix="bilinear",
    temporal_embedding_key="temporal_embedding",
)
encoder.load_state_dict(state_dict, strict=True)
# %%
x = torch.randn((16, 3, clip_length, 224, 224), dtype=torch.float32)
x = x.to("cuda")
encoder = encoder.to("cuda")
# %%
input_size = 224
patch_size = (14, 14)
window_size = (
    clip_length,
    input_size // patch_size[0],
    input_size // patch_size[1],
)
mask_ratio = 0.9
bool_masked_pos = (
    TubeMaskingGeneratorGPU(
        x.shape[0], window_size, mask_ratio, device="cuda"
    )()
    .flatten(1)
    .to(torch.bool)
)
cls_masked_pos = (
    torch.zeros((clip_length, 1))
    .to(bool_masked_pos.dtype)
    .to(bool_masked_pos.device)
)
bool_masked_pos = torch.concat([cls_masked_pos, bool_masked_pos], -1)
# %%
encoder.eval()
with amp.autocast(enabled=True):
    with torch.no_grad():
        y = encoder(x, bool_masked_pos)
print(y.shape)
# %%
model = PretrainVisionTransformerCLIP2MAE(
    img_size=224,
    patch_size=14,
    encoder_embed_dim=1024,
    encoder_depth=24,
    encoder_num_heads=16,
    decoder_num_classes=588,
    mlp_ratio=4,
    num_frames=16,
    ckpt_path=ckpt_path,
)
# %%
model.eval()
model = model.to("cuda")
with amp.autocast(enabled=True):
    with torch.no_grad():
        y = model(x, bool_masked_pos)
# %%
