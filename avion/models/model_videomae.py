from collections import OrderedDict
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from flash_attn.modules.mha import MHA as FlashMHA
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from avion.models import transformer
from avion.models.utils import inflate_positional_embeds


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 400,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        **kwargs,
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # comment this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
        use_flash_attn=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if not use_flash_attn:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim,
            )
        else:
            self.attn = FlashMHA(
                dim,
                num_heads,
                cross_attn=False,
                qkv_proj_bias=qkv_bias,
                dropout=attn_drop,
                use_flash_attn=True,
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_frames=16,
        tubelet_size=2,
        channel_last=False,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (num_frames // self.tubelet_size)
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.channel_last = channel_last
        if channel_last:
            self.proj = nn.Linear(
                in_features=in_chans
                * tubelet_size
                * patch_size[0]
                * patch_size[1],
                out_features=embed_dim,
            )
        else:
            self.proj = nn.Conv3d(
                in_channels=in_chans,
                out_channels=embed_dim,
                kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
                stride=(self.tubelet_size, patch_size[0], patch_size[1]),
            )

    def forward(self, x, **kwargs):
        if self.channel_last:
            x = rearrange(
                x,
                "b c (t p0) (h p1) (w p2) -> b (t h w) (c p0 p1 p2)",
                p0=self.tubelet_size,
                p1=self.patch_size[0],
                p2=self.patch_size[1],
            )
            # x = rearrange(x, 'b (t h w) (p0 p1 p2) c -> b (t h w) (c p0 p1 p2)')
            x = self.proj(x)
            return x
        else:
            B, C, T, H, W = x.shape
            # FIXME look at relaxing size constraints
            assert (
                H == self.img_size[0] and W == self.img_size[1]
            ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            x = self.proj(x).flatten(2).transpose(1, 2)
            return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False
    ).unsqueeze(0)


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        fc_drop_rate=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.0,
        use_learnable_pos_emb=False,
        init_scale=0.001,
        all_frames=16,
        tubelet_size=2,
        channel_last=False,
        use_checkpoint=False,
        use_flash_attn=False,
        use_mean_pooling=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=all_frames,
            tubelet_size=self.tubelet_size,
            channel_last=channel_last,
        )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(
                num_patches, embed_dim
            )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    use_flash_attn=use_flash_attn,
                )
                for i in range(depth)
            ]
        )
        self.norm = (
            nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        )
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fc_dropout = (
            nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        )
        self.head = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = (
                x
                + self.pos_embed.expand(B, -1, -1)
                .type_as(x)
                .to(x.device)
                .clone()
                .detach()
            )
        x = self.pos_drop(x)

        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(self.fc_dropout(x))
        return x


class PretrainVisionTransformerEncoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        tubelet_size=2,
        use_checkpoint=False,
        use_flash_attn=False,
        channel_last=False,
        use_learnable_pos_emb=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            tubelet_size=tubelet_size,
            channel_last=channel_last,
        )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim)
            )
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(
                num_patches, embed_dim
            )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    use_flash_attn=use_flash_attn,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

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
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward_features(self, x, mask):
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible

        if self.use_checkpoint:
            for blk in self.blocks:
                x_vis = checkpoint.checkpoint(blk, x_vis)
        else:
            for blk in self.blocks:
                x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x


class PretrainVisionTransformerDecoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        patch_size=16,
        num_classes=768,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        num_patches=196,
        tubelet_size=2,
        use_checkpoint=False,
        use_flash_attn=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size**2
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    use_flash_attn=use_flash_attn,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

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
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward(self, x, return_token_num):
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        if return_token_num > 0:
            x = self.head(
                self.norm(x[:, -return_token_num:])
            )  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x


class PretrainVisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        encoder_in_chans=3,
        encoder_num_classes=0,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_num_classes=1536,  #  decoder_num_classes=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.0,
        use_learnable_pos_emb=False,
        use_checkpoint=False,
        use_flash_attn_at_encoder=False,
        use_flash_attn_at_decoder=False,
        tubelet_size=2,
        channel_last=False,
        num_classes=0,  # avoid the error from create_fn in timm
        in_chans=0,  # avoid the error from create_fn in timm
    ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
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
            use_flash_attn=use_flash_attn_at_encoder,
            use_learnable_pos_emb=use_learnable_pos_emb,
            channel_last=channel_last,
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
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

        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim
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


def VIDEOMAE_VITB16(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def VIDEOMAE_VITL16(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
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
        if mask is not None:
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
        decoder_depth=4,
        decoder_num_heads=16,
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
        clip_ckpt_path=None,
    ):
        super().__init__()
        # load CLIP checkpoint
        ckpt = torch.load(clip_ckpt_path, map_location="cpu")
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
        # delete unused layers
        del self.encoder.ln_post
        del self.encoder.image_projection
        # make ViT decoder
        self.decoder = PretrainVisionTransformerDecoder(
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
        self.pos_embed = get_sinusoid_encoding_table(
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


class FeatureVisionTransformerCLIP2MAE(PretrainVisionTransformerCLIP2MAE):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_num_classes=1536,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=16,
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
        num_classes=0,
        in_chans=0,
        encoder_in_chans=3,
        encoder_num_classes=0,
        num_frames=16,
        clip_ckpt_path=None,
    ):
        super().__init__(
            img_size,
            patch_size,
            encoder_embed_dim,
            encoder_depth,
            encoder_num_heads,
            decoder_num_classes,
            decoder_embed_dim,
            decoder_depth,
            decoder_num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            init_values,
            use_learnable_pos_emb,
            use_checkpoint,
            use_flash_attn_at_encoder,
            use_flash_attn_at_decoder,
            tubelet_size,
            channel_last,
            num_classes,
            in_chans,
            encoder_in_chans,
            encoder_num_classes,
            num_frames,
            clip_ckpt_path,
        )
        del self.encoder_to_decoder
        del self.pos_embed
        del self.decoder

    def forward(self, x):
        x_vis = self.encoder(x)
        x_vis = x_vis[:, 1:, :]  # remove cls token
        x_vis = self.ln_post(x_vis)
        return x_vis


def VIDEOMAE_CLIP_VITL14(pretrained=False, **kwargs):
    model = PretrainVisionTransformerCLIP2MAE(
        img_size=224,
        patch_size=14,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        decoder_num_classes=588,
        mlp_ratio=4,
        num_frames=16,
        # clip_ckpt_path="checkpoints/avion_pretrain_lavila_vitl_best.pt",
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def VIDEOMAE_CLIP_VITL14_FEATURE(pretrained=False, **kwargs):
    model = FeatureVisionTransformerCLIP2MAE(
        img_size=224,
        patch_size=14,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        decoder_num_classes=588,
        mlp_ratio=4,
        num_frames=16,
        # clip_ckpt_path="checkpoints/avion_pretrain_lavila_vitl_best.pt",
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
