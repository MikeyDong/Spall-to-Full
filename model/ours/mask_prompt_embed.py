import math
from typing import Optional, Callable, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class MaskPatchEmbed(nn.Module):
    """
    Encode three-channel one-hot semantic masks into multi-scale patch tokens.
    Adapted from the SparseDepthEmbed design in Any2Full.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        image_hw = make_2tuple(img_size)
        patch_hw = make_2tuple(patch_size)
        patch_grid_size = (
            image_hw[0] // patch_hw[0],
            image_hw[1] // patch_hw[1],
        )

        self.img_size = image_hw
        self.patch_size = patch_hw
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.embed_dim = embed_dim

        p = patch_hw[0]
        p2 = max(1, p // 2)
        p4 = max(1, p // 4)

        # Multi-scale patch encoding for dense one-hot semantic masks.
        self.mask_encoder = nn.ModuleList([
            nn.Conv2d(in_chans, embed_dim, kernel_size=p, stride=p),
            nn.Conv2d(in_chans, embed_dim, kernel_size=p2, stride=p2),
            nn.Conv2d(in_chans, embed_dim, kernel_size=p4, stride=p4),
        ])

        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W]，one-hot mask
        Returns:
            x_feat: [B, N, C]
            patch_mask: [B, N, 1]，这里 mask 是稠密的，因此全 1。
        """
        b, _, h, w = x.shape
        target_h = h // self.patch_size[0]
        target_w = w // self.patch_size[1]

        feat_list = []
        for conv in self.mask_encoder:
            feat = conv(x)  # [B, C, h', w']
            feat = F.adaptive_avg_pool2d(feat, (target_h, target_w))
            feat = feat.flatten(2).transpose(1, 2)  # [B, N, C]
            feat_list.append(feat)

        x_feat = torch.stack(feat_list, dim=0).sum(dim=0)
        if self.norm is not None:
            x_feat = self.norm(x_feat)

        patch_mask = torch.ones(
            (b, x_feat.shape[1], 1),
            dtype=x_feat.dtype,
            device=x_feat.device,
        )
        return x_feat, patch_mask


class MaskPromptEmbed(nn.Module):
    """
    Semantic mask-prompt encoder for background, intact concrete, and spalled concrete.
    """

    def __init__(
        self,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.patch_embed = MaskPatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.num_tokens = 1
        self.interpolate_offset = 0.1
        self.interpolate_antialias = False
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        # Multiplicative and additive alignment with RGB/fused feature tokens.
        self.rgbm_proj_s = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.rgbm_proj_b = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        n = self.pos_embed.shape[1] - 1
        if npatch == n and w == h:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_n = math.sqrt(n)
        sx, sy = float(w0) / sqrt_n, float(h0) / sqrt_n
        if torch.__version__ >= '2.0.0':
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(sqrt_n), int(sqrt_n), dim).permute(0, 3, 1, 2),
                scale_factor=(sx, sy),
                mode='bicubic',
                antialias=self.interpolate_antialias,
            )
        else:
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(sqrt_n), int(sqrt_n), dim).permute(0, 3, 1, 2),
                scale_factor=(sx, sy),
                mode='bicubic',
            )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, rgb_feat_l):
        """
        Args:
            x: [B, 3, H, W]，one-hot mask
            rgb_feat_l: [B, N+1, C]，Reference features used for aligning the token space.
        Returns:
            x: [B, N+1, C]
            mask: [B, N+1, 1]
        """
        b, _, w, h = x.shape
        x, mask = self.patch_embed(x)

        x_mean = x.mean(dim=1, keepdim=True)
        x = torch.cat((x_mean, x), dim=1)
        mask = torch.cat((torch.ones([1, 1]).expand(x.shape[0], -1, -1).to(mask), mask), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)

        rgb_feat = rgb_feat_l.detach()
        x = rgb_feat * (self.rgbm_proj_s(torch.cat((rgb_feat, x), dim=-1)) + 1) + self.rgbm_proj_b(torch.cat((rgb_feat, x), dim=-1))
        return x, mask

    def forward(self, x, rgb_feat_l):
        x, mask = self.prepare_tokens_with_masks(x, rgb_feat_l)
        return x, mask
