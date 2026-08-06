import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeRegionRouting(nn.Module):
    """
    Route fused features according to background, intact-concrete,
    and spalled-concrete semantic regions.
    """

    def __init__(self, channels: int, concrete_kernel_size: int = 5, spall_kernel_size: int = 5):
        super().__init__()
        self.channels = channels
        self.concrete_kernel_size = concrete_kernel_size
        self.spall_kernel_size = spall_kernel_size


        self.control_head = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(channels, 2, kernel_size=1, stride=1, padding=0),
        )


        self.concrete_refine = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.spall_refine = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def _masked_local_mean(feat: torch.Tensor, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Compute a local feature mean restricted to the specified semantic region.
        """
        pad = kernel_size // 2
        feat_sum = F.avg_pool2d(feat * mask, kernel_size=kernel_size, stride=1, padding=pad) * (kernel_size * kernel_size)
        mask_sum = F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad) * (kernel_size * kernel_size)
        local_mean = feat_sum / (mask_sum + 1e-6)
        return local_mean

    def forward(self, fused_feat: torch.Tensor, mask_prompt_feat: torch.Tensor, hard_mask: torch.Tensor) -> torch.Tensor:

        assert fused_feat.shape == mask_prompt_feat.shape, \
            'fused_feat and mask_prompt_feat must have identical shapes'

        assert hard_mask.shape[1] == 3, \
            'hard_mask must contain three one-hot semantic channels'


        mask_bg = hard_mask[:, 0:1, :, :]
        mask_concrete = hard_mask[:, 1:2, :, :]
        mask_spall = hard_mask[:, 2:3, :, :]

        # -----------------------------
        # Lightweight control head
        # -----------------------------

        control_input = torch.cat([fused_feat, mask_prompt_feat], dim=1)
        control_maps = self.control_head(control_input)


        bg_ctrl = torch.sigmoid(control_maps[:, 0:1, :, :])


        spall_ctrl = torch.sigmoid(control_maps[:, 1:2, :, :])

        # -----------------------------
        # Background branch
        # -----------------------------

        foreground = torch.clamp(mask_concrete + mask_spall, min=0.0, max=1.0)
        foreground_smooth = F.avg_pool2d(foreground, kernel_size=7, stride=1, padding=3)

        bg_scale = 0.2 + 0.3 * (0.5 * foreground_smooth + 0.5 * bg_ctrl)
        bg_scale = torch.clamp(bg_scale, min=0.2, max=0.5)

        bg_branch = mask_bg * (bg_scale * fused_feat) + (1.0 - mask_bg) * (0.0 * fused_feat)

        # -----------------------------
        # Intact-concrete branch
        # -----------------------------

        concrete_mean = self._masked_local_mean(fused_feat, mask_concrete, self.concrete_kernel_size)


        concrete_branch = mask_concrete * self.concrete_refine(concrete_mean) + (1.0 - mask_concrete) * (0.0 * fused_feat)

        # -----------------------------
        # Spalled-concrete branch
        # -----------------------------

        spall_mean = self._masked_local_mean(fused_feat, mask_spall, self.spall_kernel_size)


        spall_residual = fused_feat - spall_mean


        spall_residual_coef = 0.5 + 0.5 * spall_ctrl


        spall_feat = spall_mean + spall_residual_coef * spall_residual
        spall_branch = mask_spall * self.spall_refine(spall_feat) + (1.0 - mask_spall) * (0.0 * fused_feat)

        # -----------------------------
        # Merge region-specific features
        # -----------------------------
        routed_feat = bg_branch + concrete_branch + spall_branch
        return routed_feat
