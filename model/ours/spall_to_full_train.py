import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ours.config import model_configs
from model.ours.logger import Log
from model.ours.depth_anything_v2.dpt import DPTHead
from model.ours.prompt_dinov2 import DINOv2 as PromptDINOv2
from model.ours.mask_prompt_embed import MaskPromptEmbed
from model.ours.three_region_routing import ThreeRegionRouting

# NOTICE: The class name Any2Full includes the new module purchase, but does not include post-processing methods.
# NOTICE: The class name is just a name, the essence is Spall to Full
class Any2Full(nn.Module):
    def __init__(self, encoder='vitl', da_ckpt_path='checkpoints/promptda_vitl.ckpt', args=None):
        super().__init__()
        self.args = args
        self.patch_size = 14
        self.use_bn = False
        self.use_clstoken = False
        self.output_act = 'None'

        model_config = model_configs[encoder]
        self.encoder = encoder
        self.model_config = model_config
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39],
        }

        self.pretrained = PromptDINOv2(
            model_name=encoder,
            blocks_to_take_list=self.intermediate_layer_idx[encoder],
        )

        dim = self.pretrained.blocks[0].attn.qkv.in_features

        # 原始 Any2Full：Depth Prompt 与 RGB 特征的逐层融合头（保持不变）
        self.pretrained_prompt_depth_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            for _ in range(len(self.intermediate_layer_idx[encoder]) - 1)
        ])
        self.pretrained_prompt_depth_scale = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            for _ in range(len(self.intermediate_layer_idx[encoder]) - 1)
        ])

        # 新增：Mask Prompt 编码器
        self.prompt_mask_embedding = MaskPromptEmbed(
            patch_size=self.patch_size,
            in_chans=3,
            embed_dim=dim,
        )

        # 新增：每个尺度一个三区域路由模块
        self.three_region_routing = nn.ModuleList([
            ThreeRegionRouting(channels=dim, concrete_kernel_size=5, spall_kernel_size=5)
            for _ in range(len(self.intermediate_layer_idx[encoder]))
        ])

        self.infer_time = 0

        self.depth_head = DPTHead(
            in_channels=dim,
            features=model_config['features'],
            out_channels=model_config['out_channels'],
            use_bn=self.use_bn,
            use_clstoken=self.use_clstoken,
        )

        if da_ckpt_path is not None:
            self.load_pretrainedDA(da_ckpt_path)
            print('Monodcular Depth Model-Encoder FREEZE !!')
            for name, var in self.pretrained.named_parameters():
                if 'prompt_depth' not in name:
                    var.requires_grad = False
            print('Monodcular Depth Model-Decoder Bias Tuning !!')
            for _, var in self.depth_head.named_parameters():
                var.requires_grad = False

    def load_pretrainedDA(self, da_ckpt_path):
        if os.path.exists(da_ckpt_path):
            Log.info(f'Loading pretrained DepthAnything checkpoint from {da_ckpt_path}')
            checkpoint = torch.load(da_ckpt_path, map_location='cpu')
            if self.args.stage == 1:
                missing_keys = set(dict(self.named_parameters()).keys()) - set(checkpoint.keys())
                print('\nMissing keys:')
                for key in missing_keys:
                    print(key)
                self.load_state_dict(checkpoint, strict=False)
                for name, var in self.pretrained.named_parameters():
                    if 'prompt_depth_' in name:
                        base_name = name.replace('prompt_depth_', '')
                        if hasattr(self.pretrained, base_name):
                            base_var = getattr(self.pretrained, base_name)
                            var.data.copy_(base_var.data)
                            print(f'Copied data from {base_name} to {name}')

            del checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            Log.warn(f'Checkpoint {da_ckpt_path} not found')

    def _tokens_to_map(self, tokens: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        """[B, N, C] -> [B, C, patch_h, patch_w]"""
        b, n, c = tokens.shape
        assert n == patch_h * patch_w, f'token 数量 {n} 与 patch 网格 {patch_h}x{patch_w} 不匹配'
        return tokens.reshape(b, patch_h, patch_w, c).permute(0, 3, 1, 2).contiguous()

    def _map_to_tokens(self, feat_map: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, N, C]"""
        return feat_map.flatten(2).transpose(1, 2).contiguous()

    def forward(self, x, prompt_depth=None):
        start_time = time.time()
        resize_mode = 'resize'

        if prompt_depth is None:
            prompt_depth = x['dep']
            rgb = x['rgb']
            # 新增：语义 mask（3 通道 one-hot）
            mask_onehot = x.get('mask', None)
        else:
            rgb = x
            mask_onehot = None

        if resize_mode == 'pad':
            rgb, pad = self.pad_to_multiple(rgb, mode='replicate')
            prompt_depth, pad = self.pad_to_multiple(prompt_depth, mode='constant')
            if mask_onehot is not None:
                mask_onehot, _ = self.pad_to_multiple(mask_onehot, mode='constant')
        elif resize_mode == 'resize':
            rgb, diff = self.resize_to_multiple(rgb, mode='bicubic')
            prompt_depth, diff = self.resize_to_multiple(prompt_depth, mode='nearest')
            if mask_onehot is not None:
                mask_onehot, _ = self.resize_to_multiple(mask_onehot, mode='nearest')

        # 原始 Any2Full：将输入稀疏深度转为内部使用的相对量，并做样本内标准化
        prompt_disparity = self.disparity_to_depth(prompt_depth)
        bias, scale = self.get_depth_bias_scale(prompt_disparity)
        prompt_disparity = (prompt_disparity - bias.view(-1, 1, 1, 1).detach()) / (scale.view(-1, 1, 1, 1).detach())
        assert torch.isfinite(prompt_disparity).all(), 'Input contains nan or inf'

        # 从改造后的 encoder 中取出多层 RGB token + Depth Prompt token
        features = self.pretrained.get_intermediate_layers(
            rgb,
            prompt_disparity,
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )

        h, w = rgb.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        fused_features = []
        for i, feat_pack in enumerate(features):
            rgb_tokens = feat_pack[0]          # [B, N, C]
            cls_token = feat_pack[1]          # [B, C]

            # 1) 先保持原始 Any2Full 的 RGB + Depth Prompt 融合不变
            if i == 0:
                fused_x = rgb_tokens
            else:
                prompt_v = feat_pack[-1][0]   # [B, N, C]，Depth Prompt 空间 token
                fused_x = (
                    (self.pretrained_prompt_depth_scale[i - 1](prompt_v) + 1) * rgb_tokens
                    + self.pretrained_prompt_depth_fusion[i - 1](torch.cat((rgb_tokens, prompt_v), dim=-1))
                )

            # 2) 若提供了 one-hot mask，则再做 Mask Prompt + 三区域路由
            if mask_onehot is not None:
                # Mask Prompt 与当前层特征做 token 对齐：
                # 这里用“当前层已经融合好的 token + cls token”作为参考 token。
                rgb_like_tokens = torch.cat([cls_token.unsqueeze(1), fused_x], dim=1)
                mask_prompt_tokens, _ = self.prompt_mask_embedding(mask_onehot, rgb_like_tokens)
                mask_prompt_spatial = mask_prompt_tokens[:, 1:, :]  # 去掉 cls token，仅保留 patch token

                # 转成空间特征图，方便局部均值与区域路由处理
                fused_map = self._tokens_to_map(fused_x, patch_h, patch_w)
                mask_prompt_map = self._tokens_to_map(mask_prompt_spatial, patch_h, patch_w)
                mask_down = F.interpolate(mask_onehot.float(), size=(patch_h, patch_w), mode='nearest')

                # 三区域路由：
                # 输入 = [原始融合特征图, Mask Prompt 特征图, 下采样 one-hot mask]
                # 输出维度与尺寸完全不变。
                routed_map = self.three_region_routing[i](fused_map, mask_prompt_map, mask_down)
                fused_x = self._map_to_tokens(routed_map)

            if self.use_clstoken:
                fused_features.append([fused_x, cls_token])
            else:
                fused_features.append([fused_x])

        disparity_pre = self.depth_head(fused_features, patch_h, patch_w, return_feat=False)

        self.infer_time = self.infer_time + time.time() - start_time
        if self.args.init_scailing:
            depth = self.disparity_to_depth(
                torch.clamp(
                    self.init_scailing(disparity_pre, self.disparity_to_depth(prompt_depth)),
                    min=1 / self.args.max_depth,
                )
            )
            depth = torch.clamp(depth, min=self.args.min_depth, max=self.args.max_depth)
        else:
            bias_0, scale_0 = self.get_depth_bias_scale(disparity_pre)
            disparity_pre_norm = (disparity_pre - bias_0.view(-1, 1, 1, 1).detach()) / (scale_0.view(-1, 1, 1, 1).detach())
            depth = self.disparity_to_depth(
                torch.clamp(
                    disparity_pre_norm * scale.view(-1, 1, 1, 1) + bias.view(-1, 1, 1, 1),
                    min=1 / self.args.max_depth,
                )
            )
            depth = torch.clamp(depth, min=self.args.min_depth, max=self.args.max_depth)

        if resize_mode == 'pad':
            depth = self.unpad(depth, pad)
            disparity_pre = self.unpad(disparity_pre, pad)
        elif resize_mode == 'resize':
            depth = self.unresize(depth, diff)
            disparity_pre = self.unresize(disparity_pre, diff)

        output = {
            'pred': depth,
            'disparity_pre': disparity_pre,
            'disparity_ori': None,
            'prompt_depth_features': None,
            'guidance': None,
            'confidence': None,
        }
        return output

    def _concat(self, fd, fe, dim=1):
        _, _, hd, wd = fd.shape
        _, _, he, we = fe.shape
        if hd > he:
            fd = fd[:, :, :-(hd - he), :]
        if wd > we:
            fd = fd[:, :, :, :-(wd - we)]
        return torch.cat((fd, fe), dim=dim)

    def init_scailing(self, pred, sparse, align_points_num=1e10):
        depth = pred.clone().detach()
        for i in range(pred.shape[0]):
            target = sparse[i]
            idx_nnz = torch.nonzero(target.view(-1) > 0.00001, as_tuple=False)
            num_points = idx_nnz.shape[0]
            if num_points > align_points_num:
                perm = torch.randperm(num_points, device=idx_nnz.device)
                idx_nnz = idx_nnz[perm[:align_points_num]]
            b = target.view(-1)[idx_nnz]
            a = depth[i].view(-1)[idx_nnz]
            a = a + torch.rand(*a.shape, device=a.device) * 1e-10
            num_dep = a.shape[0]
            a = torch.cat((a, torch.ones(num_dep, 1).to(a)), dim=1)
            x = torch.pinverse(a) @ b
            x = x.to(pred)
            depth[i] = pred[i] * x[0] + x[1]
        return depth

    def disparity_to_depth(self, disparity):
        disparity = torch.clamp(disparity, min=0)
        eps = 1e-8
        return torch.where(disparity > 0, 1.0 / (disparity + eps), torch.zeros_like(disparity))

    @torch.no_grad()
    def predict(self, image: torch.Tensor, prompt_depth: torch.Tensor):
        return self.forward(image, prompt_depth)

    def normalize(self, prompt_depth: torch.Tensor):
        b, c, h, w = prompt_depth.shape
        min_val = torch.quantile(prompt_depth.reshape(b, -1), 0.0, dim=1, keepdim=True)[:, :, None, None]
        max_val = torch.quantile(prompt_depth.reshape(b, -1), 1.0, dim=1, keepdim=True)[:, :, None, None]
        prompt_depth = (prompt_depth - min_val) / (max_val - min_val + 1e-8)
        return prompt_depth, min_val, max_val

    def denormalize(self, depth: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor):
        return depth * (max_val - min_val + 1e-8) + min_val

    def pad_to_multiple(self, x, multiple_of=14, mode='constant'):
        _, _, h, w = x.shape
        pad_h = (multiple_of - h % multiple_of) % multiple_of
        pad_w = (multiple_of - w % multiple_of) % multiple_of
        padded_x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode)
        return padded_x, (pad_h, pad_w)

    def unpad(self, x, pad):
        pad_h, pad_w = pad
        if pad_h == 0 and pad_w == 0:
            return x
        return x[..., :x.size(-2) - pad_h, :x.size(-1) - pad_w]

    def resize_to_multiple(self, x, multiple_of=14, mode='bilinear', resize_lower_size=1456):
        b, _, h, w = x.shape
        scale = max(resize_lower_size / h, resize_lower_size / w)
        new_h = int(((h * scale + multiple_of - 1) // multiple_of) * multiple_of)
        new_w = int(((w * scale + multiple_of - 1) // multiple_of) * multiple_of)
        if new_h == h and new_w == w:
            return x, (0, 0)
        if b == 1:
            if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
                resized_x = F.interpolate(x, size=(new_h, new_w), mode=mode, align_corners=True)
            else:
                resized_x = F.interpolate(x, size=(new_h, new_w), mode=mode)
        else:
            results = []
            for i in range(b):
                single_x = x[i:i + 1]
                if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
                    single_resized_x = F.interpolate(single_x, size=(new_h, new_w), mode=mode, align_corners=True)
                else:
                    single_resized_x = F.interpolate(single_x, size=(new_h, new_w), mode=mode)
                results.append(single_resized_x)
            resized_x = torch.cat(results, dim=0)
        return resized_x, (new_h - h, new_w - w)

    def unresize(self, x, size_diff):
        h_diff, w_diff = size_diff
        if h_diff == 0 and w_diff == 0:
            return x
        _, _, h, w = x.shape
        return F.interpolate(x, size=(h - h_diff, w - w_diff), mode='bilinear', align_corners=True)

    def get_depth_bias_scale(self, prompt_depth):
        b, c, h, w = prompt_depth.shape
        mask = prompt_depth != 0
        means = torch.zeros(b, device=prompt_depth.device, dtype=prompt_depth.dtype)
        stds = torch.zeros(b, device=prompt_depth.device, dtype=prompt_depth.dtype)
        for i in range(b):
            nonzero_elements = prompt_depth[i][mask[i]]
            if nonzero_elements.numel() > 0:
                mean = nonzero_elements.mean()
                if nonzero_elements.numel() > 1:
                    std = nonzero_elements.std()
                    if torch.isnan(std) or std == 0:
                        print(f'Warning: Std is NaN for sample {i}')
                        std = 1.0
                else:
                    std = 1.0
                means[i] = mean
                stds[i] = std
            else:
                means[i] = 0
                stds[i] = 1
        return means, stds
