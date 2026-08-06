# Adapted from Any2Full (https://github.com/zhiyuandaily/Any2Full).
# Extended with semantic mask prompting, three-region routing, and local residual correction.
import os
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

from model.ours.config import model_configs
from model.ours.logger import Log
from model.ours.depth_anything_v2.dpt import DPTHead
from model.ours.prompt_dinov2 import DINOv2 as PromptDINOv2
from model.ours.mask_prompt_embed import MaskPromptEmbed
from model.ours.three_region_routing import ThreeRegionRouting


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
            model_input_size = int(getattr(self.args, "model_input_size", 1456))
        
            original_h, original_w = rgb.shape[-2:]
        
            # 保留长宽比：短边缩放到约 1456，并调整到 14 的倍数
            rgb, diff = self.resize_to_multiple(
                rgb,
                mode='bicubic',
                resize_lower_size=model_input_size
            )
        
            prompt_depth, diff = self.resize_to_multiple(
                prompt_depth,
                mode='nearest',
                resize_lower_size=model_input_size
            )
        
            if mask_onehot is not None:
                mask_onehot, _ = self.resize_to_multiple(
                    mask_onehot,
                    mode='nearest',
                    resize_lower_size=model_input_size
                )
        

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
                    self.init_scailing(
                        disparity_pre,
                        self.disparity_to_depth(prompt_depth),
                        rgb=rgb,
                        mask_onehot=mask_onehot
                    ),
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

    def _mask_to_type_map(self, mask_onehot_i, h, w, device):
        """
        把 one-hot mask 转成单通道类型图：
        0/1/2 ...
        若没有 mask，则全图默认为同一类型 0
        """
        if mask_onehot_i is None:
            return torch.zeros((h, w), device=device, dtype=torch.long)

        if mask_onehot_i.dim() == 3 and mask_onehot_i.shape[0] > 1:
            return torch.argmax(mask_onehot_i, dim=0).long()

        if mask_onehot_i.dim() == 2:
            return mask_onehot_i.long()

        return torch.zeros((h, w), device=device, dtype=torch.long)

    def _build_local_residual_features(self, pred_i, ls_i, rgb_i, eps=1e-6, std_win=7):
        """
        Construct features for local KDTree-based residual retrieval.

        The residual_use_normalized_features switch is not used here.
        Spatial coordinates x and y are fixed to [-1, 1]. The LS, gradient,
        Laplacian, and local-standard-deviation channels are not min-max
        normalized here. Feature standardization is applied later using
        mean and standard deviation statistics from the KDTree key set.
        """
        device = pred_i.device
        dtype = pred_i.dtype
        h, w = pred_i.shape

        # ---- RGB 自动处理到 [0,1] ----
        rgb_i = rgb_i.to(device=device, dtype=dtype)
        if rgb_i.shape[0] > 3:
            rgb_i = rgb_i[:3]
        if rgb_i.shape[0] == 1:
            rgb_i = rgb_i.repeat(3, 1, 1)

        # 与代码2保持一致：不强行除以 255，只 clamp
        # 如果你输入确实是 0~255，这里需要你自己确认前面是否已经处理过
        rgb_i = torch.clamp(rgb_i, 0.0, 1.0)

        r = rgb_i[0]
        g = rgb_i[1]
        b = rgb_i[2]

        # ---- RGB 转 YCbCr ----
        y = 0.299000 * r + 0.587000 * g + 0.114000 * b
        cb = -0.168736 * r - 0.331264 * g + 0.500000 * b + 0.5
        cr = 0.500000 * r - 0.418688 * g - 0.081312 * b + 0.5

        y = torch.clamp(y, 0.0, 1.0)
        cb = torch.clamp(cb, 0.0, 1.0)
        cr = torch.clamp(cr, 0.0, 1.0)

        # ---- 坐标固定为 [-1,1]，照搬代码2 ----
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing='ij'
        )

        # ---- 基于 LS 基线计算一阶导、二阶导、局部标准差 ----
        ls_4d = ls_i.unsqueeze(0).unsqueeze(0)

        dx = F.pad(ls_4d[:, :, :, 1:] - ls_4d[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(ls_4d[:, :, 1:, :] - ls_4d[:, :, :-1, :], (0, 0, 0, 1))
        grad = torch.sqrt(dx * dx + dy * dy + eps)[0, 0]

        dxx = F.pad(
            ls_4d[:, :, :, 2:] - 2.0 * ls_4d[:, :, :, 1:-1] + ls_4d[:, :, :, :-2],
            (1, 1, 0, 0)
        )
        dyy = F.pad(
            ls_4d[:, :, 2:, :] - 2.0 * ls_4d[:, :, 1:-1, :] + ls_4d[:, :, :-2, :],
            (0, 0, 1, 1)
        )
        lap = torch.abs(dxx + dyy)[0, 0]

        mu = F.avg_pool2d(ls_4d, kernel_size=std_win, stride=1, padding=std_win // 2)
        mu2 = F.avg_pool2d(ls_4d * ls_4d, kernel_size=std_win, stride=1, padding=std_win // 2)
        std = torch.sqrt(torch.clamp(mu2 - mu * mu, min=0.0) + eps)[0, 0]

        # ---- Attention 用特征：照搬代码2，不提前归一化 ----
        feat_map = torch.stack([
            xx,     # x
            yy,     # y
            ls_i,   # LS 基线
            y,      # Y
            cb,     # Cb
            cr,     # Cr
            grad,   # 一阶导
            lap,    # 二阶导
            std     # 局部标准差
        ], dim=-1)


        retrieval_map = feat_map
        return feat_map, retrieval_map

    def _memory_cross_attention_residual(
            self,
            pred_i,
            target_i,
            ls_i,
            rgb_i,
            mask_onehot_i=None,
            topk=8,
            prefilter_factor=4,
            pred_tol_scale=0.5,
            chunk_size=4096
    ):
        """
"""Estimate local residual corrections using KDTree retrieval and top-k attention."""

        修改点：
        - 不使用 local_known_density_gate；
        - Query 为当前 mask 类型下所有像素；
        - 特征构造使用代码2版本；
        - 不理会 residual_use_normalized_features 开关；
        - 其余 KDTree、pred_tol、topk、attention 逻辑保持。
        """
        device = pred_i.device
        dtype = pred_i.dtype
        h, w = pred_i.shape

        known_mask = target_i > 1e-5
        if known_mask.sum() == 0:
            return torch.zeros_like(ls_i)

        # ---- 真实深度只用于构造残差 ----
        residual_map = torch.zeros_like(ls_i)
        residual_map[known_mask] = target_i[known_mask] - ls_i[known_mask]

        # ---- 只在 KDTree 分支中使用代码2的特征构造 ----
        feat_map, retrieval_map = self._build_local_residual_features(
            pred_i,
            ls_i,
            rgb_i
        )

        # ---- mask 类型图 ----
        type_map = self._mask_to_type_map(mask_onehot_i, h, w, device=device)

        # ---- flatten 到 CPU，便于 KDTree 检索 ----
        d = feat_map.shape[-1]
        feat_flat = feat_map.view(-1, d).detach().cpu()

        retrieval_dim = retrieval_map.shape[-1]
        retrieval_flat = retrieval_map.reshape(-1, retrieval_dim).detach().cpu()

        pred_flat = pred_i.reshape(-1).detach().cpu()
        type_flat = type_map.reshape(-1).detach().cpu()
        known_flat = known_mask.reshape(-1).detach().cpu()
        residual_flat = residual_map.reshape(-1).detach().cpu()

        out_residual_flat = torch.zeros_like(pred_flat)

        unique_types = torch.unique(type_flat)

        for t in unique_types.tolist():
            # 代码2逻辑：同 mask 类型下所有像素都作为 Query
            q_idx = torch.nonzero(type_flat == t, as_tuple=False).squeeze(1)

            # Key/Value：同 mask 类型下的已知真实深度点
            k_idx = torch.nonzero((type_flat == t) & known_flat, as_tuple=False).squeeze(1)

            if q_idx.numel() == 0 or k_idx.numel() == 0:
                continue

            k_feat = feat_flat[k_idx]  # [Nk, D]
            k_retrieval = retrieval_flat[k_idx].clone()  # [Nk, 3]
            k_pred = pred_flat[k_idx]  # [Nk]
            k_residual = residual_flat[k_idx]  # [Nk]

            # ---- Attention 特征标准化：照搬代码2 ----
            feat_mean = k_feat.mean(dim=0, keepdim=True)
            feat_std = k_feat.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            k_feat_n = (k_feat - feat_mean) / feat_std

            # ---- pred 只用于 pred_tol ----
            pred_std = k_pred.std(unbiased=False).clamp_min(1e-6)

            # ---- KDTree 检索特征标准化：照搬代码2 ----
            retrieval_mean = k_retrieval.mean(dim=0, keepdim=True)
            retrieval_std = k_retrieval.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)

            k_retrieval = (k_retrieval - retrieval_mean) / retrieval_std

            # 让 LS 基线在 KDTree 检索中更重要
            if retrieval_dim >= 3:
                k_retrieval[:, 2] = k_retrieval[:, 2] * 2.0

            # ---- KDTree 建在同 mask 类型的已知点上 ----
            tree = cKDTree(k_retrieval.numpy())

            pre_k = int(min(max(topk * prefilter_factor, topk), k_idx.numel()))
            pred_tol = max(float(pred_std.item()) * pred_tol_scale, 1e-4)

            for start in range(0, q_idx.numel(), chunk_size):
                end = min(start + chunk_size, q_idx.numel())
                q_idx_chunk = q_idx[start:end]

                q_feat = feat_flat[q_idx_chunk]  # [Nq, D]
                q_retrieval = retrieval_flat[q_idx_chunk].clone()  # [Nq, 3]
                q_pred = pred_flat[q_idx_chunk]  # [Nq]

                q_feat_n = (q_feat - feat_mean) / feat_std

                q_retrieval = (q_retrieval - retrieval_mean) / retrieval_std

                if retrieval_dim >= 3:
                    q_retrieval[:, 2] = q_retrieval[:, 2] * 2.0

                # ---- KDTree 预检索 ----
                _, nn_local = tree.query(q_retrieval.numpy(), k=pre_k, workers=-1)
                nn_local = np.asarray(nn_local)

                if nn_local.ndim == 1:
                    nn_local = nn_local[:, None]

                nn_local_t = torch.from_numpy(nn_local).long()

                cand_pred = k_pred[nn_local_t]  # [Nq, pre_k]
                valid = (cand_pred - q_pred.unsqueeze(1)).abs() <= pred_tol

                # 如果 pred_tol 内没有候选，则回退到全部 pre_k
                no_valid = ~valid.any(dim=1)
                if no_valid.any():
                    valid[no_valid] = True

                cand_feat = k_feat_n[nn_local_t]  # [Nq, pre_k, D]
                cand_residual = k_residual[nn_local_t]  # [Nq, pre_k]

                # ---- cross-attention 分数 ----
                scores = (cand_feat * q_feat_n.unsqueeze(1)).sum(dim=-1) / math.sqrt(float(d))
                scores = scores.masked_fill(~valid, -1e9)

                attn = torch.softmax(scores, dim=1)
                residual_chunk = (attn * cand_residual).sum(dim=1)

                out_residual_flat[q_idx_chunk] = residual_chunk

        return out_residual_flat.view(h, w).to(device=device, dtype=dtype)

    def init_scailing(self, pred, sparse, rgb=None, mask_onehot=None, align_points_num=1e10):
        # Global least-squares alignment
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

        if rgb is None:
            return depth

        topk = int(getattr(self.args, 'residual_topk', 8))
        prefilter_factor = int(getattr(self.args, 'residual_prefilter_factor', 4))
        pred_tol_scale = float(getattr(self.args, 'residual_pred_tol_scale', 0.5))
        chunk_size = int(getattr(self.args, 'residual_chunk_size', 4096))

        # Local KDTree-based residual correction
        for i in range(pred.shape[0]):
            pred_i = pred[i, 0]
            ls_i = depth[i, 0]
            target_i = sparse[i, 0]
            rgb_i = rgb[i]

            mask_i = None
            if mask_onehot is not None:
                mask_i = mask_onehot[i]

            residual_corr = self._memory_cross_attention_residual(
                pred_i=pred_i,
                target_i=target_i,
                ls_i=ls_i,
                rgb_i=rgb_i,
                mask_onehot_i=mask_i,
                topk=topk,
                prefilter_factor=prefilter_factor,
                pred_tol_scale=pred_tol_scale,
                chunk_size=chunk_size
            )

            depth[i, 0] = ls_i + residual_corr

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

    def resize_to_multiple(self, x, multiple_of=14, mode='bilinear', resize_lower_size=518):
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


