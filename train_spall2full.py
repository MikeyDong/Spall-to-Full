
import os

import math
import random
import csv
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
from model.ours.spall_to_full_train import Any2Full
# ============================================================
# 1. Path
# ============================================================
DATA_ROOT = Path("data")

TRAIN_ROOT = DATA_ROOT / "Train"
VAL_ROOT = DATA_ROOT / "Val"

TRAIN_RGB_DIR = TRAIN_ROOT / "RGB"
TRAIN_DEPTH_DIR = TRAIN_ROOT / "Depth"
TRAIN_MASK_ONEHOT_DIR = TRAIN_ROOT / "One hot Mask"

VAL_RGB_DIR = VAL_ROOT / "RGB"
VAL_DEPTH_DIR = VAL_ROOT / "Depth"
VAL_MASK_ONEHOT_DIR = VAL_ROOT / "One hot Mask"

CHECKPOINT_PATH = Path("checkpoints") / "Any2Full_vitl.pth.tar"
OUTPUT_DIR = Path("outputs")


# ============================================================
# 2. Hyperparameters
# ============================================================
SEED = 42
VAL_RATIO = 0.20
EXPECTED_SIZE = 1456
TRAIN_RANDOM_CROP_NUM = 4
BATCH_SIZE = 1
NUM_WORKERS = 12
EPOCHS = 30
LR = 5e-5
WEIGHT_DECAY = 0.02
WARMUP_STEPS = 0
MAX_DEPTH_MM = 3000.0
MIN_VALID_MM = 1e-6

CLASS_WEIGHT_BG = 0.1
CLASS_WEIGHT_CONCRETE = 3.0
CLASS_WEIGHT_SPALL = 10.0
# Sparse Depth Pseudo-Degeneration (Applicable Only to Depth Input, Not to Semantic Masks)
MASK_RATIO = 0.60
MASK_BLOCK_RATIO = 0.70
MASK_SCATTER_RATIO = 0.30

# Data Augmentation
BLUR_PROB = 0.5
BLUR_RADIUS_MIN = 0.1
BLUR_RADIUS_MAX = 0.6
LOW_CONTRAST_PROB = 0.50
CONTRAST_FACTOR_MIN = 0.88
CONTRAST_FACTOR_MAX = 0.98
BRIGHTNESS_FACTOR_MIN = 0.90
BRIGHTNESS_FACTOR_MAX = 0.98
ROTATE_DEG = 45.0
HFLIP_PROB = 0.5
VFLIP_PROB = 0.1


FREEZE_BACKBONE_LIKE_OFFICIAL = True
TRAIN_DEPTH_HEAD = False

# Loss weights
LAMBDA_SSI = 5.0
LAMBDA_GM = 100.0
LAMBDA_ANCHOR = 5.0
LAMBDA_RSSIM = 2.0
USE_RSSIM = False
LAMBDA_ABS_MAE = 0.0

LAMBDA_CURV = 1200.0
LAMBDA_RANK = 400.0
LAMBDA_PIT_EDGE = 0.1
CURV_NEIGHBOR_SIZE = 15
# ============================================================
# 3. Utility functions
# ============================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_triplets(rgb_dir: Path, depth_dir: Path, mask_dir: Path) -> List[Tuple[str, Path, Path, Path]]:
    rgb_map = {p.stem: p for p in rgb_dir.glob("*.jpg")}
    rgb_map.update({p.stem: p for p in rgb_dir.glob("*.jpeg")})
    rgb_map.update({p.stem: p for p in rgb_dir.glob("*.png")})
    rgb_map.update({p.stem: p for p in rgb_dir.glob("*.JPG")})
    rgb_map.update({p.stem: p for p in rgb_dir.glob("*.JPEG")})
    rgb_map.update({p.stem: p for p in rgb_dir.glob("*.PNG")})

    depth_map = {p.stem: p for p in depth_dir.glob("*.png")}
    depth_map.update({p.stem: p for p in depth_dir.glob("*.PNG")})

    mask_map = {p.stem: p for p in mask_dir.glob("*.png")}
    mask_map.update({p.stem: p for p in mask_dir.glob("*.PNG")})

    stems = sorted(set(rgb_map.keys()) & set(depth_map.keys()) & set(mask_map.keys()))
    return [(s, rgb_map[s], depth_map[s], mask_map[s]) for s in stems]

def prepare_physical_train_val_split() -> None:
    train_rgb_dir = Path(TRAIN_RGB_DIR)
    train_depth_dir = Path(TRAIN_DEPTH_DIR)
    train_mask_dir = Path(TRAIN_MASK_ONEHOT_DIR)

    val_rgb_dir = Path(VAL_RGB_DIR)
    val_depth_dir = Path(VAL_DEPTH_DIR)
    val_mask_dir = Path(VAL_MASK_ONEHOT_DIR)

    val_rgb_dir.mkdir(parents=True, exist_ok=True)
    val_depth_dir.mkdir(parents=True, exist_ok=True)
    val_mask_dir.mkdir(parents=True, exist_ok=True)

    train_samples = list_triplets(train_rgb_dir, train_depth_dir, train_mask_dir)
    val_samples = list_triplets(val_rgb_dir, val_depth_dir, val_mask_dir)

    total_samples = len(train_samples) + len(val_samples)
    if total_samples == 0:
        raise RuntimeError(
            f"训练/验证总样本为空，请检查路径：\n"
            f"TRAIN_RGB_DIR={TRAIN_RGB_DIR}\n"
            f"TRAIN_DEPTH_DIR={TRAIN_DEPTH_DIR}\n"
            f"TRAIN_MASK_ONEHOT_DIR={TRAIN_MASK_ONEHOT_DIR}\n"
            f"VAL_RGB_DIR={VAL_RGB_DIR}\n"
            f"VAL_DEPTH_DIR={VAL_DEPTH_DIR}\n"
            f"VAL_MASK_ONEHOT_DIR={VAL_MASK_ONEHOT_DIR}"
        )

    expected_val = max(1, int(round(total_samples * VAL_RATIO)))

    # If the number of validation sets is already correct, proceed directly to the next step without re-partitioning.
    if len(val_samples) == expected_val:
        print(f"[Info] 验证集已存在且数量正确，跳过物理划分。Val={len(val_samples)}")
        print(f"[Info] 当前训练集目录: {TRAIN_RGB_DIR}")
        print(f"[Info] 当前验证集目录: {VAL_RGB_DIR}")
        return

    # If files already exist in the val directory but the quantity does not match the expected number, directly raise an error.
    if len(val_samples) != 0:
        raise RuntimeError(
            f"当前验证集目录中已存在 {len(val_samples)} 个样本，但期望为 {expected_val} 个。\n"
            f"请先手动检查或清空以下目录后再运行：\n"
            f"{VAL_RGB_DIR}\n{VAL_DEPTH_DIR}\n{VAL_MASK_ONEHOT_DIR}"
        )

    # From the training set, a physical shear of 20% is applied to the validation set
    indices = list(range(len(train_samples)))
    rng = random.Random(SEED)
    rng.shuffle(indices)

    selected_idx = set(indices[:expected_val])
    selected_val_samples = [train_samples[i] for i in range(len(train_samples)) if i in selected_idx]

    print(f"[Info] 开始物理划分验证集：总样本={total_samples}, 目标验证集={expected_val}")

    for stem, rgb_path, depth_path, mask_path in tqdm(selected_val_samples, desc="Moving val triplets"):
        shutil.move(str(rgb_path), str(val_rgb_dir / rgb_path.name))
        shutil.move(str(depth_path), str(val_depth_dir / depth_path.name))
        shutil.move(str(mask_path), str(val_mask_dir / mask_path.name))

    # Re-examine
    train_samples_after = list_triplets(train_rgb_dir, train_depth_dir, train_mask_dir)
    val_samples_after = list_triplets(val_rgb_dir, val_depth_dir, val_mask_dir)

    print(f"[Info] 物理划分完成。Train={len(train_samples_after)}, Val={len(val_samples_after)}")
    print(f"[Info] 训练集 RGB 目录: {TRAIN_RGB_DIR}")
    print(f"[Info] 训练集 Depth 目录: {TRAIN_DEPTH_DIR}")
    print(f"[Info] 训练集 Mask 目录: {TRAIN_MASK_ONEHOT_DIR}")
    print(f"[Info] 验证集 RGB 目录: {VAL_RGB_DIR}")
    print(f"[Info] 验证集 Depth 目录: {VAL_DEPTH_DIR}")
    print(f"[Info] 验证集 Mask 目录: {VAL_MASK_ONEHOT_DIR}")
    
def stable_int_from_string(s: str) -> int:
    import hashlib
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def pil_depth_from_array(depth_arr: np.ndarray) -> Image.Image:
    return Image.fromarray(depth_arr.astype(np.float32), mode="F")


def _torch_generator_from_seed(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def generate_dynamic_mask(
    h: int,
    w: int,
    seed: Optional[int] = None,
    mask_ratio: float = MASK_RATIO,
    block_ratio: float = MASK_BLOCK_RATIO,
    scatter_ratio: float = MASK_SCATTER_RATIO,
) -> torch.Tensor:
    """
    生成稀疏深度保留 mask：1=保留，0=删除。
    只作用于深度输入口，不影响语义 one-hot mask。
    """
    total = h * w
    drop_total = int(round(mask_ratio * total))
    drop_block = int(round(block_ratio * drop_total))
    drop_scatter = drop_total - drop_block

    gen = None if seed is None else _torch_generator_from_seed(seed)
    keep = torch.ones((h, w), dtype=torch.float32)

    lr_h = max(4, h // 16)
    lr_w = max(4, w // 16)
    noise = torch.rand((1, 1, lr_h, lr_w), generator=gen)
    noise = F.interpolate(noise, size=(h, w), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    noise = F.avg_pool2d(noise.unsqueeze(0).unsqueeze(0), kernel_size=7, stride=1, padding=3).squeeze(0).squeeze(0)

    if drop_block > 0:
        q = 1.0 - (drop_block / float(total))
        q = min(max(q, 0.0), 1.0)
        threshold = torch.quantile(noise.reshape(-1), q)
        block_drop = noise >= threshold
        keep[block_drop] = 0.0

    if drop_scatter > 0:
        remain = torch.nonzero(keep > 0.5, as_tuple=False)
        if remain.shape[0] > 0:
            n = min(drop_scatter, remain.shape[0])
            perm = torch.randperm(remain.shape[0], generator=gen)[:n]
            pts = remain[perm]
            keep[pts[:, 0], pts[:, 1]] = 0.0

    return keep

# ============================================================
# 4. Dataset
# ============================================================
class SpallingDataset(Dataset):
    def __init__(
        self,
        rgb_dir: Optional[str] = None,
        depth_dir: Optional[str] = None,
        mask_onehot_dir: Optional[str] = None,
        is_train: bool = True,
        samples: Optional[List[Tuple[str, Path, Path, Path]]] = None,
    ):
        self.is_train = is_train

        if samples is not None:
            self.rgb_dir = None
            self.depth_dir = None
            self.mask_onehot_dir = None
            self.samples = samples
            if len(self.samples) == 0:
                raise RuntimeError("传入的 samples 为空。")
        else:
            self.rgb_dir = Path(rgb_dir)
            self.depth_dir = Path(depth_dir)
            self.mask_onehot_dir = Path(mask_onehot_dir)
            self.samples = list_triplets(self.rgb_dir, self.depth_dir, self.mask_onehot_dir)
            if len(self.samples) == 0:
                raise RuntimeError(
                    f"数据集为空: rgb_dir={rgb_dir}, depth_dir={depth_dir}, mask_onehot_dir={mask_onehot_dir}"
                )

    def __len__(self):
        if self.is_train:
            return len(self.samples) * TRAIN_RANDOM_CROP_NUM
        return len(self.samples)
        
    def _fixed_val_mask_seed(self, stem: str) -> int:
        return SEED * 100000 + stable_int_from_string(stem)


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_train:
            real_idx = idx // TRAIN_RANDOM_CROP_NUM
        else:
            real_idx = idx
    
        stem, rgb_path, depth_path, mask_path = self.samples[real_idx]
    
        rgb_img = Image.open(rgb_path).convert("RGB")
        depth_arr = np.array(Image.open(depth_path), dtype=np.float32)
        mask_img = Image.open(mask_path)
    

        if self.is_train:
            if random.random() < BLUR_PROB:
                rgb_img = rgb_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(BLUR_RADIUS_MIN, BLUR_RADIUS_MAX)))
    
            if random.random() < LOW_CONTRAST_PROB:
                rgb_img = TF.adjust_contrast(rgb_img, contrast_factor=random.uniform(CONTRAST_FACTOR_MIN, CONTRAST_FACTOR_MAX))
                rgb_img = TF.adjust_brightness(rgb_img, brightness_factor=random.uniform(BRIGHTNESS_FACTOR_MIN, BRIGHTNESS_FACTOR_MAX))
    
            if random.random() < HFLIP_PROB:
                rgb_img = TF.hflip(rgb_img)
                depth_arr = np.fliplr(depth_arr).copy()
                mask_img = TF.hflip(mask_img)
    
            if random.random() < VFLIP_PROB:
                rgb_img = TF.vflip(rgb_img)
                depth_arr = np.flipud(depth_arr).copy()
                mask_img = TF.vflip(mask_img)
    
            angle = random.uniform(-ROTATE_DEG, ROTATE_DEG)
            rgb_img = TF.rotate(rgb_img, angle, interpolation=TF.InterpolationMode.BICUBIC, fill=0)
    
            depth_pil = pil_depth_from_array(depth_arr)
            depth_pil = TF.rotate(depth_pil, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0.0)
            depth_arr = np.array(depth_pil, dtype=np.float32)
    
            mask_img = TF.rotate(mask_img, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)
    

            h, w = depth_arr.shape
            crop_size = EXPECTED_SIZE
    
            if h < crop_size or w < crop_size:
                raise RuntimeError(
                    f"{stem} 的尺寸小于裁剪尺寸 {crop_size}x{crop_size}，当前 depth 尺寸为 {h}x{w}"
                )
    
            top = random.randint(0, h - crop_size)
            left = random.randint(0, w - crop_size)
    
            rgb_img = TF.crop(rgb_img, top, left, crop_size, crop_size)
            depth_arr = depth_arr[top:top + crop_size, left:left + crop_size].copy()
            mask_img = TF.crop(mask_img, top, left, crop_size, crop_size)
    
        else:
 
            rgb_img = TF.resize(
                rgb_img,
                [EXPECTED_SIZE, EXPECTED_SIZE],
                interpolation=TF.InterpolationMode.BICUBIC
            )
    
            depth_pil = pil_depth_from_array(depth_arr)
            depth_pil = TF.resize(
                depth_pil,
                [EXPECTED_SIZE, EXPECTED_SIZE],
                interpolation=TF.InterpolationMode.NEAREST
            )
            depth_arr = np.array(depth_pil, dtype=np.float32)
    
            mask_img = TF.resize(
                mask_img,
                [EXPECTED_SIZE, EXPECTED_SIZE],
                interpolation=TF.InterpolationMode.NEAREST
            )
    

        rgb_tensor = TF.to_tensor(rgb_img)
        rgb_tensor = TF.normalize(
            rgb_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
        gt = torch.from_numpy(depth_arr).float()
        sem_mask = TF.to_tensor(mask_img)
        sem_mask = (sem_mask > 0.5).float()
    

        if rgb_tensor.shape[-2:] != (EXPECTED_SIZE, EXPECTED_SIZE):
            raise RuntimeError(
                f"{stem} 的 RGB 尺寸不是 {EXPECTED_SIZE}x{EXPECTED_SIZE}，而是 {tuple(rgb_tensor.shape[-2:])}"
            )
        if gt.shape != (EXPECTED_SIZE, EXPECTED_SIZE):
            raise RuntimeError(
                f"{stem} 的 Depth 尺寸不是 {EXPECTED_SIZE}x{EXPECTED_SIZE}，而是 {tuple(gt.shape)}"
            )
        if sem_mask.shape[-2:] != (EXPECTED_SIZE, EXPECTED_SIZE):
            raise RuntimeError(
                f"{stem} 的 Mask 尺寸不是 {EXPECTED_SIZE}x{EXPECTED_SIZE}，而是 {tuple(sem_mask.shape[-2:])}"
            )
    
   
        if self.is_train:
            keep_mask = generate_dynamic_mask(gt.shape[0], gt.shape[1], seed=None)
        else:
            keep_mask = generate_dynamic_mask(
                gt.shape[0],
                gt.shape[1],
                seed=self._fixed_val_mask_seed(stem)
            )
    
        sparse_depth = gt * keep_mask
    
        return {
            "stem": stem,
            "rgb": rgb_tensor,                    # [3,H,W]
            "dep": sparse_depth.unsqueeze(0),     # [1,H,W]
            "gt": gt.unsqueeze(0),                # [1,H,W]
            "keep_mask": keep_mask.unsqueeze(0),  # [1,H,W]
            "sem_mask": sem_mask,                # [3,H,W]
        }

# ============================================================
# 5. Loss
# ============================================================
def depth_to_relative_disparity(depth_mm: torch.Tensor) -> torch.Tensor:
    valid = depth_mm > MIN_VALID_MM
    rel = torch.zeros_like(depth_mm)
    rel[valid] = 1.0 / (depth_mm[valid] + 1e-8)
    return rel


def normalize_relative_per_sample(rel: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(rel)
    bsz = rel.shape[0]
    for b in range(bsz):
        vals = rel[b][valid[b]]
        if vals.numel() >= 2:
            mu = vals.mean()
            sigma = vals.std().clamp_min(1e-6)
            out[b] = (rel[b] - mu) / sigma
        else:
            out[b] = rel[b]
    return out

def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:

    if values.numel() == 0:
        return values.new_tensor(0.0)
    weights = weights.to(values.dtype)
    denom = weights.sum().clamp_min(1e-6)
    return (values * weights).sum() / denom


def build_class_weight_map(sem_mask: torch.Tensor) -> torch.Tensor:

    bg = sem_mask[:, 0:1, :, :]
    concrete = sem_mask[:, 1:2, :, :]
    spall = sem_mask[:, 2:3, :, :]

    class_weight_map = (
        CLASS_WEIGHT_BG * bg
        + CLASS_WEIGHT_CONCRETE * concrete
        + CLASS_WEIGHT_SPALL * spall
    )


    invalid_onehot = (sem_mask.sum(dim=1, keepdim=True) < 0.5)
    class_weight_map = torch.where(
        invalid_onehot,
        torch.full_like(class_weight_map, CLASS_WEIGHT_BG),
        class_weight_map
    )

    return class_weight_map

def gradient_matching_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    class_weight_map: torch.Tensor
) -> torch.Tensor:
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_g = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    vx = valid[:, :, :, 1:] & valid[:, :, :, :-1]

    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_g = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    vy = valid[:, :, 1:, :] & valid[:, :, :-1, :]


    wx = 0.5 * (class_weight_map[:, :, :, 1:] + class_weight_map[:, :, :, :-1])
    wy = 0.5 * (class_weight_map[:, :, 1:, :] + class_weight_map[:, :, :-1, :])

    loss_x = weighted_mean((dx_p - dx_g).abs()[vx], wx[vx]) if vx.any() else pred.new_tensor(0.0)
    loss_y = weighted_mean((dy_p - dy_g).abs()[vy], wy[vy]) if vy.any() else pred.new_tensor(0.0)
    return loss_x + loss_y


def masked_mae_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    class_weight_map: torch.Tensor
) -> torch.Tensor:
    valid = valid.bool()
    return weighted_mean((pred - gt).abs()[valid], class_weight_map[valid]) if valid.any() else pred.new_tensor(0.0)

def gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)
    return w.repeat(channels, 1, 1, 1)

def masked_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    valid: torch.Tensor,
    class_weight_map: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5
) -> torch.Tensor:
    channels = x.shape[1]
    window = gaussian_window(window_size, sigma, channels, x.device, x.dtype)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=channels)
    sigma_x = F.conv2d(x * x, window, padding=window_size // 2, groups=channels) - mu_x * mu_x
    sigma_y = F.conv2d(y * y, window, padding=window_size // 2, groups=channels) - mu_y * mu_y
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=channels) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )

    valid = valid.bool()
    return weighted_mean((1.0 - ssim_map)[valid], class_weight_map[valid]) if valid.any() else x.new_tensor(0.0)



def smooth_l1_weighted_mean(diff: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute a class-weighted Smooth L1 loss."""
    if diff.numel() == 0:
        return diff.new_tensor(0.0)

    values = F.smooth_l1_loss(
        diff,
        torch.zeros_like(diff),
        reduction="none"
    )
    return weighted_mean(values, weights)


def laplacian_map(x: torch.Tensor) -> torch.Tensor:
    """Approximate local depth curvature using a four-neighbor discrete Laplacian."""
    kernel = x.new_tensor([
        [0.0,  1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0,  1.0, 0.0],
    ]).view(1, 1, 3, 3)

    return F.conv2d(x, kernel, padding=1)


def laplacian_valid_mask(valid: torch.Tensor) -> torch.Tensor:
    """Retain pixels whose center and four neighboring pixels all contain valid depth."""
    kernel = valid.float().new_tensor([
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
    ]).view(1, 1, 3, 3)

    count = F.conv2d(valid.float(), kernel, padding=1)
    return count >= 5.0


def local_mean_with_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
    kernel_size: int = CURV_NEIGHBOR_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute a local mean using only valid pixels within the specified mask."""
    mask_f = mask.float()
    pad = kernel_size // 2
    area = float(kernel_size * kernel_size)

    x_sum = F.avg_pool2d(x * mask_f, kernel_size=kernel_size, stride=1, padding=pad) * area
    m_sum = F.avg_pool2d(mask_f, kernel_size=kernel_size, stride=1, padding=pad) * area

    local_mean = x_sum / m_sum.clamp_min(1e-6)
    local_valid = m_sum > 0.5
    return local_mean, local_valid


def curvature_laplacian_loss(
    pred_rel: torch.Tensor,
    gt_rel: torch.Tensor,
    valid_gt: torch.Tensor,
    sem_mask: torch.Tensor,
    class_weight_map: torch.Tensor
) -> torch.Tensor:
    """
    Match the local Laplacian curvature of predicted and ground-truth depth
    within intact and spalled concrete regions.
    """
    foreground = (sem_mask[:, 1:2, :, :] + sem_mask[:, 2:3, :, :]) > 0.5

    valid_lap = laplacian_valid_mask(valid_gt) & foreground

    k_pred = laplacian_map(pred_rel)
    k_gt = laplacian_map(gt_rel)

    if not valid_lap.any():
        return pred_rel.new_tensor(0.0)

    diff = k_pred - k_gt


    return smooth_l1_weighted_mean(diff[valid_lap], class_weight_map[valid_lap])

 


def local_interpolate_sparse_depth(
    sparse_mm: torch.Tensor,
    sparse_valid: torch.Tensor,
    kernel_size: int = CURV_NEIGHBOR_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct a coarse local depth reference from valid sparse-depth neighbors.
    The coarse result is used only as the reference for the ranking loss.
    """
    coarse_mm, coarse_valid = local_mean_with_mask(
        sparse_mm,
        sparse_valid,
        kernel_size=kernel_size
    )


    coarse_mm = torch.where(sparse_valid, sparse_mm, coarse_mm)
    coarse_valid = coarse_valid | sparse_valid

    return coarse_mm, coarse_valid


def curvature_rank_loss(
    pred_rel: torch.Tensor,
    gt_rel: torch.Tensor,
    coarse_rel: torch.Tensor,
    valid_gt: torch.Tensor,
    coarse_valid: torch.Tensor,
    sem_mask: torch.Tensor,
    class_weight_map: torch.Tensor
) -> torch.Tensor:
    """
    Penalize the prediction only when its local curvature error exceeds
    that of the coarse sparse-depth interpolation.
    """
    foreground = (sem_mask[:, 1:2, :, :] + sem_mask[:, 2:3, :, :]) > 0.5

    valid_lap = (
        laplacian_valid_mask(valid_gt)
        & laplacian_valid_mask(coarse_valid)
        & foreground
    )

    if not valid_lap.any():
        return pred_rel.new_tensor(0.0)

    k_pred = laplacian_map(pred_rel)
    k_gt = laplacian_map(gt_rel)
    k_coarse = laplacian_map(coarse_rel)

    e_pred = (k_pred - k_gt).abs()
    e_coarse = (k_coarse - k_gt).abs()


    rank_penalty = F.relu(e_pred - e_coarse)

    return weighted_mean(rank_penalty[valid_lap], class_weight_map[valid_lap])


def spall_edge_pit_loss(
    pred_mm: torch.Tensor,
    gt_mm: torch.Tensor,
    valid_gt: torch.Tensor,
    sem_mask: torch.Tensor,
    class_weight_map: torch.Tensor,
    kernel_size: int = CURV_NEIGHBOR_SIZE
) -> torch.Tensor:
    """
    Constrain the magnitude and direction of the predicted spall-edge
    depression relative to neighboring intact concrete.
    """
    concrete = (sem_mask[:, 1:2, :, :] > 0.5) & valid_gt
    spall = (sem_mask[:, 2:3, :, :] > 0.5) & valid_gt

    ref_pred, has_concrete_neighbor = local_mean_with_mask(
        pred_mm,
        concrete,
        kernel_size=kernel_size
    )
    ref_gt, _ = local_mean_with_mask(
        gt_mm,
        concrete,
        kernel_size=kernel_size
    )


    edge_valid = spall & has_concrete_neighbor

    if not edge_valid.any():
        return pred_mm.new_tensor(0.0)

    r_pred = pred_mm - ref_pred
    r_gt = gt_mm - ref_gt

    r_pred_v = r_pred[edge_valid]
    r_gt_v = r_gt[edge_valid]
    w_v = class_weight_map[edge_valid]

 
    mag_loss = F.smooth_l1_loss(r_pred_v, r_gt_v, reduction="none")


    sign_gt = torch.sign(r_gt_v)
    dir_loss = F.relu(-r_pred_v * sign_gt)

 
    reliable_dir = r_gt_v.abs() > MIN_VALID_MM
    dir_loss = torch.where(reliable_dir, dir_loss, torch.zeros_like(dir_loss))

    return weighted_mean(mag_loss + dir_loss, w_v)
    
class Any2FullLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_mm: torch.Tensor,
        gt_mm: torch.Tensor,
        sparse_mm: torch.Tensor,
        prompt_valid: torch.Tensor,
        sem_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        valid_gt = (gt_mm > MIN_VALID_MM) & (gt_mm <= MAX_DEPTH_MM) & torch.isfinite(gt_mm)
        prompt_valid = prompt_valid.bool() & valid_gt


        class_weight_map = build_class_weight_map(sem_mask)

        if valid_gt.any():
            pred_rel = normalize_relative_per_sample(depth_to_relative_disparity(pred_mm), valid_gt)
            gt_rel = normalize_relative_per_sample(depth_to_relative_disparity(gt_mm), valid_gt)

            diff_rel = (pred_rel - gt_rel).abs()

  
            loss_ssi = weighted_mean(diff_rel[valid_gt], class_weight_map[valid_gt])

            loss_grad = gradient_matching_loss(pred_rel, gt_rel, valid_gt, class_weight_map)

            loss_anchor = (
                weighted_mean(diff_rel[prompt_valid], class_weight_map[prompt_valid])
                if prompt_valid.any()
                else pred_mm.new_tensor(0.0)
            )


            loss_rssim = masked_ssim(pred_rel, gt_rel, valid_gt, class_weight_map) if USE_RSSIM else pred_mm.new_tensor(0.0)

            loss_abs_mae = masked_mae_loss(pred_mm, gt_mm, valid_gt, class_weight_map)

            loss_curv = curvature_laplacian_loss(
                pred_rel=pred_rel,
                gt_rel=gt_rel,
                valid_gt=valid_gt,
                sem_mask=sem_mask,
                class_weight_map=class_weight_map
            )

  
            coarse_mm, coarse_valid = local_interpolate_sparse_depth(
                sparse_mm=sparse_mm,
                sparse_valid=prompt_valid,
                kernel_size=CURV_NEIGHBOR_SIZE
            )

            coarse_valid = coarse_valid & valid_gt
            coarse_rel = normalize_relative_per_sample(
                depth_to_relative_disparity(coarse_mm),
                coarse_valid
            )

            loss_rank = curvature_rank_loss(
                pred_rel=pred_rel,
                gt_rel=gt_rel,
                coarse_rel=coarse_rel,
                valid_gt=valid_gt,
                coarse_valid=coarse_valid,
                sem_mask=sem_mask,
                class_weight_map=class_weight_map
            )


            loss_pit_edge = spall_edge_pit_loss(
                pred_mm=pred_mm,
                gt_mm=gt_mm,
                valid_gt=valid_gt,
                sem_mask=sem_mask,
                class_weight_map=class_weight_map,
                kernel_size=CURV_NEIGHBOR_SIZE
            )

        else:
            z = pred_mm.sum() * 0.0
            z_detach = pred_mm.new_tensor(0.0)
            return {
                "total": z,
                "ssi": z_detach,
                "grad": z_detach,
                "anchor": z_detach,
                "rssim": z_detach,
                "abs_mae": z_detach,
                "curv": z_detach,
                "rank": z_detach,
                "pit_edge": z_detach,
            }

        total = (
            LAMBDA_SSI * loss_ssi
            + LAMBDA_GM * loss_grad
            + LAMBDA_ANCHOR * loss_anchor
            + LAMBDA_RSSIM * loss_rssim
            + LAMBDA_ABS_MAE * loss_abs_mae
            + LAMBDA_CURV * loss_curv
            + LAMBDA_RANK * loss_rank
            + LAMBDA_PIT_EDGE * loss_pit_edge
        )

        return {
            "total": total,
            "ssi": (LAMBDA_SSI * loss_ssi).detach(),
            "grad": (LAMBDA_GM * loss_grad).detach(),
            "anchor": (LAMBDA_ANCHOR * loss_anchor).detach(),
            "rssim": (LAMBDA_RSSIM * loss_rssim).detach(),
            "abs_mae": (LAMBDA_ABS_MAE * loss_abs_mae).detach(),
            "curv": (LAMBDA_CURV * loss_curv).detach(),
            "rank": (LAMBDA_RANK * loss_rank).detach(),
            "pit_edge": (LAMBDA_PIT_EDGE * loss_pit_edge).detach(),
        }
# ============================================================
# 6. Model Construction and Freezing
# ============================================================
def build_model(device: torch.device) -> nn.Module:
    args = type("Args", (), {
        "stage": 1,
        "init_scailing": True,
        "max_depth": 2e3,
        "min_depth": 1e-6,
        "model_input_size": EXPECTED_SIZE,
    })()

    model = Any2Full(encoder="vitl", da_ckpt_path=None, args=args)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    cleaned = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
    msg = model.load_state_dict(cleaned, strict=False)
    print(f"[Info] Pre-trained weights loading completed,missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    if FREEZE_BACKBONE_LIKE_OFFICIAL:
        for name, param in model.named_parameters():
            trainable = ("prompt_depth" in name) or ("prompt_mask" in name) or ("three_region_routing" in name)
            if TRAIN_DEPTH_HEAD and ("depth_head" in name):
                trainable = True
            param.requires_grad = trainable
    else:
        for param in model.parameters():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[Info] Trainable params: {trainable_params:,}")
    print(f"[Info] Frozen params:    {frozen_params:,}")
    return model


def set_dynamic_depth_bounds(model: nn.Module, dep: torch.Tensor) -> None:
    target_model = model.module if hasattr(model, "module") else model
    valid = torch.isfinite(dep) & (dep > MIN_VALID_MM)
    if valid.any():
        dep_valid = dep[valid]
        cur_min = float(dep_valid.min().item())
        cur_max = float(dep_valid.max().item())
        cur_min = max(cur_min, 1e-6)
        cur_max = max(cur_max, cur_min + 1e-6)
        target_model.args.min_depth = cur_min
        target_model.args.max_depth = cur_max
    else:
        target_model.args.min_depth = 1e-6
        target_model.args.max_depth = 1e3

# ============================================================
# 7. Train and valid
# ============================================================
def make_scheduler(optimizer: optim.Optimizer, total_steps: int):
    def lr_lambda(step: int):
        if step < WARMUP_STEPS:
            return float(step + 1) / float(max(1, WARMUP_STEPS))
        progress = (step - WARMUP_STEPS) / float(max(1, total_steps - WARMUP_STEPS))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def meter_to_str(m: Dict[str, float]) -> str:
    return (
        f"total={m['total']:.4f}, ssi={m['ssi']:.4f}, grad={m['grad']:.4f}, "
        f"anchor={m['anchor']:.4f}, rssim={m['rssim']:.4f}, abs_mae={m['abs_mae']:.4f}, "
        f"curv={m['curv']:.4f}, rank={m['rank']:.4f}, pit_edge={m['pit_edge']:.4f}"
    )


def run_one_epoch(model, loader, criterion, optimizer, scheduler, device, train: bool) -> Dict[str, float]:
    model.train() if train else model.eval()

    sums = {
    "total": 0.0,
    "ssi": 0.0,
    "grad": 0.0,
    "anchor": 0.0,
    "rssim": 0.0,
    "abs_mae": 0.0,
    "curv": 0.0,
    "rank": 0.0,
    "pit_edge": 0.0,
    }

    n_batches = 0
    pbar = tqdm(loader, desc="Train" if train else "Val")

    for batch in pbar:
        rgb = batch["rgb"].to(device, non_blocking=True)           # [B,3,H,W]
        dep = batch["dep"].to(device, non_blocking=True)           # [B,1,H,W]
        gt = batch["gt"].to(device, non_blocking=True)             # [B,1,H,W]
        keep_mask = batch["keep_mask"].to(device, non_blocking=True)   # [B,1,H,W]
        sem_mask = batch["sem_mask"].to(device, non_blocking=True)     # [B,3,H,W]

        batch_sums = {
        "total": 0.0,
        "ssi": 0.0,
        "grad": 0.0,
        "anchor": 0.0,
        "rssim": 0.0,
        "abs_mae": 0.0,
        "curv": 0.0,
        "rank": 0.0,
        "pit_edge": 0.0,
        }

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            set_dynamic_depth_bounds(model, dep)
            output = model({"rgb": rgb, "dep": dep, "mask": sem_mask})
            pred = output["pred"]

            losses = criterion(pred, gt, dep, keep_mask > 0.5, sem_mask)
            loss = losses["total"]

            if train:
                loss.backward()
                optimizer.step()
                scheduler.step()

        for k in batch_sums.keys():
            batch_sums[k] += float(losses[k].item())

        for k in sums.keys():
            sums[k] += batch_sums[k]

        n_batches += 1

        avg = {k: v / max(1, n_batches) for k, v in sums.items()}
        pbar.set_postfix({
        "loss": f"{avg['total']:.4f}",
        "ssi": f"{avg['ssi']:.4f}",
        "grad": f"{avg['grad']:.4f}",
        "anchor": f"{avg['anchor']:.4f}",
        "curv": f"{avg['curv']:.4f}",
        "rank": f"{avg['rank']:.4f}",
        "pit": f"{avg['pit_edge']:.4f}",
        })

    return {k: v / max(1, n_batches) for k, v in sums.items()}

# ============================================================
# 8. Main Function
# ============================================================
def main() -> None:
    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device={device}")

  
    prepare_physical_train_val_split()

    train_dataset = SpallingDataset(
        rgb_dir=TRAIN_RGB_DIR,
        depth_dir=TRAIN_DEPTH_DIR,
        mask_onehot_dir=TRAIN_MASK_ONEHOT_DIR,
        is_train=True,
    )
    val_dataset = SpallingDataset(
        rgb_dir=VAL_RGB_DIR,
        depth_dir=VAL_DEPTH_DIR,
        mask_onehot_dir=VAL_MASK_ONEHOT_DIR,
        is_train=False,
    )

    print(f"[Info] Train images={len(train_dataset.samples)}, Val images={len(val_dataset.samples)}")
    print(f"[Info] Train_Path:")
    print(f"        RGB   : {TRAIN_RGB_DIR}")
    print(f"        Depth : {TRAIN_DEPTH_DIR}")
    print(f"        Mask  : {TRAIN_MASK_ONEHOT_DIR}")
    print(f"[Info] Valid_Path:")
    print(f"        RGB   : {VAL_RGB_DIR}")
    print(f"        Depth : {VAL_DEPTH_DIR}")
    print(f"        Mask  : {VAL_MASK_ONEHOT_DIR}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(device).to(device)
    criterion = Any2FullLoss()

    base_params = []
    new_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if ("prompt_mask_embedding" in name) or ("three_region_routing" in name):
            new_params.append(param)
        else:
            base_params.append(param)

    optimizer = optim.Adam(
        [
            {"params": base_params, "lr": LR},
            {"params": new_params, "lr": LR * 5.0},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    print(f"[Info] base_params groups = {len(base_params)}")
    print(f"[Info] new_params groups  = {len(new_params)}")
    print(f"[Info] base lr = {LR:.6e}, new lr = {LR * 5.0:.6e}")

    total_steps = EPOCHS * max(1, len(train_loader))
    scheduler = make_scheduler(optimizer, total_steps)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "train_val_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
        "epoch", "lr",
        "train_total", "train_ssi", "train_grad", "train_anchor", "train_rssim", "train_abs_mae",
        "train_curv", "train_rank", "train_pit_edge",
        "val_total", "val_ssi", "val_grad", "val_anchor", "val_rssim", "val_abs_mae",
        "val_curv", "val_rank", "val_pit_edge",
        ])

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{EPOCHS}] lr={optimizer.param_groups[0]['lr']:.6e}")
        train_meter = run_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, train=True)
        val_meter = run_one_epoch(model, val_loader, criterion, optimizer, scheduler, device, train=False)

        print(f"[Train] {meter_to_str(train_meter)}")
        print(f"[Val]   {meter_to_str(val_meter)}")

        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
            epoch,
            optimizer.param_groups[0]["lr"],
            train_meter["total"], train_meter["ssi"], train_meter["grad"], train_meter["anchor"], train_meter["rssim"], train_meter["abs_mae"],
            train_meter["curv"], train_meter["rank"], train_meter["pit_edge"],
            val_meter["total"], val_meter["ssi"], val_meter["grad"], val_meter["anchor"], val_meter["rssim"], val_meter["abs_mae"],
            val_meter["curv"], val_meter["rank"], val_meter["pit_edge"],
            ])

        last_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val": best_val,
            "train_meter": train_meter,
            "val_meter": val_meter,
        }
        torch.save(last_payload, output_dir / "last_model.pth")
        if epoch == 15:
            torch.save(last_payload, output_dir / "epoch_15_model.pth")
            print(" The 15th round model has been additionally saved：epoch_15_model.pth")
        if val_meter["total"] < best_val:
            best_val = val_meter["total"]
            best_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val": best_val,
                "train_meter": train_meter,
                "val_meter": val_meter,
            }
            torch.save(best_payload, output_dir / "best_model.pth")
            print(f"Best model has been updated，best val total = {best_val:.6f}")


if __name__ == "__main__":
    main()
