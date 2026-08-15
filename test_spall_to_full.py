"""
Evaluation script for Spall-to-Full.

This script evaluates the final Spall-to-Full model on RGB-D test data
under different sparse-depth conditions.

Inputs:
    - RGB image
    - Sparse depth map
    - Three-class semantic mask:
        0 = background
        1 = intact concrete
        2 = spalled concrete

The evaluation ROI includes all concrete pixels (mask > 0), i.e.,
both intact and spalled concrete regions.

The script reports depth-estimation errors and saves predicted depth maps
and evaluation results.
"""

import argparse
import re
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T

from model.ours.spall_to_full_inference import Any2Full

# =========================
# 1. Command-line arguments
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the final Spall-to-Full model."
    )

    parser.add_argument("--rgb_dir", type=str, required=True)
    parser.add_argument("--depth_input_dirs", nargs="+", required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--gt_depth_dir", type=str, required=True)
    parser.add_argument("--roi_mask_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_root", type=str, default="outputs/evaluation")

    parser.add_argument("--encoder", type=str, default="vitl",
                        choices=["vits", "vitb", "vitl"])
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--da_ckpt_path", type=str, default=None)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--min_depth", type=float, default=1e-6)

    parser.add_argument("--residual_topk", type=int, default=8)
    parser.add_argument("--residual_prefilter_factor", type=int, default=4)
    parser.add_argument("--residual_pred_tol_scale", type=float, default=0.5)
    parser.add_argument("--residual_chunk_size", type=int, default=4096)

    parser.add_argument("--save_npy", action="store_true")
    parser.add_argument("--recursive_rgb", action="store_true")
    parser.add_argument("--recursive_mask", action="store_true")
    parser.add_argument("--recursive_roi", action="store_true")

    parser.add_argument("--model_input_size", type=int, default=1456)

    return parser.parse_args()
# =========================
# 2. Method definition
# =========================
METHODS = OrderedDict([
    ("spall_to_full", {
        "folder": "Spall_to_Full",
        "model": "Spall-to-Full",
        "branch": "local",
        "desc": "Spall-to-Full with semantic prompting, region routing, and local residual correction.",
    }),
])
# =========================
# 3. File and data utilities
# =========================
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder: str, recursive: bool = False):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Directory does not exist: {folder}")

    files = []
    pattern = "**/*" if recursive else "*"
    for p in folder.glob(pattern):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(str(p))
    return sorted(files)


def normalize_stem(stem: str) -> str:
    """Normalize file stems exported by MATLAB Image Labeler."""
    s = stem
    s = re.sub(r"^Label_\d+_", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^label_\d+_", "", s, flags=re.IGNORECASE)
    return s


def build_file_map(folder: str, recursive: bool = False):
    """Build file indices using both original and normalized stems."""
    files = list_images(folder, recursive=recursive)
    m = {}
    for p in files:
        stem = Path(p).stem
        m.setdefault(stem, p)
        m.setdefault(normalize_stem(stem), p)
    return m


def find_by_stem(file_map: dict, key: str):
    """Find a paired file by exact or normalized stem matching."""
    if key in file_map:
        return file_map[key]

    key_norm = normalize_stem(key)
    if key_norm in file_map:
        return file_map[key_norm]

    candidates = []
    for k, v in file_map.items():
        if k == key or normalize_stem(k) == key_norm or key_norm in normalize_stem(k):
            candidates.append(v)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        candidates = sorted(candidates, key=lambda x: len(Path(x).stem))
        return candidates[0]

    return None


def read_depth_array(path: str, depth_scale: float = 1.0) -> np.ndarray:
    if path.lower().endswith(".npy"):
        arr = np.load(path).astype(np.float32)
    else:
        arr = np.array(Image.open(path)).astype(np.float32)
        arr = arr / float(depth_scale)

    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3):
            arr = arr[0]
        else:
            arr = arr[:, :, 0]

    return arr.astype(np.float32)


def read_label_mask(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def resize_np(arr: np.ndarray, size_hw, is_mask_or_depth: bool):
    """Resize RGB with Lanczos and mask/depth data with nearest-neighbor interpolation."""
    target_h, target_w = size_hw
    if arr.shape[:2] == (target_h, target_w):
        return arr

    pil = Image.fromarray(arr)
    resample = Image.Resampling.NEAREST if is_mask_or_depth else Image.Resampling.LANCZOS
    out = pil.resize((target_w, target_h), resample=resample)
    return np.array(out)


def load_rgb_tensor(path: str):
    img = Image.open(path).convert("RGB")
    t_rgb = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return t_rgb(img).unsqueeze(0), img.size[::-1]  # tensor, (H, W)


def load_depth_tensor_aligned(path: str, size_hw, depth_scale: float):
    arr = read_depth_array(path, depth_scale=depth_scale)
    arr = resize_np(arr, size_hw, is_mask_or_depth=True).astype(np.float32)
    return T.ToTensor()(Image.fromarray(arr)).unsqueeze(0), arr


def load_mask_onehot_aligned(path: str, size_hw):
    mask_arr = read_label_mask(path)
    mask_arr = resize_np(mask_arr, size_hw, is_mask_or_depth=True)

    bg = (mask_arr == 0).astype(np.float32)
    concrete = (mask_arr == 1).astype(np.float32)
    spall = (mask_arr == 2).astype(np.float32)
    onehot = np.stack([bg, concrete, spall], axis=0)

    return torch.from_numpy(onehot).unsqueeze(0), mask_arr


def valid_depth_mask(gt: np.ndarray):
    return np.isfinite(gt) & (gt > 0)


def roi_bool_mask(roi_path: str, size_hw):
    """Use all non-background concrete pixels as the evaluation ROI."""
    roi = read_label_mask(roi_path)
    roi = resize_np(roi, size_hw, is_mask_or_depth=True)
    return np.isfinite(roi.astype(np.float32)) & (roi > 0)


def get_input_known_max(depth_arr: np.ndarray, fallback: float = 1487.0):
    m = np.isfinite(depth_arr) & (depth_arr > 0)
    if np.any(m):
        return float(np.nanmax(depth_arr[m]))
    return float(fallback)


def sanitize_filename(s: str):
    s = str(s)
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = re.sub(r"\s+", "_", s)
    return s


# =========================
# 4. Model loading and inference
# =========================
def load_checkpoint(model, ckpt_path: str, device: str):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    cleaned = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
    missing, unexpected = model.load_state_dict(cleaned, strict=True)

    print(f"[Info] Checkpoint loaded: {ckpt_path}")
    print(f"[Info] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("[Warning] First 20 missing keys:", missing[:20])
    if unexpected:
        print("[Warning] First 20 unexpected keys:", unexpected[:20])
    return model


def build_model_args(cli_args, max_depth: float):
    """Build model arguments required for Spall-to-Full evaluation."""
    return SimpleNamespace(
        encoder=cli_args.encoder,
        da_ckpt_path=cli_args.da_ckpt_path,
        init_scailing=True,
        stage=cli_args.stage,
        max_depth=float(max_depth),
        min_depth=float(cli_args.min_depth),
        residual_topk=int(cli_args.residual_topk),
        residual_prefilter_factor=int(cli_args.residual_prefilter_factor),
        residual_pred_tol_scale=float(cli_args.residual_pred_tol_scale),
        residual_chunk_size=int(cli_args.residual_chunk_size),
        model_input_size=int(cli_args.model_input_size),
    )

def create_spall_to_full_model(cli_args, device: str, max_depth: float):
    model_args = build_model_args(cli_args, max_depth)

    model = Any2Full(
        encoder=cli_args.encoder,
        da_ckpt_path=cli_args.da_ckpt_path,
        args=model_args,
    )

    model = load_checkpoint(model, cli_args.checkpoint, device)
    return model.to(device).eval()
@torch.no_grad()
def predict_spall_to_full_once(
    model, rgb_t, dep_t, mask_t, device, max_depth: float
):
    """Run inference with the final Spall-to-Full model."""
    model.args.max_depth = float(max_depth)

    sample = {
        "rgb": rgb_t.to(device),
        "dep": dep_t.to(device),
        "mask": mask_t.to(device),
    }

    out = model(sample)
    pred = (
        out["pred"]
        .squeeze(0)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    return pred



def drop_unwanted_excel_fields(metrics: dict):
    """Remove fields intentionally omitted from the output tables."""
    return {k: v for k, v in metrics.items() if k not in {"ValidPixels"}}


# =========================
# 5. Evaluation metrics
# =========================
def compute_metrics(pred: np.ndarray, gt: np.ndarray, eval_mask: np.ndarray):
    """
    Compute depth-estimation metrics within the specified evaluation region.

    All depth errors are reported in millimetres. In addition to MAE and RMSE,
    the function reports MAPE, MSE, percentile-based errors, and statistics
    for the highest-error 25% of valid pixels.
    """
    valid = eval_mask & valid_depth_mask(gt) & np.isfinite(pred)

    n = int(valid.sum())
    if n == 0:
        return {
            "ValidPixels": 0,
            "MAE_mm": np.nan,
            "MAPE_percent": np.nan,
            "RMSE_mm": np.nan,
            "MSE_mm2": np.nan,
            "MaxAbsError_mm": np.nan,
            "MinAbsError_mm": np.nan,
            "Error95Percentile_mm": np.nan,
            "Top25ErrorThreshold_mm": np.nan,
            "Top25Pixels": 0,
            "Top25PixelRatio": np.nan,
            "Top25_MAE_mm": np.nan,
            "Top25_MAPE_percent": np.nan,
            "Top25_RMSE_mm": np.nan,
            "Top25_MSE_mm2": np.nan,
        }

    err = pred[valid] - gt[valid]
    abs_err = np.abs(err)
    sq_err = err ** 2

    gt_valid = gt[valid]
    denom = np.maximum(np.abs(gt_valid), 1e-6)

    q95 = float(np.percentile(abs_err, 95))
    top25_thr = float(np.percentile(abs_err, 75))
    top25 = abs_err >= top25_thr
    top25_n = int(top25.sum())

    return {
        "ValidPixels": n,
        "MAE_mm": float(np.mean(abs_err)),
        "MAPE_percent": float(np.mean(abs_err / denom) * 100.0),
        "RMSE_mm": float(np.sqrt(np.mean(sq_err))),
        "MSE_mm2": float(np.mean(sq_err)),
        "MaxAbsError_mm": float(np.max(abs_err)),
        "MinAbsError_mm": float(np.min(abs_err)),
        "Error95Percentile_mm": q95,
        "Top25ErrorThreshold_mm": top25_thr,
        "Top25Pixels": top25_n,
        "Top25PixelRatio": float(top25_n / n),
        "Top25_MAE_mm": float(np.mean(abs_err[top25])) if top25_n > 0 else np.nan,
        "Top25_MAPE_percent": float(np.mean(abs_err[top25] / denom[top25]) * 100.0) if top25_n > 0 else np.nan,
        "Top25_RMSE_mm": float(np.sqrt(np.mean(sq_err[top25]))) if top25_n > 0 else np.nan,
        "Top25_MSE_mm2": float(np.mean(sq_err[top25])) if top25_n > 0 else np.nan,
    }


def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return np.nan
    return float(s.mean())


def make_summary(df: pd.DataFrame):
    rows = []

    method_values = sorted(df["Method"].dropna().unique()) if "Method" in df.columns else [None]
    model_values = sorted(df["Model"].dropna().unique()) if "Model" in df.columns else [None]

    for depth_case in sorted(df["DepthCase"].dropna().unique()):
        part_case = df[df["DepthCase"] == depth_case]

        for method_name in method_values:
            part_method = part_case if method_name is None else part_case[part_case["Method"] == method_name]

            for model_name in model_values:
                part = part_method if model_name is None else part_method[part_method["Model"] == model_name]

                for region in ["ROI", "FullImage"]:
                    sub = part[part["Region"] == region]
                    if len(sub) == 0:
                        continue

                    row = {
                        "DepthCase": depth_case,
                        "Method": method_name if method_name is not None else "",
                        "Model": model_name if model_name is not None else "",
                        "Branch": str(sub["Branch"].iloc[0]) if "Branch" in sub.columns else "",
                        "Region": region,
                        "NumImages": int(sub["Image"].nunique()),
                        "Mean_MAE_mm": safe_mean(sub["MAE_mm"]),
                        "Mean_MAPE_percent": safe_mean(sub["MAPE_percent"]),
                        "Mean_RMSE_mm": safe_mean(sub["RMSE_mm"]),
                        "Mean_MSE_mm2": safe_mean(sub["MSE_mm2"]),
                        "Mean_MaxAbsError_mm": safe_mean(sub["MaxAbsError_mm"]),
                        "Mean_MinAbsError_mm": safe_mean(sub["MinAbsError_mm"]),
                        "Mean_Error95Percentile_mm": safe_mean(sub["Error95Percentile_mm"]),
                        "Mean_Top25ErrorThreshold_mm": safe_mean(sub["Top25ErrorThreshold_mm"]),
                        "Mean_Top25_MAE_mm": safe_mean(sub["Top25_MAE_mm"]),
                        "Mean_Top25_RMSE_mm": safe_mean(sub["Top25_RMSE_mm"]),
                    }
                    rows.append(row)

    return pd.DataFrame(rows)


def sort_error_df(df: pd.DataFrame):
    if len(df) == 0:
        return df

    region_order = {"ROI": 0, "FullImage": 1}
    method_order = {key: i for i, key in enumerate(METHODS.keys())}

    df = df.copy()
    df["_region_order"] = df["Region"].map(region_order).fillna(99)
    df["_method_order"] = df["Method"].map(method_order).fillna(99)

    sort_cols = ["DepthCase", "_method_order", "_region_order", "Model", "Image"]
    sort_cols = [c for c in sort_cols if c in df.columns]

    df = df.sort_values(
        by=sort_cols,
        ascending=[True] * len(sort_cols)
    ).drop(columns=["_region_order", "_method_order"], errors="ignore")

    return df


# =========================
# 6. Output utilities
# =========================
def save_raw_depth_png(pred: np.ndarray, out_path: Path, ref_path: str, depth_scale: float):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ref = np.array(Image.open(ref_path))
    raw = pred * float(depth_scale)

    if np.issubdtype(ref.dtype, np.integer):
        info = np.iinfo(ref.dtype)
        raw = np.nan_to_num(raw, nan=0.0, posinf=info.max, neginf=0.0)
        raw = np.clip(raw, info.min, info.max).astype(ref.dtype)
    else:
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(ref.dtype)

    Image.fromarray(raw).save(out_path)


def make_color_depth(depth: np.ndarray, valid_mask: np.ndarray = None, vmin=None, vmax=None):
    d = depth.astype(np.float32)
    finite = np.isfinite(d)
    if valid_mask is not None:
        finite = finite & valid_mask

    if vmin is None:
        vmin = float(np.nanmin(d[finite])) if np.any(finite) else 0.0
    if vmax is None:
        vmax = float(np.nanmax(d[finite])) if np.any(finite) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    norm = (d - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)

    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    rgb = (cmap(gray)[:, :, :3] * 255).astype(np.uint8)

    if valid_mask is not None:
        rgb[~valid_mask] = 0

    return rgb


def save_compare_figure(gt: np.ndarray, pred: np.ndarray, out_path: Path, title: str = ""):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    valid = valid_depth_mask(gt) & np.isfinite(pred)
    if np.any(valid):
        vmin = float(np.nanpercentile(np.concatenate([gt[valid], pred[valid]]), 1))
        vmax = float(np.nanpercentile(np.concatenate([gt[valid], pred[valid]]), 99))
        if vmax <= vmin:
            vmax = float(np.nanmax(gt[valid]))
            vmin = float(np.nanmin(gt[valid]))
    else:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    im0 = axes[0].imshow(gt, cmap="Spectral_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("GT")
    axes[0].axis("off")

    im1 = axes[1].imshow(pred, cmap="Spectral_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=10)

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.78)
    cbar.set_label("Depth (mm)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)




def save_pred_outputs(pred: np.ndarray, method_out_dir: Path, stem: str, ref_depth_path: str, depth_scale: float, save_npy: bool):
    raw_dir = method_out_dir / "01_predictions"
    color_dir = method_out_dir / "02_colorized_predictions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)

    raw_png_path = raw_dir / f"{stem}_pred_raw.png"
    color_png_path = color_dir / f"{stem}_pred_color.png"

    save_raw_depth_png(pred, raw_png_path, ref_path=ref_depth_path, depth_scale=depth_scale)

    npy_path = None
    if save_npy:
        npy_path = raw_dir / f"{stem}_pred.npy"
        np.save(npy_path, pred.astype(np.float32))

    pred_valid = np.isfinite(pred)
    color = make_color_depth(pred, valid_mask=pred_valid)
    Image.fromarray(color).save(color_png_path)

    return raw_png_path, color_png_path, npy_path


def save_method_error_files(df: pd.DataFrame, method_out_dir: Path, case_name: str, method_key: str):
    error_dir = method_out_dir / "04_metrics"
    error_dir.mkdir(parents=True, exist_ok=True)

    df = sort_error_df(df)

    safe_method = sanitize_filename(method_key)
    xlsx_path = error_dir / f"{case_name}_{safe_method}_metrics.xlsx"
    csv_path = error_dir / f"{case_name}_{safe_method}_per_image_metrics.csv"
    summary_csv_path = error_dir / f"{case_name}_{safe_method}_summary.csv"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Per-image metrics", index=False)
        if len(df) > 0:
            summary_df = make_summary(df)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return xlsx_path, csv_path


def add_metric_rows(case_rows, all_rows, method_rows_map, case_name, stem, method_key, pred, gt_arr, roi_mask, full_mask):
    method_info = METHODS[method_key]

    for region_name, eval_mask in [("ROI", roi_mask), ("FullImage", full_mask)]:
        metrics = drop_unwanted_excel_fields(compute_metrics(pred, gt_arr, eval_mask))
        row = {
            "DepthCase": case_name,
            "Image": stem,
            "Region": region_name,
            "Method": method_key,
            "Description": method_info["desc"],
            "Model": method_info["model"],
            "Branch": method_info["branch"],
            **metrics,
        }
        case_rows.append(row)
        all_rows.append(row)
        method_rows_map[method_key].append(row)


# =========================
# 7. Main evaluation
# =========================


def main():
    total_start_time = time.time()

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] device = {device}")
    print("[Info] Evaluating the final Spall-to-Full model:")
    for key, info in METHODS.items():
        print(f"       {key}: {info['desc']}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rgb_files = list_images(args.rgb_dir, recursive=args.recursive_rgb)
    if len(rgb_files) == 0:
        raise RuntimeError(f"No RGB images found in: {args.rgb_dir}")

    mask_map = build_file_map(args.mask_dir, recursive=args.recursive_mask)
    roi_map = build_file_map(args.roi_mask_dir, recursive=args.recursive_roi)

    # Ground-truth depth is read from the root directory only.
    gt_map = build_file_map(args.gt_depth_dir, recursive=False)

    all_rows = []

   
    model = create_spall_to_full_model(
        args,
        device,
        max_depth=1487.0,
    )
    for depth_dir in args.depth_input_dirs:
        case_start_time = time.time()

        depth_dir = str(depth_dir)
        case_name = Path(depth_dir).name
        print("\n" + "=" * 90)
        print(f"[Case] {case_name}")
        print("=" * 90)

        depth_map = build_file_map(depth_dir, recursive=False)

        case_out = out_root / case_name
        case_out.mkdir(parents=True, exist_ok=True)

        method_out_dirs = OrderedDict()
        for method_key, method_info in METHODS.items():
            method_out_dirs[method_key] = case_out / method_info["folder"]
            (method_out_dirs[method_key] / "01_predictions").mkdir(parents=True, exist_ok=True)
            (method_out_dirs[method_key] / "02_colorized_predictions").mkdir(parents=True, exist_ok=True)
            (method_out_dirs[method_key] / "03_gt_prediction_comparisons").mkdir(parents=True, exist_ok=True)
            (method_out_dirs[method_key] / "04_metrics").mkdir(parents=True, exist_ok=True)

        case_rows = []
        method_rows_map = OrderedDict((key, []) for key in METHODS.keys())

        for idx, rgb_path in enumerate(rgb_files, start=1):
            image_start_time = time.time()

            stem = Path(rgb_path).stem

            depth_path = find_by_stem(depth_map, stem)
            mask_path = find_by_stem(mask_map, stem)
            gt_path = find_by_stem(gt_map, stem)
            roi_path = find_by_stem(roi_map, stem)

            if depth_path is None or mask_path is None or gt_path is None or roi_path is None:
                print(f"[Skip] {stem}: missing paired files "
                      f"depth={depth_path is not None}, mask={mask_path is not None}, "
                      f"gt={gt_path is not None}, roi={roi_path is not None}")
                continue

            print(f"[{idx}/{len(rgb_files)}] {stem}")

            rgb_t, size_hw = load_rgb_tensor(rgb_path)
            dep_t, dep_arr = load_depth_tensor_aligned(depth_path, size_hw, args.depth_scale)
            mask_t, mask_arr = load_mask_onehot_aligned(mask_path, size_hw)

            gt_arr = read_depth_array(gt_path, depth_scale=args.depth_scale)
            gt_arr = resize_np(gt_arr, size_hw, is_mask_or_depth=True).astype(np.float32)

            roi_mask = roi_bool_mask(roi_path, size_hw)
            full_mask = np.ones(size_hw, dtype=bool)

            max_depth = get_input_known_max(dep_arr, fallback=1487.0)

            pred = predict_spall_to_full_once(
                model,
                rgb_t=rgb_t,
                dep_t=dep_t,
                mask_t=mask_t,
                device=device,
                max_depth=max_depth,
            )

            if pred.shape != size_hw:
                pred = resize_np(
                    pred,
                    size_hw,
                    is_mask_or_depth=True,
                ).astype(np.float32)

            preds = OrderedDict([
                ("spall_to_full", pred),
            ])
            # =========================================================
            # Information and Data Preservation
            # =========================================================
            for method_key, pred in preds.items():
                method_info = METHODS[method_key]
                method_out_dir = method_out_dirs[method_key]
                safe_method = sanitize_filename(method_key)

                save_pred_outputs(
                    pred,
                    method_out_dir,
                    stem=f"{stem}_{safe_method}",
                    ref_depth_path=depth_path,
                    depth_scale=args.depth_scale,
                    save_npy=args.save_npy,
                )

                compare_dir = method_out_dir / "03_gt_prediction_comparisons"

                save_compare_figure(
                    gt_arr,
                    pred,
                    compare_dir / f"{stem}_{safe_method}_gt_vs_pred.png",
                    title=f"{case_name} | {stem} | {method_info['desc']}"
                )

            
                add_metric_rows(
                    case_rows=case_rows,
                    all_rows=all_rows,
                    method_rows_map=method_rows_map,
                    case_name=case_name,
                    stem=stem,
                    method_key=method_key,
                    pred=pred,
                    gt_arr=gt_arr,
                    roi_mask=roi_mask,
                    full_mask=full_mask,
                )

            image_elapsed = time.time() - image_start_time
            print(f"    [Time] {stem} completed in {image_elapsed:.2f} s")

        # =========================================================
        # Save metrics for the current method.
        # =========================================================
        for method_key, rows in method_rows_map.items():
            method_df = pd.DataFrame(rows)
            method_out_dir = method_out_dirs[method_key]
            method_xlsx, method_csv = save_method_error_files(
                method_df,
                method_out_dir,
                case_name=case_name,
                method_key=method_key,
            )
            print(f"[Saved] {method_xlsx}")
            print(f"[Saved] {method_csv}")

        # =========================================================
        # Save metrics for the current sparse-depth case.
        # =========================================================
        case_df = pd.DataFrame(case_rows)
        case_df = sort_error_df(case_df)

        case_xlsx = case_out / f"{case_name}_metrics.xlsx"
        case_csv = case_out / f"{case_name}_per_image_metrics.csv"
        case_summary_csv = case_out / f"{case_name}_summary.csv"

        with pd.ExcelWriter(case_xlsx, engine="openpyxl") as writer:
            case_df.to_excel(writer, sheet_name="Per-image metrics", index=False)
            if len(case_df) > 0:
                case_summary_df = make_summary(case_df)
                case_summary_df.to_excel(writer, sheet_name="Summary", index=False)
                case_summary_df.to_csv(case_summary_csv, index=False, encoding="utf-8-sig")

        case_df.to_csv(case_csv, index=False, encoding="utf-8-sig")

        case_elapsed = time.time() - case_start_time
        print(f"[Saved] {case_xlsx}")
        print(f"[Saved] {case_csv}")
        print(
            f"[Case Time] {case_name} completed in "
            f"{case_elapsed:.2f} s / {case_elapsed / 60.0:.2f} min"
        )

    # =========================================================
    # Save metrics across all sparse-depth cases.
    # =========================================================
    all_df = pd.DataFrame(all_rows)
    all_df = sort_error_df(all_df)

    total_xlsx = out_root / "all_cases_metrics.xlsx"
    total_csv = out_root / "all_cases_per_image_metrics.csv"
    total_summary_csv = out_root / "all_cases_summary.csv"

    with pd.ExcelWriter(total_xlsx, engine="openpyxl") as writer:
        all_df.to_excel(writer, sheet_name="per_image_metrics", index=False)
        if len(all_df) > 0:
            total_summary_df = make_summary(all_df)
            total_summary_df.to_excel(writer, sheet_name="all_cases_summary", index=False)
            total_summary_df.to_csv(total_summary_csv, index=False, encoding="utf-8-sig")

    all_df.to_csv(total_csv, index=False, encoding="utf-8-sig")

    total_elapsed = time.time() - total_start_time
    print("\n[Done] Evaluation completed.")
    print(f"[Output root] {out_root}")
    print(f"[Total Excel] {total_xlsx}")
    print(f"[Total CSV] {total_csv}")
    print(f"[Total Summary CSV] {total_summary_csv}")
    print(
        f"[Total Time] {total_elapsed:.2f} s / "
        f"{total_elapsed / 60.0:.2f} min / "
        f"{total_elapsed / 3600.0:.3f} h"
    )


if __name__ == "__main__":
    main()
