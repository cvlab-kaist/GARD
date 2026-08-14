import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
import cv2 

EPS = 1e-6

@dataclass(frozen=True)
class DepthEvalConfig:
    min_depth: float = 1e-3
    max_depth: float = float("inf")
    delta_thresholds: Tuple[float, float, float] = (1.25, 1.25**2, 1.25**3)


def _valid_mask(pred: np.ndarray, gt: np.ndarray, gt_valid: np.ndarray, cfg: DepthEvalConfig) -> np.ndarray:
    m = gt_valid.astype(bool)
    m &= np.isfinite(pred) & np.isfinite(gt)
    m &= gt > cfg.min_depth
    if np.isfinite(cfg.max_depth):
        m &= gt < cfg.max_depth
    return m

def resize_depth_nearest(depth: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    if depth.shape[0] == h and depth.shape[1] == w:
        return depth
    return cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

def abs_rel(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt) / np.maximum(gt, EPS)))


def sq_rel(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean((pred - gt) ** 2 / np.maximum(gt, EPS)))


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def rmse_log(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.maximum(pred, EPS)
    gt = np.maximum(gt, EPS)
    return float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2)))


def delta_accuracy(pred: np.ndarray, gt: np.ndarray, thr: float) -> float:
    pred = np.maximum(pred, EPS)
    gt = np.maximum(gt, EPS)
    ratio = np.maximum(pred / gt, gt / pred)
    return float(np.mean(ratio < thr) * 100.0)


def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    gt_valid_mask: np.ndarray,
    cfg: DepthEvalConfig = DepthEvalConfig()
) -> Dict[str, float]:

    pred = np.asarray(pred_depth, dtype=np.float32)
    gt = np.asarray(gt_depth, dtype=np.float32)

    valid = _valid_mask(pred, gt, gt_valid_mask, cfg)

    n_valid = int(valid.sum())
    if n_valid == 0:
        return {}

    p = pred[valid]
    g = gt[valid]

    if g.max() > 100:  
        g = g / 1000.0
    if p.max() > 100: 
        p = p / 1000.0

    scale = np.median(g) / (np.median(p) + EPS)
    p = p * scale

    p = np.clip(p, cfg.min_depth, cfg.max_depth)

    metrics = {}
    metrics["abs_rel"] = abs_rel(p, g)
    metrics["sq_rel"] = sq_rel(p, g)
    metrics["rmse"] = rmse(p, g)
    metrics["rmse_log"] = rmse_log(p, g)

    for i, thr in enumerate(cfg.delta_thresholds, start=1):
        metrics[f"d{i}"] = delta_accuracy(p, g, thr)

    metrics["valid_pixels_pct"] = 100.0 * n_valid / max(valid.size, 1)

    return metrics