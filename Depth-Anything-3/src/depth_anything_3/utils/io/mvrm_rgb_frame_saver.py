"""Utilities for saving per-frame RGB decoder outputs for MVRM evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import torch
import torchvision.transforms.functional as TF


DEFAULT_MVRM_RESTORED_ROOT = "result_val/mvrm_restored"


def _to_vchw(x: torch.Tensor, name: str) -> torch.Tensor:
    """Convert supported tensor layouts to (V, 3, H, W)."""
    if x.ndim == 5:
        # Expected shape is (B, V, 3, H, W) during inference.
        if x.shape[0] != 1:
            raise ValueError(f"{name} must have batch size 1, got {x.shape[0]}")
        x = x.squeeze(0)
    if x.ndim != 4:
        raise ValueError(f"{name} must be 4D or 5D, got shape {tuple(x.shape)}")
    if x.shape[1] != 3:
        raise ValueError(f"{name} channel dim must be 3, got shape {tuple(x.shape)}")
    return x.float()


def _denorm_imagenet(vchw: torch.Tensor) -> torch.Tensor:
    """Convert ImageNet-normalized tensor to [0, 1]."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=vchw.device, dtype=vchw.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=vchw.device, dtype=vchw.dtype).view(1, 3, 1, 1)
    return (vchw * std + mean).clamp(0, 1)


def _frame_stems_from_paths(image_files: Iterable[str], num_views: int) -> list[str]:
    stems: list[str] = []
    for p in image_files:
        stem = Path(p).stem
        stems.append(stem if stem else f"frame_{len(stems):04d}")

    if len(stems) < num_views:
        stems.extend([f"frame_{i:04d}" for i in range(len(stems), num_views)])
    return stems[:num_views]


def save_mvrm_decoder_rgb_frames(
    rgb_hq: torch.Tensor,
    rgb_lq: torch.Tensor,
    rgb_res: torch.Tensor,
    scene: str,
    image_files: Iterable[str],
    output_root: str = DEFAULT_MVRM_RESTORED_ROOT,
    denorm_imagenet: bool = True,
) -> dict[str, list[str]]:
    """Save decoder RGB outputs per frame under hq/lq/res folders.

    Directory layout:
        output_root/
            hq/<scene>/<frame>.png
            lq/<scene>/<frame>.png
            res/<scene>/<frame>.png

    Args:
        rgb_hq: HQ decoder output, shape (B, V, 3, H, W) or (V, 3, H, W).
        rgb_lq: LQ decoder output, shape (B, V, 3, H, W) or (V, 3, H, W).
        rgb_res: Restored decoder output, shape (B, V, 3, H, W) or (V, 3, H, W).
        scene: Scene name used as directory name.
        image_files: Original frame paths used to derive frame file names.
        output_root: Root output directory.
        denorm_imagenet: If True, apply ImageNet de-normalization before saving.

    Returns:
        Mapping of split name to list of saved file paths.
    """
    hq = _to_vchw(rgb_hq, "rgb_hq")
    lq = _to_vchw(rgb_lq, "rgb_lq")
    res = _to_vchw(rgb_res, "rgb_res")

    num_views = hq.shape[0]
    if lq.shape[0] != num_views or res.shape[0] != num_views:
        raise ValueError(
            f"View count mismatch: hq={hq.shape[0]}, lq={lq.shape[0]}, res={res.shape[0]}"
        )

    if denorm_imagenet:
        hq = _denorm_imagenet(hq)
        lq = _denorm_imagenet(lq)
        res = _denorm_imagenet(res)
    else:
        hq = hq.clamp(0, 1)
        lq = lq.clamp(0, 1)
        res = res.clamp(0, 1)

    frame_stems = _frame_stems_from_paths(image_files, num_views)
    outputs = {"hq": hq, "lq": lq, "res": res}

    saved_paths: dict[str, list[str]] = {"hq": [], "lq": [], "res": []}
    for split, tensor in outputs.items():
        scene_dir = os.path.join(output_root, split, scene)
        os.makedirs(scene_dir, exist_ok=True)

        for i in range(num_views):
            file_name = f"{frame_stems[i]}.png"
            save_path = os.path.join(scene_dir, file_name)
            TF.to_pil_image(tensor[i].cpu()).save(save_path)
            saved_paths[split].append(save_path)

    return saved_paths
