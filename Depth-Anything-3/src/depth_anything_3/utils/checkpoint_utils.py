# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Resolves a checkpoint config value to a local file path.

Most weights in this repo (e.g. `encoder_pretrained_path: depth-anything/DA3-Base`)
are already resolved through a model's own `from_pretrained`. Custom checkpoints
such as `mae_weight` and `denoiser.ckpt` are plain `torch.load(path)` calls, which
only ever accepted a local filesystem path. This module extends that convention
with a lightweight `hf://` scheme so those same config fields can also point at a
file hosted on the Hugging Face Hub, without breaking existing local-path configs.
"""

from __future__ import annotations

import os
from typing import Optional

HF_SCHEME = "hf://"


def resolve_ckpt_path(path: Optional[str], cache_dir: Optional[str] = None) -> Optional[str]:
    """
    Resolve a checkpoint path/URI to a local file path.

    Accepts:
      - A local path (absolute or relative), returned unchanged if it exists.
      - An `hf://<owner>/<repo>/<filename>` URI, e.g.
            "hf://your-username/GARD-weights/mae_adapter_giant.pt"
        which is downloaded (and cached) via `huggingface_hub.hf_hub_download`.

    Args:
        path: Local path or "hf://..." URI. None is passed through unchanged.
        cache_dir: Optional override for the Hugging Face Hub cache directory.

    Returns:
        A local filesystem path to the checkpoint, or None if `path` is None.
    """
    if path is None:
        return None

    if path.startswith(HF_SCHEME):
        from huggingface_hub import hf_hub_download

        repo_and_file = path[len(HF_SCHEME):]
        parts = repo_and_file.split("/")
        if len(parts) < 3:
            raise ValueError(
                f"Malformed checkpoint URI: {path!r}. "
                "Expected 'hf://<owner>/<repo>/<filename>'."
            )
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path!r}. Provide a valid local path, or an "
            "'hf://<owner>/<repo>/<filename>' URI to fetch it from the Hugging Face Hub."
        )
    return path
