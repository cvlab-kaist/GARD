<div align="center">

<h1>
GARD: Geometry-Aware Representation Denoising for <br>Robust Multi-view 3D Reconstruction</h1>



[**Jin Hyeon Kim**](https://github.com/jinlovespho)<sup>1*</sup>,&nbsp;&nbsp;
[**Jaeeun Lee**](https://github.com/babywhale03)<sup>1*</sup>,&nbsp;&nbsp;
**Claire Kim**<sup>1</sup>,&nbsp;&nbsp;
**Kyoungjin Oh**<sup>1</sup>,&nbsp;&nbsp;
**Paul Hyunbin Cho**<sup>1</sup>,&nbsp;&nbsp;
[**Jaewon Min**](https://github.com/Min-Jaewon/)<sup>1</sup>,&nbsp;&nbsp; 
**Yeji Choi**<sup>1</sup>,&nbsp;&nbsp;
<br>
**Jihye Park**<sup>2</sup>,&nbsp;&nbsp;
**Hyunhee Park**<sup>2</sup>,&nbsp;&nbsp;
**Minkyu Park**<sup>2</sup>,&nbsp;&nbsp;
[**Seungryong Kim**](https://scholar.google.com/citations?hl=zh-CN&user=cIK1hS8AAAAJ)<sup>1&dagger;</sup>

  <p align="center">
    <sup>1</sup> KAIST&nbsp;AI · 
    <sup>2</sup> Samsung&nbsp;Electronics
  </p>

  <p align="center" style="font-size: 0.9em; color: gray;">
    <sup>*</sup> Equal&nbsp;contribution. <sup>&dagger;</sup> Corresponding&nbsp;author.
  </p>

<a href="https://arxiv.org/abs/2605.26230">
  <img src="https://img.shields.io/badge/arXiv-2605.26230-B31B1B">
</a>
        <a href="https://cvlab-kaist.github.io/GARD/"><img src="https://img.shields.io/badge/Project%20Page-online-1E90FF"></a>

</p>

<img src="assets/teaser.png" width="850">

</div>

# 🔈 News 
- 📄 **[2026-05-25]** GARD paper released in [arxiv](https://arxiv.org/abs/2605.26230) 
- 🔥 **[2026-08-16]**  Initial release of the training/inferece code, data, and model weights



# 🚀 Overview

Feed-forward multi-view 3D reconstruction models perform well under clean, ideal imaging conditions, but degrade sharply on real-world captures corrupted by motion blur, noise, and other artifacts. **GARD** (Geometry-Aware Representation Denoising) addresses this by performing diffusion-based multi-view restoration *directly in the feature space* of a feed-forward 3D reconstruction model ([Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3)), rather than restoring pixels first and reconstructing geometry afterward. By denoising geometry-aware feature representations, GARD recovers accurate scene geometry from degraded multi-view inputs; an accompanying RGB decoder then reconstructs high-quality images from the same denoised representations, enabling the **simultaneous recovery of 3D scene geometry and high-quality imagery** in a single restoration pass.

This repository provides:
- The GARD denoiser implementation ([`src/gard/GARD.py`](src/gard/GARD.py)) and its pretrained checkpoints
- Training code and recipes on Hypersim + TartanAir with synthesized multi-view degradations
- Evaluation code on [DA3-BENCH](https://huggingface.co/datasets/depth-anything/DA3-BENCH) (paired with our own synthesized degradations) and on real-world camera motion blur scenes from [DeblurNeRF](https://limacv.github.io/deblurnerf/)





#  👟 Installation Walkthrough

```bash
# clone the GARD repo 
git clone https://github.com/cvlab-kaist/GARD.git

# Move inside the GARD repo 
cd GARD/

# Set up the environment using uv 
uv sync 

# Activate the uv environment
source .venv/bin/activate
```



# 🚀 Inference 

### 🎯 Checkpoint preparation
Download the GARD checkpoints from [HuggingFace](https://huggingface.co/jinlovespho/GARD) into `ckpts/`:
```bash
bash download_scripts/gard_ckpt/download_ckpts.sh
```

| File | Description | Size |
|------|-------------|------|
| `ckpts/gard_denoiser.pt` | GARD denoiser (DiT with a DDT head) that denoises DA3 feature representations | 5.6G |
| `ckpts/mae_adapter_giant.pt` | RGB decoder that reconstructs high-quality images from the denoised representations | 2.0G |

### 📁 Evaluation data preparation
GARD is evaluated on [DA3-BENCH](https://huggingface.co/datasets/depth-anything/DA3-BENCH) (7-Scenes, DTU, DTU64, ETH3D, HiRoom, ScanNet++) paired with our own synthesized degradations, and on the real-world camera motion blur scenes from [DeblurNeRF](https://limacv.github.io/deblurnerf/). The clean DA3-BENCH images are re-downloaded from their official source (please review and comply with each dataset's original license); our synthesized degraded counterpart of DA3-BENCH and a mirror of the DeblurNeRF real captures (the original Google Drive folder download is unreliable at this file count) are hosted on our own [HuggingFace dataset](https://huggingface.co/datasets/jinlovespho/GARD-eval-bench).
```bash
# downloads the clean + degraded DA3-BENCH images and the real camera motion blur scenes into data/eval/
bash download_scripts/eval_bench/download_eval_data.sh
```

### 🏃 Running Evaluation
```bash
# evaluate on DA3-BENCH (pose / depth / unposed reconstruction)
bash run_scripts/val/val_GARD_da3_bench.sh

# evaluate on the real-world camera motion blur scenes
bash run_scripts/val/val_GARD_real_bench.sh
```
Results are written to `result_val/GARD_da3_bench/` and `result_val/GARD_real_bench/` respectively.




# 🔥 Training 

### 📁 Training data preparation
```bash
# download hypersim
bash download_scripts/hypersim/download_hypersim.sh

# download tartanair 
bash download_scripts/tartanair_tools/download_tartanair.sh
```



### 🛠️ Training Recipe
```bash
bash run_scripts/train/train_GARD.sh
```
Trains the GARD denoiser using [`run_configs/train/train_GARD.yaml`](run_configs/train/train_GARD.yaml) — every field (model architecture, loss weights, data/degradation settings, logging, etc.) is documented inline as comments there.

**GPU setup**: set `NUM_GPUS` and `CUDA` (passed through as `CUDA_VISIBLE_DEVICES`) in [`run_scripts/train/train_GARD.sh`](run_scripts/train/train_GARD.sh) to however many GPUs / which device(s) you want to train on.

## Citation

```
@article{kim2026geometry,
  title={Geometry-Aware Representation Denoising for Robust Multi-view 3D Reconstruction},
  author={Kim, Jin Hyeon and Lee, Jaeeun and Kim, Claire and Oh, Kyoungjin and Cho, Paul Hyunbin and Min, Jaewon and Choi, Yeji and Park, Jihye and Park, Hyunhee and Park, Minkyu and others},
  journal={arXiv preprint arXiv:2605.26230},
  year={2026}
}
```

## Acknowledgement
We thank the authors of [RAE](https://github.com/bytetriper/RAE), [DepthAnything3](https://github.com/bytedance-seed/depth-anything-3), [GLD](https://github.com/cvlab-kaist/GLD), and [Motionblur](https://github.com/LeviBorodenko/motionblur) for their excellent work and code, which served as the foundation for this project.