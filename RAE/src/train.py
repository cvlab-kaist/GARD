# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import argparse
import math
import os
from collections import defaultdict
import cv2 
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm 
import argparse
from pathlib import Path
import math
from omegaconf import OmegaConf

from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors   
from depth_anything_3.utils.geometry import unproject_depth, affine_inverse, as_homogeneous


##### model imports
from stage2.transport import create_transport, Sampler

##### general utils
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import *
from utils.wandb_utils import *
from utils.dist_utils import *
from utils.vis_utils import *
from utils.loss_utils import velocity_direction_loss, camera_loss_single

import torch.nn.functional as F 
from torchvision.utils import save_image 
from utils.vis_utils import depth_to_colormap, depth_error_to_colormap_thresholded, tensor_to_uint8_image
import torchvision

from einops import rearrange
from RAE.src import initialize
from motionblur.motionblur import Kernel 
import matplotlib.pyplot as plt


from mvr.utils.featsim_utils import *
from mvr.utils.metric_utils import *
from mvr.utils.freq_utils import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 transport model on RAE latents.")
    parser.add_argument("--config", type=str, required=True, help="YAML config containing stage_1 and stage_2 sections.")
    args = parser.parse_args()
    return args


def main():
    
    # set up ddp setting
    rank, world_size, device = setup_distributed()
    
    # load configs
    args = parse_args()
    full_cfg = OmegaConf.load(args.config)
    training_cfg = full_cfg.training 
    
    # set logger and directories
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(full_cfg, rank)
    
    # config setting
    time_dist_shift = math.sqrt(full_cfg.misc.time_dist_shift_dim / full_cfg.misc.time_dist_shift_base)
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    num_epochs = int(training_cfg.get("epochs", 1400))
    global_batch_size = training_cfg.get("global_batch_size", None) # optional global batch size for override
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
    else:
        batch_size = int(training_cfg.get("batch_size", 16))
        global_batch_size = batch_size * world_size * grad_accum_steps
    log_interval = int(training_cfg.get("log_interval", 100))
    ckpt_step_interval = int(training_cfg.get('ckpt_step_interval', 25000))
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    global_seed = int(training_cfg.get("global_seed", 0))
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    
    # load encoder and denoiser 
    models, processors = initialize.load_model(full_cfg, rank, device)
    
    # load training and validation data 
    train_loader, train_sampler = initialize.load_train_data(full_cfg, micro_batch_size, rank, world_size)
    loader_batches = len(train_loader)
    steps_per_epoch = math.ceil(loader_batches / grad_accum_steps)

    # load optimizer
    optimizer, optim_msg = build_optimizer([p for p in models['denoiser'].parameters() if p.requires_grad], training_cfg)

    # load scheduler 
    if full_cfg.get('scheduler'):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)
    else:
        scheduler=None
        sched_msg=None
    
    # load Transport 
    transport = create_transport(**full_cfg.transport.params, time_dist_shift=time_dist_shift,)
    transport_sampler = Sampler(transport)

    # load sampler 
    eval_sampler = initialize.load_sampler(full_cfg, transport_sampler)
    ema_model_fn = models['ema_denoiser'].forward
    val_noise_generator = torch.Generator(device=device)
    val_noise_generator.manual_seed(global_seed)  # any fixed seed you like

    
    ### Resuming and checkpointing
    start_epoch = 0
    global_train_step = 0
    optimizer_step = 0 
    running_loss = 0.0

    
    # maybe_resume_ckpt_path = find_resume_checkpoint(experiment_dir)
    maybe_resume_ckpt_path = full_cfg.stage_2.ckpt
    if maybe_resume_ckpt_path is not None:
        logger.info(f"Experiment resume checkpoint found at {maybe_resume_ckpt_path}, automatically resuming...")
        ckpt_path = Path(maybe_resume_ckpt_path)
        if ckpt_path.is_file():
            start_epoch, global_train_step = initialize.load_checkpoint(
                ckpt_path,
                models['ddp_denoiser'],
                models['ema_denoiser'],
                optimizer,
                scheduler,
            )
            logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_train_step}).")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # starting from fresh, save worktree and configs
        if rank == 0:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")
            
            
    ### Logging experiment details
    if rank == 0:
        num_params = sum(p.numel() for p in models['encoder'].parameters())
        logger.info(f"Stage-1 Encoder parameters: {num_params/1e6:.2f}M")
        num_params = sum(p.numel() for p in models['denoiser'].parameters() if p.requires_grad)
        logger.info(f"Stage-2 Denoiser parameters: {num_params/1e6:.2f}M")
        logger.info(f"Clipping gradients to max norm {clip_grad}.")
        # print optim and schel
        logger.info(optim_msg)
        print(sched_msg if sched_msg else "No LR scheduler.")
        logger.info(f"Training for {num_epochs} epochs, batch size {micro_batch_size} per GPU. grad accum {full_cfg.training.grad_accum_steps} per GPU")
        logger.info(f"Dataset contains total {len(train_loader.dataset)} training samples, {steps_per_epoch} steps per epoch.")
        for train_ds in train_loader.dataset.datasets:
            logger.info(f'  - {train_ds.ds_name}: {len(train_ds)}')
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")

    IMAGENET_NORMALIZE = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],)

    dist.barrier() 
    for epoch in range(start_epoch, num_epochs):
        models['ddp_denoiser'].train()
        train_sampler.set_epoch(epoch)
        epoch_metrics = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)

        # train loop
        for train_step, batch in enumerate(train_loader):

            # load batch data
            train_hq_id = batch['hq_ids']                     # len(hq_id) = b, len(hq_id[i]) = v
            train_hq_views = batch['hq_views'].to(device)     # b v 3 378 504
            train_lq_views = batch['lq_views'].to(device)     # b v 3 378 504

            # apply imagenet normalization
            train_b, train_v, train_c, train_h, train_w = train_hq_views.shape 
            train_hq_views = IMAGENET_NORMALIZE(train_hq_views.view(train_b*train_v, train_c, train_h, train_w)).view(train_b, train_v, train_c, train_h, train_w)
            train_lq_views = IMAGENET_NORMALIZE(train_lq_views.view(train_b*train_v, train_c, train_h, train_w)).view(train_b, train_v, train_c, train_h, train_w)
            print(train_hq_views.shape)
            
            # lq view forward pass
            with torch.no_grad():
                lq_encoder_out, lq_mvrm_out = models['encoder'](
                                                    image=train_lq_views, 
                                                    export_feat_layers=[], 
                                                    mvrm_cfg=full_cfg.mvrm.train, 
                                                    mode='train'
                                                    )
            lq_pred_pose_enc = lq_encoder_out.pose_enc
            lq_pred_pose = lq_encoder_out['extrinsics'] # b v 3 4
            lq_ref_b_idx = lq_encoder_out.ref_b_idx
            lq_encoder_out = processors['encoder_output_processor'](lq_encoder_out)
            train_lq_pred_depth_np = lq_encoder_out.depth                  # b v 378 504
            train_lq_pred_depth = torch.from_numpy(train_lq_pred_depth_np).to(device) 
            lq_latent = lq_mvrm_out[('extract_feat', full_cfg.mvrm.train.extract_feat_layers[0])]
                        
            # hq forward pass
            with torch.no_grad():
                hq_encoder_out, hq_mvrm_out = models['encoder'](
                                                    image=train_hq_views, 
                                                    export_feat_layers=[], 
                                                    mvrm_cfg=full_cfg.mvrm.train, 
                                                    mode='train',
                                                    ref_b_idx=lq_ref_b_idx,
                                                    # ref_b_idx=None
                                                    analysis = full_cfg.analysis_HQ
                                                    )
            hq_pred_pose_enc = hq_encoder_out.pose_enc
            hq_pred_intrinsics = hq_encoder_out['intrinsics']  # (b, v, 3, 3) tensor
            hq_pred_pose = hq_encoder_out['extrinsics'] # b v 3 4
            hq_encoder_out = processors['encoder_output_processor'](hq_encoder_out)
            train_hq_pred_depth_np = hq_encoder_out.depth                  # b v 378 504
            train_hq_pred_depth = torch.from_numpy(train_hq_pred_depth_np).to(device) 
            hq_latent = hq_mvrm_out[('extract_feat', full_cfg.mvrm.train.extract_feat_layers[0])]
            assert lq_latent.shape == hq_latent.shape 
            
            # processing for when batch size = 1
            if train_b==1 and len(train_hq_pred_depth_np.shape)<4 and len(train_lq_pred_depth_np.shape)<4:
                train_hq_pred_depth_np = np.expand_dims(train_hq_pred_depth_np, axis=0)
                train_lq_pred_depth_np = np.expand_dims(train_lq_pred_depth_np, axis=0)                


            # compute loss (per microbatch)
            transport_output = transport.training_losses_mvrm(
                model=models['ddp_denoiser'],
                x1=hq_latent,
                xcond=lq_latent,
                model_img_size=(train_h, train_w),
                cfg=full_cfg
            )
            
            
            mvrm_maps = None 
            if full_cfg.mvrm.analysis.vis_attn_map:
                mvrm_maps = transport_output.get('mvrm_maps', None)
                
                            
            # flow matching velocity loss
            transport_loss = transport_output["loss"].mean()
            loss = transport_loss
                        
                        
            def cross_entropy_attn(pred, target, row_mask=None, eps=1e-8):
                """CAMEO-style cross-entropy between attention probability distributions.
                row_mask: (b, v*n) bool tensor; if given, averages only over True rows."""
                per_row = -(target * (pred + eps).log()).sum(dim=-1)  # (b, v*n)
                if row_mask is not None:
                    return per_row[row_mask].mean()
                return per_row.mean()

            # ---------------------------
            # Attention Alignment
            # ---------------------------
            attn_loss = torch.tensor(0.0, device=device)
            if full_cfg.mvrm.loss.attn_align.use and mvrm_maps is not None:
                lambda_attn = full_cfg.mvrm.loss.attn_align.lambda_coeff
                mvrm_key = ('mvrm', full_cfg.mvrm.loss.attn_align.mvrm_layer_idx, 'global')
                pred_map = mvrm_maps[mvrm_key]  # (1, v*(n+1), v*(n+1))
                if full_cfg.mvrm.loss.attn_align.use and full_cfg.mvrm.loss.attn_align.da3_point_cloud.use:
                    print(f'attention alignment - HQ point cloud correspondence map - temperature {full_cfg.mvrm.loss.attn_align.da3_point_cloud.get("vis_pc_temperature", "None")}')
                    # target: geometric correspondence from HQ point cloud
                    depth_t = train_hq_pred_depth.unsqueeze(-1).float()                 # (b, v, H, W, 1)
                    c2w_t   = affine_inverse(as_homogeneous(hq_pred_pose.float()))      # (b, v, 4, 4)
                    pts = unproject_depth(depth_t, hq_pred_intrinsics.float(), c2w_t)  # (b, v, H, W, 3)

                    PATCH_SIZE = 14
                    b, v, H, W, _ = pts.shape
                    Ph, Pw = H // PATCH_SIZE, W // PATCH_SIZE
                    n = Ph * Pw  # spatial tokens per view

                    # pool to patch resolution: (b, v, Ph, Pw, 3)
                    pts_patch  = pts.reshape(b, v, Ph, PATCH_SIZE, Pw, PATCH_SIZE, 3).mean(dim=(3, 5))
                    # flatten all views' patches: (b, v*n, 3)
                    pts_flat   = pts_patch.reshape(b, v * n, 3)

                    # reorder view blocks to match pred_map (ref view first)
                    if lq_ref_b_idx is None:
                        lq_ref_b_idx = torch.tensor([0], device=device)  # default to first view as reference if not provided by encoder
                    ref_v      = int(lq_ref_b_idx[0].item())
                    view_order = [ref_v] + [vi for vi in range(v) if vi != ref_v]
                    perm       = torch.cat([torch.arange(vi * n, vi * n + n, device=pts_flat.device) for vi in view_order])
                    pts_flat   = pts_flat[:, perm, :]  # (b, v*n, 3) reordered

                    # pairwise neg L2: (b, v*n, v*n)
                    diff       = pts_flat.unsqueeze(1) - pts_flat.unsqueeze(2)  # (b, v*n, v*n, 3)
                    neg_l2     = -torch.norm(diff, dim=-1)                                    # (b, v*n, v*n)
                    T          = full_cfg.mvrm.loss.attn_align.da3_point_cloud.get('vis_pc_temperature', 1.0)
                    if T == -1:  # hard one-hot assignment (nearest neighbour per view block)
                        print('hard assignment for point cloud')
                        _blocked   = neg_l2.reshape(b, v * n, v, n)
                        geo_target = torch.zeros_like(_blocked).scatter_(-1, _blocked.argmax(dim=-1, keepdim=True), 1.0)
                        geo_target = geo_target.reshape(b, v * n, v * n)
                    else:
                        print('soft assignment for point cloud')
                        geo_target = (neg_l2 / T).softmax(dim=-1)                       # (b, v*n, v*n)
                    # --- Visibility mask ---
                    # Controls which (query patch, target view) pairs contribute to the loss.
                    # Unmasked geo_target rows that fail visibility are zeroed and renormalized.
                    pc_cfg       = full_cfg.mvrm.loss.attn_align.da3_point_cloud
                    vis_mask_type = pc_cfg.get('visibility_mask', 'none')  # [none, cycle_consistency, reprojection]

                    if vis_mask_type != 'none':
                        vis_mask = torch.zeros(b, v * n, v * n, dtype=torch.bool, device=device)
                        if vis_mask_type == 'cycle_consistency':
                            print(f'Using cycle consistency visibility mask - cycle threshold {pc_cfg.get("vis_pc_cycle_threshold", "None")}')
                            ref_idx = torch.arange(n, device=device).unsqueeze(0).expand(b, -1)  # (b, n)
                            for va in range(v):
                                for vb in range(v):
                                    va_s, va_e = va * n, (va + 1) * n
                                    vb_s, vb_e = vb * n, (vb + 1) * n
                                    if va == vb:
                                        vis_mask[:, va_s:va_e, vb_s:vb_e] = True
                                        continue
                                    corr_ab = geo_target[:, va_s:va_e, vb_s:vb_e]   # (b, n, n)
                                    corr_ba = geo_target[:, vb_s:vb_e, va_s:va_e]   # (b, n, n)
                                    fwd       = corr_ab.argmax(dim=-1)                 # (b, n)
                                    bwd       = corr_ba.argmax(dim=-1)                 # (b, n)
                                    roundtrip = torch.gather(bwd, 1, fwd)              # (b, n)
                                    cycle_thresh = pc_cfg.get('vis_pc_cycle_threshold', 0)
                                    if cycle_thresh == 0:
                                        visible = (roundtrip == ref_idx)
                                    else:
                                        rt_r,  rt_c  = roundtrip // Pw, roundtrip % Pw
                                        ref_r, ref_c = ref_idx   // Pw, ref_idx   % Pw
                                        distance    = torch.max((rt_r - ref_r).abs(), (rt_c - ref_c).abs())
                                        visible = distance <= cycle_thresh
                                    vis_mask[:, va_s:va_e, vb_s:vb_e] = visible.unsqueeze(-1).expand(-1, -1, n)
                        # Zero out non-visible correspondences and renormalize rows
                        geo_target = geo_target * vis_mask.float()
                        row_sum    = geo_target.sum(dim=-1, keepdim=True)          # (b, v*n, 1)
                        valid_rows = (row_sum.squeeze(-1) > 0)                     # (b, v*n) rows with mass
                        geo_target = geo_target / row_sum.clamp(min=1e-8)
                    # slice CLS tokens out of pred_map: positions 0, n+1, 2*(n+1), ...
                    N_total = pred_map.shape[-1]
                    spatial_mask = torch.ones(N_total, dtype=torch.bool, device=device)
                    for vi in range(v):
                        spatial_mask[vi * (n + 1)] = False
                    pred_map_spatial = pred_map[:, spatial_mask, :][:, :, spatial_mask]  # (1, v*n, v*n)
                    row_mask  = valid_rows if vis_mask_type != 'none' else None
                    attn_loss = cross_entropy_attn(pred_map_spatial, geo_target, row_mask=row_mask)
                    loss += lambda_attn * attn_loss
                    
                    
            # ---------------------------
            # Backward
            # ---------------------------
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    models['ddp_denoiser'].parameters(),
                    clip_grad
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            update_ema(
                models['ema_denoiser'],
                models['ddp_denoiser'].module,
                decay=ema_decay,
            )
            optimizer_step += 1
            
            
            # ---------------------------
            # Logging
            # ---------------------------
            running_loss += loss.item()
            epoch_metrics['loss'] += loss.detach()
            if rank == 0 and global_train_step % log_interval == 0:
                avg_loss = running_loss / log_interval
                stats = {
                    "train/loss_interval_avg": avg_loss,
                    "train/loss_transport": transport_loss.item(),
                    "train_attn/loss_attn": attn_loss.item(),
                    "train_etc/lq_drop_prob": full_cfg.training.guidance.lq_drop,
                    "train_etc/lr": optimizer.param_groups[0]["lr"],
                }
                logger.info(
                    f"[Epoch {epoch} | Step {global_train_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if full_cfg.log.tracker.name == 'wandb':
                    wandb_utils.log(stats, step=global_train_step)
                running_loss = 0.0

    
            # ckpt saving
            if rank==0 and global_train_step > 0 and global_train_step % ckpt_step_interval == 0:
                logger.info(f"Saving checkpoint at global_train_step {global_train_step}...")
                ckpt_path = f"{checkpoint_dir}/ep-{global_train_step:07d}.pt" 
                initialize.save_checkpoint(
                    ckpt_path,
                    global_train_step,
                    epoch,
                    models['ddp_denoiser'],
                    models['ema_denoiser'],
                    optimizer,
                    scheduler,
                )                        
            
            num_batches += 1
            global_train_step += 1
        
        
        # log epoch stats
        if rank == 0 and num_batches > 0:
            avg_loss = epoch_metrics['loss'].item() / num_batches 
            epoch_stats = {
                "epoch/loss": avg_loss,
            }
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )
            if full_cfg.log.tracker.name == 'wandb':
                wandb_utils.log(epoch_stats, step=global_train_step)
    
    
    # save the final ckpt
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-last.pt" 
        initialize.save_checkpoint(
            ckpt_path,
            global_train_step,
            num_epochs,
            models['ddp_denoiser'],
            models['ema_denoiser'],
            optimizer,
            scheduler,
        )
    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()



if __name__ == "__main__":
    main()
