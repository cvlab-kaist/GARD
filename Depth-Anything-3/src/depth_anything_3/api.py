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
Depth Anything 3 API module.

This module provides the main API for Depth Anything 3, including model loading,
inference, and export capabilities. It supports both single and nested model architectures.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from PIL import Image

from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export import export
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.io.output_processor import OutputProcessor
from depth_anything_3.utils.logger import logger
from depth_anything_3.utils.pose_align import align_poses_umeyama

from torchvision.utils import save_image 
import torchvision.transforms.functional as TF

# from RAE.src import analysis
from mvr.utils.featsim_utils import *
from mvr.utils.metric_utils import *
from mvr.utils.freq_utils import *

from RAE.src.utils.vis_utils import vis_all, vis_attn_maps, vis_pointcloud_correspondence_maps, vis_pointcloud_cycle_correspondence_maps, vis_pointcloud_reproj_correspondence_maps
from RAE.src.vis_cam_pose import plot_cam_trajectory, plot_cam_trajectory_fair, plot_all_cam_trajectory_fair
from depth_anything_3.utils.geometry import unproject_depth, affine_inverse, as_homogeneous
from depth_anything_3.utils.io.mvrm_rgb_frame_saver import save_mvrm_decoder_rgb_frames


torch.backends.cudnn.benchmark = False
# logger.info("CUDNN Benchmark Disabled")

SAFETENSORS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


def _build_reproj_vis_mask(pts_patch, depth_t, w2c, K, b, v_pts, Ph, Pw, n_pts, PATCH_SIZE, depth_thr=0.1):
    """
    Reprojection-based visibility mask.
    For each patch i in view va, project its 3D point into view vb.
    A patch is "visible" in vb if:
      (1) the projected pixel is within image bounds, and
      (2) the projected depth matches the actual depth at that patch location
          within a relative threshold.

    Args:
        pts_patch : (b, v, Ph, Pw, 3)  patch-level 3D world points
        depth_t   : (b, v, H, W, 1)   full-resolution depth (original view order)
        w2c       : (b, v, 4, 4)       world-to-camera transforms
        K         : (b, v, 3, 3)       camera intrinsics (pixel space)
        depth_thr : relative depth error threshold

    Returns:
        reproj_mask : (b, v*n, v*n) bool  — True where correspondence is valid
    """
    depth_patch = depth_t.squeeze(-1).reshape(
        b, v_pts, Ph, PATCH_SIZE, Pw, PATCH_SIZE).mean(dim=(3, 5))     # (b, v, Ph, Pw)
    pts_v       = pts_patch.reshape(b, v_pts, n_pts, 3)                # (b, v, n, 3)
    reproj_mask = torch.zeros(b, v_pts * n_pts, v_pts * n_pts,
                              dtype=torch.bool, device=pts_patch.device)

    for va in range(v_pts):
        for vb in range(v_pts):
            va_s, va_e = va * n_pts, (va + 1) * n_pts
            vb_s, vb_e = vb * n_pts, (vb + 1) * n_pts
            if va == vb:
                reproj_mask[:, va_s:va_e, vb_s:vb_e] = True
                continue

            P_world = pts_v[:, va]                                      # (b, n, 3)
            R_vb    = w2c[:, vb, :3, :3]                               # (b, 3, 3)
            t_vb    = w2c[:, vb, :3, 3]                                # (b, 3)
            P_cam   = torch.bmm(P_world, R_vb.transpose(1, 2)) + t_vb.unsqueeze(1)  # (b, n, 3)
            z_proj  = P_cam[:, :, 2]                                    # (b, n)

            fx = K[:, vb, 0, 0].unsqueeze(1)
            fy = K[:, vb, 1, 1].unsqueeze(1)
            cx = K[:, vb, 0, 2].unsqueeze(1)
            cy = K[:, vb, 1, 2].unsqueeze(1)
            u_px   = fx * P_cam[:, :, 0] / z_proj.clamp(min=1e-8) + cx  # (b, n)
            v_px   = fy * P_cam[:, :, 1] / z_proj.clamp(min=1e-8) + cy
            pi_col = (u_px / PATCH_SIZE).long()
            pi_row = (v_px / PATCH_SIZE).long()

            in_bounds = (z_proj > 0) & (pi_row >= 0) & (pi_row < Ph) & (pi_col >= 0) & (pi_col < Pw)

            pi_row_c      = pi_row.clamp(0, Ph - 1)
            pi_col_c      = pi_col.clamp(0, Pw - 1)
            flat_idx      = pi_row_c * Pw + pi_col_c                   # (b, n)
            depth_vb_flat = depth_patch[:, vb].reshape(b, -1)          # (b, n)
            depth_at_proj = torch.gather(depth_vb_flat, 1, flat_idx)   # (b, n)
            rel_err       = (z_proj - depth_at_proj).abs() / (depth_at_proj.abs() + 1e-8)
            depth_ok      = rel_err < depth_thr

            visible = in_bounds & depth_ok
            reproj_mask[:, va_s:va_e, vb_s:vb_e] = visible.unsqueeze(-1).expand(-1, -1, n_pts)

    return reproj_mask


class DepthAnything3(nn.Module, PyTorchModelHubMixin):
    """
    Depth Anything 3 main API class.

    This class provides a high-level interface for depth estimation using Depth Anything 3.
    It supports both single and nested model architectures with metric scaling capabilities.

    Features:
    - Hugging Face Hub integration via PyTorchModelHubMixin
    - Support for multiple model presets (vitb, vitg, nested variants)
    - Automatic mixed precision inference
    - Export capabilities for various formats (GLB, PLY, NPZ, etc.)
    - Camera pose estimation and metric depth scaling

    Usage:
        # Load from Hugging Face Hub
        model = DepthAnything3.from_pretrained("huggingface/model-name")

        # Or create with specific preset
        model = DepthAnything3(preset="vitg")

        # Run inference
        prediction = model.inference(images, export_dir="output", export_format="glb")
    """

    _commit_hash: str | None = None  # Set by mixin when loading from Hub

    def __init__(self, model_name: str = "da3-large", **kwargs):
        """
        Initialize DepthAnything3 with specified preset.

        Args:
        model_name: The name of the model preset to use.
                    Examples: 'da3-giant', 'da3-large', 'da3metric-large', 'da3nested-giant-large'.
        **kwargs: Additional keyword arguments (currently unused).
        """
        super().__init__()
        self.model_name = model_name

        # Build the underlying network
        self.config = load_config(MODEL_REGISTRY[self.model_name])
        self.model = create_object(self.config)     
        self.model.eval()

        # Initialize processors
        self.input_processor = InputProcessor()
        self.output_processor = OutputProcessor()

        # Device management (set by user)
        self.device = None

    @torch.inference_mode()
    def forward(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        mvrm_cfg=None,
        mvrm_result=None,
        mode=None,
        ref_b_idx=None,
        front_connect_back_mvrm_cfg=None,
        analysis=None,
        export_rgb_feat_layers=False
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            image: Input batch with shape ``(B, N, 3, H, W)`` on the model device.
            extrinsics: Optional camera extrinsics with shape ``(B, N, 4, 4)``.
            intrinsics: Optional camera intrinsics with shape ``(B, N, 3, 3)``.
            export_feat_layers: Layer indices to return intermediate features for.
            infer_gs: Enable Gaussian Splatting branch.
            use_ray_pose: Use ray-based pose estimation instead of camera decoder.
            ref_view_strategy: Strategy for selecting reference view from multiple views.

        Returns:
            Dictionary containing model predictions
        """
        # Determine optimal autocast dtype
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad():
            with torch.autocast(device_type=image.device.type, dtype=autocast_dtype):
                return self.model(
                    image, extrinsics, intrinsics, export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy, mvrm_cfg, mvrm_result, mode, ref_b_idx, front_connect_back_mvrm_cfg, analysis, export_rgb_feat_layers
                    )

    def inference(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        align_to_input_ext_scale: bool = True,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        render_exts: np.ndarray | None = None,
        render_ixts: np.ndarray | None = None,
        render_hw: tuple[int, int] | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_dir: str | None = None,
        export_format: str = "mini_npz",
        export_feat_layers: Sequence[int] | None = None,
        # GLB export parameters
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        # Feat_vis export parameters
        feat_vis_fps: int = 15,
        # Other export parameters, e.g., gs_ply, gs_video
        export_kwargs: Optional[dict] = {},
        eval_sampler=None,
        denoiser=None,
        denoiser2=None,
        noise_generator=None,
        cfg=None,
        scene_info=None,
        use_pose=None,
        rgb_decoder = None,
        proj_adapter = None,
        device=None,
        has_gt: bool = True,
    ) -> Prediction:
        """
        Run inference on input images.

        Args:
            image: List of input images (numpy arrays, PIL Images, or file paths)
            extrinsics: Camera extrinsics (N, 4, 4)
            intrinsics: Camera intrinsics (N, 3, 3)
            align_to_input_ext_scale: whether to align the input pose scale to the prediction
            infer_gs: Enable the 3D Gaussian branch (needed for `gs_ply`/`gs_video` exports)
            use_ray_pose: Use ray-based pose estimation instead of camera decoder (default: False)
            ref_view_strategy: Strategy for selecting reference view from multiple views.
                Options: "first", "middle", "saddle_balanced", "saddle_sim_range".
                Default: "saddle_balanced". For single view input (S ≤ 2), no reordering is performed.
            render_exts: Optional render extrinsics for Gaussian video export
            render_ixts: Optional render intrinsics for Gaussian video export
            render_hw: Optional render resolution for Gaussian video export
            process_res: Processing resolution
            process_res_method: Resize method for processing
            export_dir: Directory to export results
            export_format: Export format (mini_npz, npz, glb, ply, gs, gs_video)
            export_feat_layers: Layer indices to export intermediate features from
            conf_thresh_percentile: [GLB] Lower percentile for adaptive confidence threshold (default: 40.0) # noqa: E501
            num_max_points: [GLB] Maximum number of points in the point cloud (default: 1,000,000)
            show_cameras: [GLB] Show camera wireframes in the exported scene (default: True)
            feat_vis_fps: [FEAT_VIS] Frame rate for output video (default: 15)
            export_kwargs: additional arguments to export functions.
            has_gt: Whether `image` carries a real clean/GT reference (as opposed to
                the LQ images echoed as a placeholder for GT-free benches). When
                False, the GT-comparison exports (pose_depth_metric_results,
                featsim_results, cam_traj_results) are skipped since they would be
                comparing the restoration against a meaningless placeholder.

        Returns:
            Prediction object containing depth maps and camera parameters
        """
        
        data, scene = scene_info 
        scene = scene.replace('/', '_') if '/' in scene else scene
        pose_setting = 'pose' if use_pose else 'unposed'
        
        
        if "gs" in export_format:
            assert infer_gs, "must set `infer_gs=True` to perform gs-related export."

        if "colmap" in export_format:
            assert isinstance(image.image_files[0], str), "`image` must be image paths for COLMAP export."


        # lq
        if 'lq_image_files' in image.keys() and cfg.MVRM_EVAL.load_lq:
            lq_imgs_cpu, _, _ = self._preprocess_inputs(
                image.lq_image_files, None, None, process_res, process_res_method
            )
            lq_imgs, _, _ = self._prepare_model_inputs(lq_imgs_cpu, None, None)

        
        # Preprocess hq images
        imgs_cpu, extrinsics, intrinsics = self._preprocess_inputs(
            image.image_files, extrinsics, intrinsics, process_res, process_res_method
        )
        # Prepare tensors for model
        imgs, ex_t, in_t = self._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)
        

        # Normalize extrinsics
        ex_t_norm = self._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)

        # Run model forward pass
        export_feat_layers = list(export_feat_layers) if export_feat_layers is not None else []

        # image input info
        b, v, c, model_H, model_W = imgs.shape

        export_feat_layers=[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,39]    


        # Populated only in the w_mvrm branch when the RGB decoder reconstructs the
        # restored output (rgb_res_dn below); used to export the actual restored
        # image instead of the raw input echo (see override near _export_results).
        restored_rgb_for_npz = None

        # Apply W_MVRM restoration
        if cfg.MVRM_EVAL.eval_method == 'w_mvrm':
            print('-'*70)      
            print('APPLYING MVRM O')
            print('-'*70)
            

            to_vis_imgs_list = []
            to_vis_pc_list = []       # (name, pts)  pts: (b, v, H, W, 3) tensor
            to_vis_pc_maps_list = []  # (name, pc_maps)  pc_maps: {('pc', 0, 'global'): (1, v*n, v*n)}
            to_vis_pc_maps_cycle_list  = []  # (name, pc_maps)  pc_maps: {('pc', 0, 'global_cycle'):  (1, v*n, v*n)}
            to_vis_pc_maps_reproj_list = []  # (name, pc_maps)  pc_maps: {('pc', 0, 'global_reproj'): (1, v*n, v*n)}
            
            

            # (W_MVRM) LQ FORWARD PASS
            print("LQ FORWARD PASS")
            lq_encoder_out, lq_mvrm_out = self._run_model_forward(
                                            lq_imgs, 
                                            ex_t_norm, 
                                            in_t, 
                                            export_feat_layers, 
                                            infer_gs, 
                                            use_ray_pose, 
                                            ref_view_strategy, 
                                            mvrm_cfg=cfg.mvrm.train, 
                                            mvrm_result=None, 
                                            mode='train',
                                            ref_b_idx=None,
                                            analysis = None,
                                            export_rgb_feat_layers=True
                                        )
            # lq_pred_pose_enc = lq_encoder_out.pose_enc      # 1 v 9
            lq_pred_pose = lq_encoder_out['extrinsics']     # 1 v 3 4
            lq_pred_intrinsics = lq_encoder_out['intrinsics']  # (b, v, 3, 3)
            lq_ref_b_idx = lq_encoder_out.ref_b_idx
            # safety check
            for key in lq_mvrm_out.keys():
                assert key[-1] in cfg.mvrm.train.extract_feat_layers, f"Extracted MVRM feature layer {key[-1]} not in config extract_feat_layers {cfg.mvrm.train.extract_feat_layers}"
            first_extract_layer_idx = cfg.mvrm.train.extract_feat_layers[0]
            lq_latent = lq_mvrm_out[('extract_feat', first_extract_layer_idx)]         # b v n+1 d
            lq_depth = lq_encoder_out.depth.unsqueeze(2)     # 1 v 1 h w


            if rgb_decoder is not None:
                mae_feats = []
                for (patches, cls_token) in lq_encoder_out.feat:
                    # patches: (B, V, N, C)
                    mae_feats.append(patches)       # b v n d  (1 10 972 1536)
                # cat dim=-1 => (B, V, N, C*4)
                z_cat = torch.cat(mae_feats, dim=-1)
                # Reshape to (B*V, N, C_total)
                b, v, n, c_tot = z_cat.shape
                z_cat = z_cat.reshape(b*v, n, c_tot)         
                       
                # Run MAE decoder
                with torch.no_grad():
                    with torch.autocast(device_type=z_cat.device.type, enabled=True, dtype=torch.bfloat16):
                        # MAE decoder forward
                        # forward(hidden_states, input_size, drop_cls_token=False)
                        if proj_adapter is not None:
                            z_cat = proj_adapter(z_cat)
                        mae_out_logits = rgb_decoder(z_cat, input_size=(model_H, model_W), drop_cls_token=False).logits
                        # Unpatchify
                        x_rec = rgb_decoder.unpatchify(mae_out_logits, (model_H, model_W)) # (B*V, 3, H, W)
                        # Reshape to (B, V, 3, H, W) to match DPT format for consistency in denorm block below
                        x_rec = x_rec.reshape(b, v, 3, model_H, model_W)
                        rgb_lq = x_rec
                    
            

            # (W_MVRM) HQ FORWARD PASS
            # Only needed for the GT-comparison outputs below (vis panels included) —
            # for GT-free benches `imgs` is just the LQ images echoed as a placeholder
            # (see callers), so this pass would compare the restoration against itself
            # and isn't run at all.
            hq_pred_pose = hq_pred_intrinsics = hq_latent = hq_depth = None
            hq_encoder_out = None
            rgb_hq = None
            if has_gt:
                print("HQ FORWARD PASS")
                hq_encoder_out, hq_mvrm_out = self._run_model_forward(
                                                imgs,
                                                ex_t_norm,
                                                in_t,
                                                export_feat_layers,
                                                infer_gs,
                                                use_ray_pose,
                                                ref_view_strategy,
                                                mvrm_cfg=cfg.mvrm.train,
                                                mvrm_result=None,
                                                mode='train',
                                                ref_b_idx=lq_ref_b_idx,
                                                analysis = None,
                                                export_rgb_feat_layers=True
                                            )
                hq_pred_pose = hq_encoder_out['extrinsics']     # 1 v 3 4
                hq_pred_intrinsics = hq_encoder_out['intrinsics']  # (b, v, 3, 3)
                hq_latent = hq_mvrm_out[('extract_feat', cfg.mvrm.train.extract_feat_layers[0])]         # b v n+1 d
                hq_depth = hq_encoder_out.depth.unsqueeze(2)    # 1 v 1 h w


                if rgb_decoder is not None:
                    mae_feats = []
                    for (patches, cls_token) in hq_encoder_out.feat:
                        # patches: (B, V, N, C)
                        mae_feats.append(patches)       # b v n d  (1 10 972 1536)
                    # cat dim=-1 => (B, V, N, C*4)
                    z_cat = torch.cat(mae_feats, dim=-1)
                    # Reshape to (B*V, N, C_total)
                    b, v, n, c_tot = z_cat.shape
                    z_cat = z_cat.reshape(b*v, n, c_tot)

                    # Run MAE decoder
                    with torch.no_grad():
                        with torch.autocast(device_type=z_cat.device.type, enabled=True, dtype=torch.bfloat16):
                            # MAE decoder forward
                            # forward(hidden_states, input_size, drop_cls_token=False)
                            if proj_adapter is not None:
                                z_cat = proj_adapter(z_cat)
                            mae_out_logits = rgb_decoder(z_cat, input_size=(model_H, model_W), drop_cls_token=False).logits
                            # Unpatchify
                            x_rec = rgb_decoder.unpatchify(mae_out_logits, (model_H, model_W)) # (B*V, 3, H, W)
                            # Reshape to (B, V, 3, H, W) to match DPT format for consistency in denorm block below
                            x_rec = x_rec.reshape(b, v, 3, model_H, model_W)
                            rgb_hq = x_rec



            # generate pure noise
            noise_generator.manual_seed(42)
            pure_noise = torch.randn(lq_latent.shape, generator=noise_generator, device=imgs.device, dtype=torch.float32)
            
            

            noise_lvl = cfg.mvrm.get('noise_lvl', None)    
            if noise_lvl is not None:
                print('Using LQ2HQ wnoise: ', noise_lvl)
                x0 = pure_noise * noise_lvl + lq_latent
            else:
                x0 = pure_noise



            guidance = cfg.mvrm.val.get('guidance', None)
            if guidance.use_cfg and guidance.cfg_scale > 1.0:
                print('Using CFG sampling with scale: ', guidance.cfg_scale)
            else:
                print('Using non-CFG sampling')
            

            
            
            lq_cond_sampling = cfg.mvrm.val.get('lq_cond_sampling', True)
            if lq_cond_sampling:          
                print('COND sampling !!')
            else:
                print('UNCOND sampling !!')

                
                            
            model_kwargs={
                'mvrm_cfg': cfg.mvrm, 
                'model_img_size': (model_H, model_W),
                'lq_latent': lq_latent         
            }

            
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.no_grad():
                with torch.autocast(device_type=imgs.device.type, dtype=autocast_dtype):         
                    restored_samples = eval_sampler(x0, denoiser.forward, **model_kwargs)[-1]     # b v n d # eval_sampler: class ode def sample() function
            

            mvrm_result={}
            mvrm_result[('restored_latent', first_extract_layer_idx)] = restored_samples

            
            # RES FORWARD PASS
            print("RES FORWARD PASS")
            raw_output, _ = self._run_model_forward(
                                            lq_imgs, 
                                            ex_t_norm, 
                                            in_t, 
                                            export_feat_layers, 
                                            infer_gs, 
                                            use_ray_pose, 
                                            ref_view_strategy, 
                                            mvrm_cfg=cfg.mvrm.val, 
                                            mvrm_result=mvrm_result, 
                                            mode='val',
                                            ref_b_idx=lq_ref_b_idx,
                                            analysis = None,
                                            export_rgb_feat_layers=True
                                        )
            res_pred_pose_enc = raw_output.pose_enc      # 1 v 9
            res_pred_pose = raw_output['extrinsics']     # 1 v 3 4
            res_pred_intrinsics = raw_output['intrinsics']  # (b, v, 3, 3)
            res_depth = raw_output.depth.unsqueeze(2)    # 1 v 1 h w
                        
            
            
            if rgb_decoder is not None:
                mae_feats = []
                for (patches, cls_token) in raw_output.feat:
                    # patches: (B, V, N, C)
                    mae_feats.append(patches)       # b v n d  (1 10 972 1536)
                # cat dim=-1 => (B, V, N, C*4)
                z_cat = torch.cat(mae_feats, dim=-1)
                # Reshape to (B*V, N, C_total)
                b, v, n, c_tot = z_cat.shape
                z_cat = z_cat.reshape(b*v, n, c_tot)         
                       
                # Run MAE decoder
                with torch.no_grad():
                    with torch.autocast(device_type=z_cat.device.type, enabled=True, dtype=torch.bfloat16):
                        # MAE decoder forward
                        # forward(hidden_states, input_size, drop_cls_token=False)
                        if proj_adapter is not None:
                            z_cat = proj_adapter(z_cat)
                        mae_out_logits = rgb_decoder(z_cat, input_size=(model_H, model_W), drop_cls_token=False).logits
                        # Unpatchify
                        x_rec = rgb_decoder.unpatchify(mae_out_logits, (model_H, model_W)) # (B*V, 3, H, W)
                        # Reshape to (B, V, 3, H, W) to match DPT format for consistency in denorm block below
                        x_rec = x_rec.reshape(b, v, 3, model_H, model_W)
                        rgb_res = x_rec


            # visualize restored images
            vis_rgb_recon_root = os.path.join(cfg.workspace.work_dir, 'vis_rgb_restored_results',  data, pose_setting)
            os.makedirs(vis_rgb_recon_root, exist_ok=True)
            mean = torch.tensor([0.485, 0.456, 0.406], device=rgb_lq.device, dtype=rgb_lq.dtype).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=rgb_lq.device, dtype=rgb_lq.dtype).view(1, 3, 1, 1)
            def denorm(x):
                return (x * std + mean).clamp(0, 1)
            rgb_lq_dn  = denorm(rgb_lq.squeeze(0).float())
            rgb_res_dn = denorm(rgb_res.squeeze(0).float())
            restored_rgb_for_npz = rgb_res_dn  # (v, 3, H, W) float in [0,1]
            # Each row: all 10 frames concatenated horizontally → [3, 378*v, 504]
            row_lq  = torch.cat([rgb_lq_dn[i]  for i in range(len(rgb_lq_dn))], dim=-2)
            row_res = torch.cat([rgb_res_dn[i] for i in range(len(rgb_res_dn))], dim=-2)
            # Stack rows vertically. HQ row is only meaningful (and only computed
            # above) when this scene actually has GT.
            rows = [row_lq, row_res]
            if has_gt:
                rgb_hq_dn = denorm(rgb_hq.squeeze(0).float())   # [v, 3, 378, 504]
                row_hq = torch.cat([rgb_hq_dn[i] for i in range(len(rgb_hq_dn))], dim=-2)
                rows = [row_hq] + rows
            combined = torch.cat(rows, dim=-1)
            img = TF.to_pil_image(combined.cpu())
            img.save(os.path.join(vis_rgb_recon_root, f'{scene}.png'))


            vis_save_root = os.path.join(cfg.workspace.work_dir, 'vis_depth_results', data, pose_setting)
            vis_all(
                vis_save_root=vis_save_root,
                scene=scene,
                hq_img=imgs[0] if has_gt else None,
                lq_img=lq_imgs[0],
                hq_depth=hq_depth[0] if has_gt else None,
                lq_depth=lq_depth[0],
                res_depth=res_depth[0],
            )
            # These all compare against `hq_*`, which for GT-free benches is just the
            # LQ images echoed as a placeholder (see callers) — skip them there since
            # the comparison would be meaningless.
            if has_gt:
                metric_save_root = os.path.join(cfg.workspace.work_dir, 'pose_depth_metric_results', data, pose_setting)
                metric_all(
                    metric_save_root=metric_save_root,
                    scene=scene,
                    poses = (hq_pred_pose[0], lq_pred_pose[0],res_pred_pose[0]),
                    depths = (hq_depth, lq_depth, res_depth)
                )
                featsim_log = featsim_all(hq_encoder_out, lq_encoder_out, raw_output)
                featsim_save_root = os.path.join(cfg.workspace.work_dir, 'featsim_results', data, pose_setting)
                plot_three_similarity_panels(
                    featsim_log,
                    save_path=f"{featsim_save_root}/{scene}_sim_all_combined.png"
                )
                cam_save_root = os.path.join(cfg.workspace.work_dir, 'cam_traj_results', data, pose_setting)
                plot_cam_trajectory_fair(hq_pred_pose[0], lq_pred_pose[0], res_pred_pose[0], visualize_direction=False, save_path=f"{cam_save_root}/fair_{scene}.png")

            
        # Convert raw output to prediction
        prediction = self._convert_to_prediction(raw_output)

        # Align prediction to extrinsincs
        prediction = self._align_to_input_extrinsics_intrinsics(
            extrinsics, intrinsics, prediction, align_to_input_ext_scale
        )

        # Add processed images for visualization
        prediction = self._add_processed_images(prediction, imgs_cpu)   # imagenet denormalization, and convert to uint8 [0,255] numpy

        # Export if requested
        if export_dir is not None:

            if "gs" in export_format:
                if infer_gs and "gs_video" not in export_format:
                    export_format = f"{export_format}-gs_video"
                if "gs_video" in export_format:
                    if "gs_video" not in export_kwargs:
                        export_kwargs["gs_video"] = {}
                    export_kwargs["gs_video"].update(
                        {
                            "extrinsics": render_exts,
                            "intrinsics": render_ixts,
                            "out_image_hw": render_hw,
                        }
                    )
            # Add GLB export parameters
            if "glb" in export_format:
                if "glb" not in export_kwargs:
                    export_kwargs["glb"] = {}
                export_kwargs["glb"].update(
                    {
                        "conf_thresh_percentile": conf_thresh_percentile,
                        "num_max_points": num_max_points,
                        "show_cameras": show_cameras,
                    }
                )
            # Add Feat_vis export parameters
            if "feat_vis" in export_format:
                if "feat_vis" not in export_kwargs:
                    export_kwargs["feat_vis"] = {}
                export_kwargs["feat_vis"].update(
                    {
                        "fps": feat_vis_fps,
                    }
                )
            # Add COLMAP export parameters
            if "colmap" in export_format:
                if "colmap" not in export_kwargs:
                    export_kwargs["colmap"] = {}
                export_kwargs["colmap"].update(
                    {
                        "image_paths": image,
                        "conf_thresh_percentile": conf_thresh_percentile,
                        "process_res_method": process_res_method,
                    }
                )
            # For npz export, use the actual restored RGB reconstruction (when the
            # w_mvrm branch computed one) instead of the raw input echo left by
            # _add_processed_images, so deblur-bench PSNR/SSIM/LPIPS reflect the
            # model's real output rather than the (blurry-input-independent) GT copy.
            if restored_rgb_for_npz is not None and "npz" in export_format:
                prediction.processed_images = (
                    restored_rgb_for_npz.clamp(0, 1).permute(0, 2, 3, 1) * 255
                ).round().to(torch.uint8).cpu().numpy()

            # export da3 predictions
            self._export_results(prediction, export_format, export_dir, **export_kwargs)

        return prediction


    def _preprocess_inputs(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Preprocess input images using input processor."""
        start_time = time.time()
        imgs_cpu, extrinsics, intrinsics = self.input_processor(
            image,
            extrinsics.copy() if extrinsics is not None else None,
            intrinsics.copy() if intrinsics is not None else None,
            process_res,
            process_res_method,
        )
        end_time = time.time()
        logger.info(
            "Processed Images Done taking",
            end_time - start_time,
            "seconds. Shape: ",
            imgs_cpu.shape,
        )
        return imgs_cpu, extrinsics, intrinsics

    def _prepare_model_inputs(
        self,
        imgs_cpu: torch.Tensor,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Prepare tensors for model input."""
        device = self._get_model_device()

        # Move images to model device
        imgs = imgs_cpu.to(device, non_blocking=True)[None].float()

        # Convert camera parameters to tensors
        ex_t = (
            extrinsics.to(device, non_blocking=True)[None].float()
            if extrinsics is not None
            else None
        )
        in_t = (
            intrinsics.to(device, non_blocking=True)[None].float()
            if intrinsics is not None
            else None
        )

        return imgs, ex_t, in_t

    def _normalize_extrinsics(self, ex_t: torch.Tensor | None) -> torch.Tensor | None:
        """Normalize extrinsics"""
        if ex_t is None:
            return None
        transform = affine_inverse(ex_t[:, :1])
        ex_t_norm = ex_t @ transform
        c2ws = affine_inverse(ex_t_norm)
        translations = c2ws[..., :3, 3]
        dists = translations.norm(dim=-1)
        median_dist = torch.median(dists)
        median_dist = torch.clamp(median_dist, min=1e-1)
        ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
        return ex_t_norm

    def _align_to_input_extrinsics_intrinsics(
        self,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        prediction: Prediction,
        align_to_input_ext_scale: bool = True,
        ransac_view_thresh: int = 10,
    ) -> Prediction:
        # breakpoint()
        """Align depth map to input extrinsics"""
        if extrinsics is None:
            return prediction
        prediction.intrinsics = intrinsics.numpy()
        _, _, scale, aligned_extrinsics = align_poses_umeyama(
            prediction.extrinsics,
            extrinsics.numpy(),
            ransac=len(extrinsics) >= ransac_view_thresh,
            return_aligned=True,
            random_state=42,
        )
        if align_to_input_ext_scale:
            prediction.extrinsics = extrinsics[..., :3, :].numpy()
            prediction.depth /= scale
        else:
            prediction.extrinsics = aligned_extrinsics
        return prediction

    def _run_model_forward(
        self,
        imgs: torch.Tensor,
        ex_t: torch.Tensor | None,
        in_t: torch.Tensor | None,
        export_feat_layers: Sequence[int] | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        mvrm_cfg=None,
        mvrm_result=None,
        mode=None,
        ref_b_idx=None,
        front_connect_back_mvrm_cfg=None,
        analysis=None,
        export_rgb_feat_layers=False
    ) -> dict[str, torch.Tensor]:
        """Run model forward pass."""
        device = imgs.device
        need_sync = device.type == "cuda"
        
        
        need_sync=False
        
        if need_sync:
            torch.cuda.synchronize(device)
        start_time = time.time()
        feat_layers = list(export_feat_layers) if export_feat_layers is not None else None
        output, mvrm_out = self.forward(imgs, ex_t, in_t, feat_layers, infer_gs, use_ray_pose, ref_view_strategy, mvrm_cfg, mvrm_result, mode, ref_b_idx, front_connect_back_mvrm_cfg, analysis, export_rgb_feat_layers)
        if need_sync:
            torch.cuda.synchronize(device)
        end_time = time.time()
        logger.info(f"Model Forward Pass Done. Time: {end_time - start_time} seconds")
        return output, mvrm_out

    def _convert_to_prediction(self, raw_output: dict[str, torch.Tensor]) -> Prediction:
        """Convert raw model output to Prediction object."""
        start_time = time.time()
        output = self.output_processor(raw_output)
        end_time = time.time()
        logger.info(f"Conversion to Prediction Done. Time: {end_time - start_time} seconds")
        return output

    def _add_processed_images(self, prediction: Prediction, imgs_cpu: torch.Tensor) -> Prediction:
        """Add processed images to prediction for visualization."""
        # Convert from (N, 3, H, W) to (N, H, W, 3) and denormalize
        processed_imgs = imgs_cpu.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, 3)

        # Denormalize from ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        processed_imgs = processed_imgs * std + mean
        processed_imgs = np.clip(processed_imgs, 0, 1)
        processed_imgs = (processed_imgs * 255).astype(np.uint8)

        prediction.processed_images = processed_imgs
        return prediction

    def _export_results(
        self, prediction: Prediction, export_format: str, export_dir: str, **kwargs
    ) -> None:
        """Export results to specified format and directory."""
        start_time = time.time()
        export(prediction, export_format, export_dir, **kwargs)
        end_time = time.time()
        logger.info(f"Export Results Done. Time: {end_time - start_time} seconds")

    def _get_model_device(self) -> torch.device:
        """
        Get the device where the model is located.

        Returns:
            Device where the model parameters are located

        Raises:
            ValueError: If no tensors are found in the model
        """
        if self.device is not None:
            return self.device

        # Find device from parameters
        for param in self.parameters():
            self.device = param.device
            return param.device

        # Find device from buffers
        for buffer in self.buffers():
            self.device = buffer.device
            return buffer.device

        raise ValueError("No tensor found in model")
