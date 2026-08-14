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
from mvr.utils.pca_utils import *
from mvr.utils.costvolume_utils import *

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


class VGGT(nn.Module, PyTorchModelHubMixin):
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

    def __init__(self, model_name: str = "vggt", **kwargs):
        """
        Initialize DepthAnything3 with specified preset.

        Args:
        model_name: The name of the model preset to use.
                    Examples: 'da3-giant', 'da3-large', 'da3metric-large', 'da3nested-giant-large'.
        **kwargs: Additional keyword arguments (currently unused).
        """
        super().__init__()
                
        from vggt.models.vggt import VGGT
        


        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Initialize the model and load the pretrained weights.
        # This will automatically download the model weights the first time it's run, which may take a while.
        self.model = VGGT.from_pretrained("facebook/VGGT-1B")

        # # Load and preprocess example images (replace with your own image paths)
        # image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]  
        # from vggt.utils.load_fn import load_and_preprocess_images
        # # images = load_and_preprocess_images(image_names).to(device)

        # with torch.no_grad():
        #     with torch.cuda.amp.autocast(dtype=dtype):
        #         # Predict attributes including cameras, depth maps, and point maps.
        #         predictions = model(images)
        
        
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
                predictions = self.model(
                    images = image, 
                    export_feat_layers = export_feat_layers
                    )
                return predictions 

                # return self.model(
                #     image, extrinsics, intrinsics, export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy, mvrm_cfg, mvrm_result, mode, ref_b_idx, front_connect_back_mvrm_cfg, analysis, export_rgb_feat_layers
                #     )

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
        device=None
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

        Returns:
            Prediction object containing depth maps and camera parameters
        """
        
        
        # PHO
        data, scene = scene_info 
        scene = scene.replace('/', '_') if '/' in scene else scene
        pose_setting = 'pose' if use_pose else 'unposed'
        
        
        if "gs" in export_format:
            assert infer_gs, "must set `infer_gs=True` to perform gs-related export."

        if "colmap" in export_format:
            assert isinstance(image.image_files[0], str), "`image` must be image paths for COLMAP export."


        # PHO
        if 'lq_image_files' in image.keys() and cfg.MVRM_EVAL.load_lq:
            lq_imgs_cpu, _, _ = self._preprocess_inputs(
                image.lq_image_files, None, None, process_res, process_res_method
            )
            lq_imgs, _, _ = self._prepare_model_inputs(lq_imgs_cpu, None, None)


        # PHO
        # breakpoint()
        if 'res_image_files' in image.keys() and cfg.MVRM_EVAL.load_res:
            res_imgs_cpu, _, _ = self._preprocess_inputs(
                image.res_image_files, None, None, process_res, process_res_method
            )
            res_imgs, _, _ = self._prepare_model_inputs(res_imgs_cpu, None, None)
            
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
        



        # VGGT INPUT IMAGE PROCESSING 
        from vggt.utils.load_fn import load_and_preprocess_images
        vggt_imgs = load_and_preprocess_images(image.image_files).to(device)
        vggt_res_imgs = load_and_preprocess_images(image.res_image_files).to(device)
        vggt_lq_imgs = load_and_preprocess_images(image.lq_image_files).to(device)
        
        


        if 'DA3' in cfg.model.path:
            if 'BASE' in cfg.model.path:
                export_feat_layers=[4, 6, 8, 10]    
            elif 'GIANT' in cfg.model.path:
                # export_feat_layers=[18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 39]    
                export_feat_layers=[14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 39]    
        else:
            export_feat_layers=[9, 11, 13, 15, 17, 19, 21, 23] 
    
    

        
        if rgb_decoder is None:    
            vis_rgb_LQ = False
            vis_rgb_HQ = False
            vis_rgb_RES = False
            vis_rgb_framewise = False
        else:
            vis_rgb_LQ = cfg.MVRM_EVAL.rgb_decoder.vis_rgb_LQ
            vis_rgb_HQ = cfg.MVRM_EVAL.rgb_decoder.vis_rgb_HQ
            vis_rgb_RES = cfg.MVRM_EVAL.rgb_decoder.vis_rgb_RES
            vis_rgb_framewise = cfg.MVRM_EVAL.rgb_decoder.vis_rgb_framewise
            
        
        
        
        

        # Apply W_MVRM restoration
        if cfg.MVRM_EVAL.eval_method == 'w_mvrm':       
            print('-'*70)      
            print('APPLYING MVRM O')
            print('-'*70)
            

            to_vis_imgs_list = []
            to_vis_attn_maps_list = []
            to_vis_pc_list = []       # (name, pts)  pts: (b, v, H, W, 3) tensor
            to_vis_pc_maps_list = []  # (name, pc_maps)  pc_maps: {('pc', 0, 'global'): (1, v*n, v*n)}
            to_vis_pc_maps_cycle_list  = []  # (name, pc_maps)  pc_maps: {('pc', 0, 'global_cycle'):  (1, v*n, v*n)}
            to_vis_pc_maps_reproj_list = []  # (name, pc_maps)  pc_maps: {('pc', 0, 'global_reproj'): (1, v*n, v*n)}
            
            

            # (W_MVRM) LQ FORWARD PASS
            print("LQ FORWARD PASS")
            
            
            # lq_encoder_out, lq_mvrm_out = self._run_model_forward(
            lq_out = self._run_model_forward(
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
                                            analysis = cfg.analysis_LQ,
                                            export_rgb_feat_layers=vis_rgb_LQ
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


            # LQ attn_map extraction 
            lq_maps = {} 
            if cfg.analysis_LQ.vis_map and len(cfg.analysis_LQ.da3_attn_map.extract_idx) != 0:
                print("LQ ATTN_MAP EXTRACTION")
                for layer_idx in cfg.analysis_LQ.da3_attn_map.extract_idx:
                    attn_idx, attn_type, attn_map = self.model.backbone.pretrained.blocks[layer_idx].attn.attn_map
                    assert layer_idx==attn_idx 
                    lq_maps[('da3', attn_idx, attn_type)] = attn_map.mean(dim=1)  # 1 head num_view*(n+1) num_view*(n+1)  -> 1 num_view*(n+1) num_view*(n+1)
                to_vis_imgs_list.append(('lq_imgs', lq_imgs))
                to_vis_attn_maps_list.append(('lq_maps', lq_maps))
                
            
            # LQ point_cloud extraction
            if cfg.analysis_LQ.vis_pc:
                print("LQ POINT_CLOUD EXTRACTION")
                with torch.no_grad():
                    lq_depth_t = lq_encoder_out.depth.unsqueeze(-1).float()                          # (b, v, H, W, 1)
                    lq_c2w_t   = affine_inverse(as_homogeneous(lq_pred_pose.float()))                # (b, v, 4, 4)
                    lq_pts     = unproject_depth(lq_depth_t, lq_pred_intrinsics.float(), lq_c2w_t)  # (b, v, H, W, 3)

                    PATCH_SIZE = 14
                    b, v_pts, H_pts, W_pts, _ = lq_pts.shape
                    Ph, Pw = H_pts // PATCH_SIZE, W_pts // PATCH_SIZE
                    n_pts  = Ph * Pw
                    pts_patch  = lq_pts.reshape(b, v_pts, Ph, PATCH_SIZE, Pw, PATCH_SIZE, 3).mean(dim=(3, 5))  # (b, v, Ph, Pw, 3)
                    pts_flat   = pts_patch.reshape(b, v_pts * n_pts, 3)                                         # (b, v*n, 3)
                    diff       = pts_flat.unsqueeze(1) - pts_flat.unsqueeze(2)                                  # (b, v*n, v*n, 3)
                    neg_l2     = -torch.norm(diff, dim=-1)                                                      # (b, v*n, v*n)
                    T = cfg.analysis_LQ.get('vis_pc_temperature', 1.0)
                    if T == -1: # hard assignment
                        _blocked = neg_l2.reshape(b, v_pts * n_pts, v_pts, n_pts)
                        lq_geo_corr = torch.zeros_like(_blocked).scatter_(-1, _blocked.argmax(dim=-1, keepdim=True), 1.0)
                        lq_geo_corr = lq_geo_corr.reshape(b, v_pts * n_pts, v_pts * n_pts)
                    else:
                        lq_geo_corr = (neg_l2 / T).softmax(dim=-1)                                              # (b, v*n, v*n)

                    # Cycle-consistency visibility mask
                    ref_idx    = torch.arange(n_pts, device=lq_geo_corr.device).unsqueeze(0).expand(b, -1)
                    cycle_mask = torch.zeros(b, v_pts * n_pts, v_pts * n_pts,
                                             dtype=torch.bool, device=lq_geo_corr.device)
                    for va in range(v_pts):
                        for vb in range(v_pts):
                            va_s, va_e = va * n_pts, (va + 1) * n_pts
                            vb_s, vb_e = vb * n_pts, (vb + 1) * n_pts
                            if va == vb:
                                cycle_mask[:, va_s:va_e, vb_s:vb_e] = True
                                continue
                            corr_ab = lq_geo_corr[:, va_s:va_e, vb_s:vb_e]
                            corr_ba = lq_geo_corr[:, vb_s:vb_e, va_s:va_e]
                            fwd     = corr_ab.argmax(dim=-1)
                            bwd     = corr_ba.argmax(dim=-1)
                            cycle_thresh = cfg.analysis_LQ.get('vis_pc_cycle_threshold', 0)
                            roundtrip = torch.gather(bwd, 1, fwd)
                            if cycle_thresh == 0:
                                visible = (roundtrip == ref_idx)
                            else:
                                rt_r,  rt_c  = roundtrip // Pw, roundtrip % Pw
                                ref_r, ref_c = ref_idx   // Pw, ref_idx   % Pw
                                dist    = torch.max((rt_r - ref_r).abs(), (rt_c - ref_c).abs())
                                visible = dist <= cycle_thresh
                            cycle_mask[:, va_s:va_e, vb_s:vb_e] = visible.unsqueeze(-1).expand(-1, -1, n_pts)
                    lq_geo_corr_masked = lq_geo_corr * cycle_mask.float()

                    # Reprojection visibility mask
                    depth_thr = cfg.analysis_LQ.get('vis_pc_reproj_depth_threshold', 0.1)
                    w2c_lq    = as_homogeneous(lq_pred_pose.float())        # (b, v, 4, 4) w2c
                    reproj_mask = _build_reproj_vis_mask(
                        pts_patch, lq_depth_t, w2c_lq, lq_pred_intrinsics.float(),
                        b, v_pts, Ph, Pw, n_pts, PATCH_SIZE, depth_thr)
                    lq_geo_corr_reproj = lq_geo_corr * reproj_mask.float()

                lq_pc_maps        = {('pc', 0, 'global'):       lq_geo_corr}
                lq_pc_maps_cycle  = {('pc', 0, 'global_cycle'): lq_geo_corr_masked}
                lq_pc_maps_reproj = {('pc', 0, 'global_reproj'): lq_geo_corr_reproj}
                to_vis_pc_list.append(('lq_pts', lq_pts))
                to_vis_pc_maps_list.append(('lq_pc_maps', lq_pc_maps))
                to_vis_pc_maps_cycle_list.append(('lq_pc_maps_cycle', lq_pc_maps_cycle))
                to_vis_pc_maps_reproj_list.append(('lq_pc_maps_reproj', lq_pc_maps_reproj))
                if not any(n == 'lq_imgs' for n, _ in to_vis_imgs_list):
                    to_vis_imgs_list.append(('lq_imgs', lq_imgs))
            else:
                T = 'none'
                depth_thr = 'none'
                cycle_thresh = 'none'  


            
            if rgb_decoder is not None and vis_rgb_LQ:
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
                    
            

            # # (W_MVRM) QUAL_RESTORMER FORWARD PASS
            # print("QUAL_RESTORMER FORWARD PASS")
            # ir_encoder_out, _ = self._run_model_forward(
            #                                 res_imgs, 
            #                                 ex_t_norm, 
            #                                 in_t, 
            #                                 export_feat_layers, 
            #                                 infer_gs, 
            #                                 use_ray_pose, 
            #                                 ref_view_strategy, 
            #                                 mvrm_cfg=cfg.mvrm.train, 
            #                                 mvrm_result=None, 
            #                                 mode=None, 
            #                                 ref_b_idx=lq_ref_b_idx
            #                             )
            # # lq_pred_pose_enc = lq_encoder_out.pose_enc      # 1 v 9
            # ir_pred_pose = ir_encoder_out['extrinsics']     # 1 v 3 4
            # # ir_ref_b_idx = ir_encoder_out.ref_b_idx
            # # safety check
            # ir_depth = ir_encoder_out.depth.unsqueeze(2)     # 1 v 1 h w



            # (W_MVRM) HQ FORWARD PASS 
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
                                            # ref_b_idx=None
                                            analysis = cfg.analysis_HQ,
                                            export_rgb_feat_layers=vis_rgb_HQ
                                        )
            # hq_pred_pose_enc = hq_encoder_out.pose_enc      # 1 v 9
            hq_pred_pose = hq_encoder_out['extrinsics']     # 1 v 3 4
            hq_pred_intrinsics = hq_encoder_out['intrinsics']  # (b, v, 3, 3)
            hq_latent = hq_mvrm_out[('extract_feat', cfg.mvrm.train.extract_feat_layers[0])]         # b v n+1 d
            hq_depth = hq_encoder_out.depth.unsqueeze(2)    # 1 v 1 h w

            hq_encoder_out_test, hq_mvrm_out_test = self._run_model_forward(
                                imgs, 
                                ex_t_norm, 
                                in_t, 
                                [5, 7, 9, 11], 
                                infer_gs, 
                                use_ray_pose, 
                                ref_view_strategy, 
                                mvrm_cfg=cfg.mvrm.train, 
                                mvrm_result=None, 
                                mode='train',
                                ref_b_idx=lq_ref_b_idx,
                                # ref_b_idx=None
                                analysis = cfg.analysis_HQ,
                                export_rgb_feat_layers=vis_rgb_HQ
                            )

            # HQ attn_map extraction 
            hq_maps = {} 
            if cfg.analysis_HQ.vis_map and len(cfg.analysis_HQ.da3_attn_map.extract_idx) != 0:
                print("HQ ATTN_MAP EXTRACTION")
                for layer_idx in cfg.analysis_HQ.da3_attn_map.extract_idx:
                    attn_idx, attn_type, attn_map = self.model.backbone.pretrained.blocks[layer_idx].attn.attn_map
                    assert layer_idx==attn_idx 
                    hq_maps[('da3', attn_idx, attn_type)] = attn_map.mean(dim=1)  # 1 head num_view*(n+1) num_view*(n+1)  -> 1 num_view*(n+1) num_view*(n+1)
                to_vis_imgs_list.append(('hq_imgs', imgs))
                to_vis_attn_maps_list.append(('hq_maps', hq_maps))
                

            
            # HQ point_cloud extraction
            if cfg.analysis_HQ.vis_pc:
                print("HQ POINT_CLOUD EXTRACTION")
                with torch.no_grad():
                    hq_depth_t = hq_encoder_out.depth.unsqueeze(-1).float()                          # (b, v, H, W, 1)
                    hq_c2w_t   = affine_inverse(as_homogeneous(hq_pred_pose.float()))                # (b, v, 4, 4)
                    hq_pts     = unproject_depth(hq_depth_t, hq_pred_intrinsics.float(), hq_c2w_t)  # (b, v, H, W, 3)


                    # from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
                    # visualizer = CameraPoseVisualizer([-50, 50], [-50, 50], [0, 100])
                    # pose = as_homogeneous(hq_pred_pose.float()).view(-1,4,4)
                    # for p in pose:
                    #     visualizer.extrinsic2pyramid(p.cpu().numpy(), 'c', 10)
                    # visualizer.save('tmp2.jpg')

                    PATCH_SIZE = 14
                    b, v_pts, H_pts, W_pts, _ = hq_pts.shape
                    Ph, Pw = H_pts // PATCH_SIZE, W_pts // PATCH_SIZE
                    n_pts  = Ph * Pw
                    pts_patch  = hq_pts.reshape(b, v_pts, Ph, PATCH_SIZE, Pw, PATCH_SIZE, 3).mean(dim=(3, 5))
                    pts_flat   = pts_patch.reshape(b, v_pts * n_pts, 3)
                    diff       = pts_flat.unsqueeze(1) - pts_flat.unsqueeze(2)
                    neg_l2     = -torch.norm(diff, dim=-1)
                    T = cfg.analysis_HQ.get('vis_pc_temperature', 1.0)
                    if T == -1: # hard assignment
                        _blocked = neg_l2.reshape(b, v_pts * n_pts, v_pts, n_pts)
                        hq_geo_corr = torch.zeros_like(_blocked).scatter_(-1, _blocked.argmax(dim=-1, keepdim=True), 1.0)
                        hq_geo_corr = hq_geo_corr.reshape(b, v_pts * n_pts, v_pts * n_pts)
                    else:
                        hq_geo_corr = (neg_l2 / T).softmax(dim=-1)                                              # (b, v*n, v*n)

                    # Cycle-consistency visibility mask
                    # For each view pair (va, vb): patch pi in va is "visible" in vb
                    # iff argmax(corr[va->vb][pi]) round-trips back to pi.
                    ref_idx    = torch.arange(n_pts, device=hq_geo_corr.device).unsqueeze(0).expand(b, -1)  # (b, n)
                    cycle_mask = torch.zeros(b, v_pts * n_pts, v_pts * n_pts,
                                             dtype=torch.bool, device=hq_geo_corr.device)
                    for va in range(v_pts):
                        for vb in range(v_pts):
                            va_s, va_e = va * n_pts, (va + 1) * n_pts
                            vb_s, vb_e = vb * n_pts, (vb + 1) * n_pts
                            if va == vb:
                                cycle_mask[:, va_s:va_e, vb_s:vb_e] = True   # self-view: always visible
                                continue
                            corr_ab  = hq_geo_corr[:, va_s:va_e, vb_s:vb_e]  # (b, n, n)
                            corr_ba  = hq_geo_corr[:, vb_s:vb_e, va_s:va_e]  # (b, n, n)
                            fwd      = corr_ab.argmax(dim=-1)                 # (b, n) best match in vb
                            bwd      = corr_ba.argmax(dim=-1)                 # (b, n) best match in va
                            # patch i is visible in vb iff round-trip lands within cycle_thresh patches of i
                            cycle_thresh = cfg.analysis_HQ.get('vis_pc_cycle_threshold', 0)
                            roundtrip = torch.gather(bwd, 1, fwd)             # (b, n)
                            if cycle_thresh == 0:
                                visible = (roundtrip == ref_idx)
                            else:
                                rt_r,  rt_c  = roundtrip // Pw, roundtrip % Pw
                                ref_r, ref_c = ref_idx   // Pw, ref_idx   % Pw
                                dist    = torch.max((rt_r - ref_r).abs(), (rt_c - ref_c).abs())
                                visible = dist <= cycle_thresh
                            cycle_mask[:, va_s:va_e, vb_s:vb_e] = visible.unsqueeze(-1).expand(-1, -1, n_pts)
                    

                    hq_geo_corr_masked = hq_geo_corr * cycle_mask.float()   # (b, v*n, v*n) zero-out invisible

                    # Reprojection visibility mask
                    depth_thr = cfg.analysis_HQ.get('vis_pc_reproj_depth_threshold', 0.1)
                    w2c_hq    = as_homogeneous(hq_pred_pose.float())        # (b, v, 4, 4) w2c
                    reproj_mask = _build_reproj_vis_mask(
                        pts_patch, hq_depth_t, w2c_hq, hq_pred_intrinsics.float(),
                        b, v_pts, Ph, Pw, n_pts, PATCH_SIZE, depth_thr)
                    hq_geo_corr_reproj = hq_geo_corr * reproj_mask.float()  # (b, v*n, v*n)

                hq_pc_maps        = {('pc', 0, 'global'):       hq_geo_corr}
                hq_pc_maps_cycle  = {('pc', 0, 'global_cycle'): hq_geo_corr_masked}
                hq_pc_maps_reproj = {('pc', 0, 'global_reproj'): hq_geo_corr_reproj}
                to_vis_pc_list.append(('hq_pts', hq_pts))
                to_vis_pc_maps_list.append(('hq_pc_maps', hq_pc_maps))
                to_vis_pc_maps_cycle_list.append(('hq_pc_maps_cycle', hq_pc_maps_cycle))
                to_vis_pc_maps_reproj_list.append(('hq_pc_maps_reproj', hq_pc_maps_reproj))
                if not any(n == 'hq_imgs' for n, _ in to_vis_imgs_list):
                    to_vis_imgs_list.append(('hq_imgs', imgs))
            else:
                T = 'none'
                depth_thr = 'none'
                cycle_thresh = 'none'
            
            
            if rgb_decoder is not None and vis_rgb_HQ:
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

            
            
            if cfg.mvrm.val.analysis.vis_attn_map:
                mvrm_vis = {} 
                mvrm_vis['mvrm_vis_attn_root'] = os.path.join(cfg.workspace.work_dir, 'pho_attn_mvrm_results', data, pose_setting)
                mvrm_vis['scene'] = scene 
                mvrm_vis['num_view'] = v
                mvrm_vis['model_img_size'] = (model_H, model_W) 
                mvrm_vis['lq_imgs'] = lq_imgs
                mvrm_vis['ref_b_idx'] = lq_ref_b_idx
                model_kwargs['mvrm_vis'] = mvrm_vis
                
               
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.no_grad():
                with torch.autocast(device_type=imgs.device.type, dtype=autocast_dtype):         
                    restored_samples = eval_sampler(x0, denoiser.forward, **model_kwargs)[-1]     # b v n d # eval_sampler: class ode def sample() function
            

            mvrm_result={}
            # OURS
            mvrm_result[('restored_latent', first_extract_layer_idx)] = restored_samples
            # CHECK UPPER BOUND
            # mvrm_result[('restored_latent', first_extract_layer_idx)] = hq_latent
            
            

            
            # REPLACE_W = 'hq_cam_tkn'
            # REPLACE_W = 'lq_cam_tkn'
            REPLACE_W = None
            
            if REPLACE_W == 'hq_cam_tkn':
                print("replacing with HQ CAMERA TOKEN")
                restored_samples[:,:,0] = hq_latent[:,:,0]   # replace the first token with the hq camera token
                
            elif REPLACE_W == 'hq_patch_tkn':
                print("replacing with HQ PATCH TOKEN")
                restored_samples[:,:,1:] = hq_latent[:,:,1:]   # replace the patch tokens with the hq patch tokens
                
            elif REPLACE_W == 'lq_cam_tkn': 
                print("replacing with LQ CAM TOKEN")
                restored_samples[:,:,0] = lq_latent[:,:,0]   # replace the first token with the lq camera token
                
            elif REPLACE_W == 'lq_patch_tkn':
                print("replacing with LQ PATCH TOKEN")
                restored_samples[:,:,1:] = lq_latent[:,:,1:]   # replace the patch tokens with the lq patch tokens
            

                
            # (W_MVRM) RES FORWARD PASS
            print("RES FORWARD PASS")
            raw_output, _ = self._run_model_forward(
                                            lq_imgs, 
                                            # imgs,           # for MVRM (HQ_RECON)
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
                                            analysis = cfg.analysis_RES,
                                            export_rgb_feat_layers=vis_rgb_RES
                                        )
            res_pred_pose_enc = raw_output.pose_enc      # 1 v 9
            res_pred_pose = raw_output['extrinsics']     # 1 v 3 4
            res_pred_intrinsics = raw_output['intrinsics']  # (b, v, 3, 3)
            res_depth = raw_output.depth.unsqueeze(2)    # 1 v 1 h w
            
            

            # RES attn_map extraction 
            res_maps = {} 
            if cfg.analysis_RES.vis_map and len(cfg.analysis_RES.da3_attn_map.extract_idx) != 0:
                print("RES ATTN_MAP EXTRACTION")
                for layer_idx in cfg.analysis_RES.da3_attn_map.extract_idx:
                    attn_idx, attn_type, attn_map = self.model.backbone.pretrained.blocks[layer_idx].attn.attn_map
                    assert layer_idx==attn_idx 
                    res_maps[('da3', attn_idx, attn_type)] = attn_map.mean(dim=1)  # 1 head num_view*(n+1) num_view*(n+1)  -> 1 num_view*(n+1) num_view*(n+1)
                to_vis_imgs_list.append(('res_imgs', lq_imgs))
                to_vis_attn_maps_list.append(('res_maps', res_maps))
                

            
            # RES point_cloud extraction
            if cfg.analysis_RES.vis_pc:
                print("RES POINT_CLOUD EXTRACTION")
                with torch.no_grad():
                    res_depth_t = raw_output.depth.unsqueeze(-1).float()                              # (b, v, H, W, 1)
                    res_c2w_t   = affine_inverse(as_homogeneous(res_pred_pose.float()))               # (b, v, 4, 4)
                    res_pts     = unproject_depth(res_depth_t, res_pred_intrinsics.float(), res_c2w_t)  # (b, v, H, W, 3)

                    PATCH_SIZE = 14
                    b, v_pts, H_pts, W_pts, _ = res_pts.shape
                    Ph, Pw = H_pts // PATCH_SIZE, W_pts // PATCH_SIZE
                    n_pts  = Ph * Pw
                    pts_patch    = res_pts.reshape(b, v_pts, Ph, PATCH_SIZE, Pw, PATCH_SIZE, 3).mean(dim=(3, 5))
                    pts_flat     = pts_patch.reshape(b, v_pts * n_pts, 3)
                    diff         = pts_flat.unsqueeze(1) - pts_flat.unsqueeze(2)
                    neg_l2       = -torch.norm(diff, dim=-1)
                    T = cfg.analysis_RES.get('vis_pc_temperature', 1.0)
                    if T == -1: # hard assignment
                        _blocked = neg_l2.reshape(b, v_pts * n_pts, v_pts, n_pts)
                        res_geo_corr = torch.zeros_like(_blocked).scatter_(-1, _blocked.argmax(dim=-1, keepdim=True), 1.0)
                        res_geo_corr = res_geo_corr.reshape(b, v_pts * n_pts, v_pts * n_pts)
                    else:
                        res_geo_corr = (neg_l2 / T).softmax(dim=-1)                                     # (b, v*n, v*n)

                    # Cycle-consistency visibility mask
                    ref_idx    = torch.arange(n_pts, device=res_geo_corr.device).unsqueeze(0).expand(b, -1)
                    cycle_mask = torch.zeros(b, v_pts * n_pts, v_pts * n_pts,
                                             dtype=torch.bool, device=res_geo_corr.device)
                    for va in range(v_pts):
                        for vb in range(v_pts):
                            va_s, va_e = va * n_pts, (va + 1) * n_pts
                            vb_s, vb_e = vb * n_pts, (vb + 1) * n_pts
                            if va == vb:
                                cycle_mask[:, va_s:va_e, vb_s:vb_e] = True
                                continue
                            corr_ab = res_geo_corr[:, va_s:va_e, vb_s:vb_e]
                            corr_ba = res_geo_corr[:, vb_s:vb_e, va_s:va_e]
                            fwd     = corr_ab.argmax(dim=-1)
                            bwd     = corr_ba.argmax(dim=-1)
                            cycle_thresh = cfg.analysis_RES.get('vis_pc_cycle_threshold', 0)
                            roundtrip = torch.gather(bwd, 1, fwd)
                            if cycle_thresh == 0:
                                visible = (roundtrip == ref_idx)
                            else:
                                rt_r,  rt_c  = roundtrip // Pw, roundtrip % Pw
                                ref_r, ref_c = ref_idx   // Pw, ref_idx   % Pw
                                dist    = torch.max((rt_r - ref_r).abs(), (rt_c - ref_c).abs())
                                visible = dist <= cycle_thresh
                            cycle_mask[:, va_s:va_e, vb_s:vb_e] = visible.unsqueeze(-1).expand(-1, -1, n_pts)
                    res_geo_corr_masked = res_geo_corr * cycle_mask.float()

                    # Reprojection visibility mask
                    depth_thr = cfg.analysis_RES.get('vis_pc_reproj_depth_threshold', 0.1)
                    w2c_res   = as_homogeneous(res_pred_pose.float())       # (b, v, 4, 4) w2c
                    reproj_mask = _build_reproj_vis_mask(
                        pts_patch, res_depth_t, w2c_res, res_pred_intrinsics.float(),
                        b, v_pts, Ph, Pw, n_pts, PATCH_SIZE, depth_thr)
                    res_geo_corr_reproj = res_geo_corr * reproj_mask.float()

                res_pc_maps        = {('pc', 0, 'global'):       res_geo_corr}
                res_pc_maps_cycle  = {('pc', 0, 'global_cycle'): res_geo_corr_masked}
                res_pc_maps_reproj = {('pc', 0, 'global_reproj'): res_geo_corr_reproj}
                to_vis_pc_list.append(('res_pts', res_pts))
                to_vis_pc_maps_list.append(('res_pc_maps', res_pc_maps))
                to_vis_pc_maps_cycle_list.append(('res_pc_maps_cycle', res_pc_maps_cycle))
                to_vis_pc_maps_reproj_list.append(('res_pc_maps_reproj', res_pc_maps_reproj))
                if not any(n == 'res_imgs' for n, _ in to_vis_imgs_list):
                    to_vis_imgs_list.append(('res_imgs', lq_imgs))
            else:
                T = 'none'
                depth_thr = 'none'
                cycle_thresh = 'none'
            
            
            
            
            if rgb_decoder is not None and vis_rgb_RES:
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



            if vis_rgb_LQ and vis_rgb_HQ and vis_rgb_RES:
                vis_rgb_recon_root = os.path.join(cfg.workspace.work_dir, 'pho_rgb_recon_results',  data, pose_setting)
                os.makedirs(vis_rgb_recon_root, exist_ok=True)                                                           
                mean = torch.tensor([0.485, 0.456, 0.406], device=rgb_hq.device, dtype=rgb_hq.dtype).view(1, 3, 1, 1) 
                std  = torch.tensor([0.229, 0.224, 0.225], device=rgb_hq.device, dtype=rgb_hq.dtype).view(1, 3, 1, 1)  
                def denorm(x):  
                    return (x * std + mean).clamp(0, 1)                                                                 
                rgb_hq_dn  = denorm(rgb_hq.squeeze(0).float())   # [v, 3, 378, 504]
                rgb_lq_dn  = denorm(rgb_lq.squeeze(0).float())                                                                   
                rgb_res_dn = denorm(rgb_res.squeeze(0).float())                                                                  
                # Each row: all 10 frames concatenated horizontally → [3, 378*v, 504]                                
                row_lq  = torch.cat([rgb_lq_dn[i]  for i in range(len(rgb_lq_dn))], dim=-2)
                row_hq  = torch.cat([rgb_hq_dn[i]  for i in range(len(rgb_hq_dn))], dim=-2)                                        
                row_res = torch.cat([rgb_res_dn[i] for i in range(len(rgb_res_dn))], dim=-2)  
                # Stack 3 rows vertically → [3, 378*v, 504*3]                                                        
                combined = torch.cat([row_hq, row_lq, row_res], dim=-1)                                                                                                                              
                img = TF.to_pil_image(combined.cpu())
                img.save(os.path.join(vis_rgb_recon_root, f'{scene}.png'))
                
                
                
            if vis_rgb_framewise:
                # vis_rgb_recon_frame_root = export_dir.replace('model_results', 'pho_rgb_recon_frame_results')
                    
                # if 'hiroom' in vis_rgb_recon_frame_root:
                #     vis_rgb_recon_frame_root = vis_rgb_recon_frame_root.replace('hiroom', 'hiroom/data')
                #     vis_rgb_recon_frame_root = vis_rgb_recon_frame_root.replace('unposed', 'image')
                
                
                assert len(rgb_res_dn) == len(image.image_files)
                
                vis_rgb_recon_frame_root = os.path.join(cfg.workspace.work_dir, 'pho_rgb_recon_frame_results')
                
                for img_idx, img_id_full_path in enumerate(image.image_files):
                    img_id_path = img_id_full_path.split('clean')[-1]
                    rgb_res_save_path = vis_rgb_recon_frame_root + img_id_path
                    img_folder = os.path.dirname(rgb_res_save_path)
                    os.makedirs(img_folder, exist_ok=True)
                    img = TF.to_pil_image(rgb_res_dn[img_idx].cpu())
                    img.save(rgb_res_save_path)
                    
                    
                                     
                # vis_rgb_recon_frame_root = os.path.join(
                #     cfg.workspace.work_dir,
                #     'pho_rgb_recon_frame_results',
                #     data,
                #     pose_setting,
                # )
                
                
                # save_mvrm_decoder_rgb_frames(
                #     rgb_hq=rgb_hq,                 # (1, V, 3, H, W) or (V, 3, H, W)
                #     rgb_lq=rgb_lq,
                #     rgb_res=rgb_res,
                #     scene=scene,                  
                #     image_files=image.image_files,
                #     output_root=vis_rgb_recon_frame_root,
                # )
                
                
            
            # ## calculate image metrics (PSNR, SSIM, LPIPS and FID)
            # if vis_rgb_LQ and vis_rgb_HQ and vis_rgb_RES:
            #     from lpips import LPIPS as LPIPSMetric
            #     from torchmetrics.functional.image import structural_similarity_index_measure
            #     from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
            #     import scipy.linalg

            #     # denorm imgs (ImageNet-normalized) → [0, 1], shape [v, 3, H, W]
            #     imgs_dn = denorm(imgs.squeeze(0).float())
            #     device = imgs_dn.device

            #     def _psnr(a, b):
            #         mse = ((a - b) ** 2).mean(dim=[1, 2, 3]).clamp(min=1e-10)
            #         return (20.0 * torch.log10(1.0 / mse.sqrt())).mean().item()

            #     def _ssim(a, b):
            #         return structural_similarity_index_measure(a, b, data_range=1.0).item()

            #     lpips_fn = LPIPSMetric(net="vgg").to(device).eval()
            #     def _lpips(a, b):
            #         a11 = a * 2.0 - 1.0
            #         b11 = b * 2.0 - 1.0
            #         with torch.no_grad():
            #             return lpips_fn(a11, b11).mean().item()

            #     fe = FeatureExtractorInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
            #     def _fid(a, b):
            #         def _feats(t):
            #             u8 = (t.clamp(0, 1) * 255).byte()
            #             with torch.no_grad():
            #                 return fe(u8)[0].double().cpu().numpy()
            #         fa, fb = _feats(a), _feats(b)
            #         mu_a, sig_a = fa.mean(0), np.cov(fa, rowvar=False)
            #         mu_b, sig_b = fb.mean(0), np.cov(fb, rowvar=False)
            #         diff = mu_a - mu_b
            #         cov = scipy.linalg.sqrtm(sig_a @ sig_b)
            #         if np.iscomplexobj(cov):
            #             cov = cov.real
            #         return float(max(diff @ diff + np.trace(sig_a + sig_b - 2.0 * cov), 0.0))

            #     print(f"\n[Image Metrics] scene={scene}")
            #     print(f"  --- imgs vs rgb_hq (HQ) ---")
            #     print(f"  PSNR : {_psnr(imgs_dn, rgb_hq_dn):.4f} dB")
            #     print(f"  SSIM : {_ssim(imgs_dn, rgb_hq_dn):.4f}")
            #     print(f"  LPIPS: {_lpips(imgs_dn, rgb_hq_dn):.4f}")
            #     print(f"  FID  : {_fid(imgs_dn, rgb_hq_dn):.4f}")
            #     print(f"  --- imgs vs rgb_res (RES) ---")
            #     print(f"  PSNR : {_psnr(imgs_dn, rgb_res_dn):.4f} dB")
            #     print(f"  SSIM : {_ssim(imgs_dn, rgb_res_dn):.4f}")
            #     print(f"  LPIPS: {_lpips(imgs_dn, rgb_res_dn):.4f}")
            #     print(f"  FID  : {_fid(imgs_dn, rgb_res_dn):.4f}")





                                                                        
            # import torch.nn.functional as F                                                                                         
                                                                                                                                    
            # # cycle_mask: (b, v*n, v*n)                                                                                             
            # mask = cycle_mask[0].float()                              # (v*n, v*n)                                                  
            # # mask_per_patch = mask.mean(dim=-1)                        # (v*n,) mean visibility per source patch                     
            # mask_per_patch = mask.any(dim=-1).float()   # (v*n,)   
            # mask_per_view  = mask_per_patch.reshape(v_pts, 1, Ph, Pw) # (v, 1, Ph, Pw)                                              
            # mask_per_view  = F.interpolate(mask_per_view, size=(H_pts, W_pts), mode='nearest')  # (v, 1, H, W)                      
            # save_image(mask_per_view, 'cycle_mask2.png', nrow=v_pts)                                                                 
                                                
            # save_image(imgs[0], 'imgs.png')

            
            
            vis_pc_root = os.path.join(cfg.workspace.work_dir, 'pho_pointcloud_da3_results',  f'temp{T}', data, pose_setting)
            vis_pointcloud_correspondence_maps(
                vis_save_root=vis_pc_root,
                scene=scene,
                num_view = v,
                model_img_size = (model_H, model_W),
                imgs_list = to_vis_imgs_list, 
                attn_maps_list = to_vis_pc_maps_list,
                ref_b_idx = lq_ref_b_idx
            )
            



            vis_pc_cycle_root = os.path.join(cfg.workspace.work_dir, 'pho_pointcloud_cycle_da3_results', f'temp{T}', f'cycle_thres{cycle_thresh}' ,data, pose_setting)
            vis_pointcloud_cycle_correspondence_maps(
                vis_save_root=vis_pc_cycle_root,
                scene=scene,
                num_view = v,
                model_img_size = (model_H, model_W),
                imgs_list = to_vis_imgs_list,
                attn_maps_list = to_vis_pc_maps_cycle_list,
                ref_b_idx = lq_ref_b_idx
            )

            vis_pc_reproj_root = os.path.join(cfg.workspace.work_dir, 'pho_pointcloud_reproj_da3_results', f'temp{T}', f'reproj_thres{depth_thr}', data, pose_setting)
            vis_pointcloud_reproj_correspondence_maps(
                vis_save_root=vis_pc_reproj_root,
                scene=scene,
                num_view = v,
                model_img_size = (model_H, model_W),
                imgs_list = to_vis_imgs_list,
                attn_maps_list = to_vis_pc_maps_reproj_list,
                ref_b_idx = lq_ref_b_idx
            )



            vis_attn_root = os.path.join(cfg.workspace.work_dir, 'pho_attn_da3_results', data, pose_setting)
            vis_attn_maps(
                vis_save_root=vis_attn_root,
                scene=scene,
                num_view = v,
                model_img_size = (model_H, model_W),
                imgs_list = to_vis_imgs_list, 
                attn_maps_list = to_vis_attn_maps_list,
                ref_b_idx = lq_ref_b_idx
            )
            

            vis_save_root = os.path.join(cfg.workspace.work_dir, 'pho_vis_results', data, pose_setting)
            vis_all(
                vis_save_root=vis_save_root,
                scene=scene,
                hq_img=imgs[0],
                lq_img=lq_imgs[0],
                hq_depth=hq_depth[0],
                lq_depth=lq_depth[0],
                res_depth=res_depth[0],
            )
            metric_save_root = os.path.join(cfg.workspace.work_dir, 'pho_metric_results', data, pose_setting)
            metric_all(
                metric_save_root=metric_save_root,
                scene=scene,
                poses = (hq_pred_pose[0], lq_pred_pose[0],res_pred_pose[0]),
                depths = (hq_depth, lq_depth, res_depth)
            )
            featsim_log = featsim_all(hq_encoder_out, lq_encoder_out, raw_output)
            featsim_save_root = os.path.join(cfg.workspace.work_dir, 'pho_featsim_results', data, pose_setting)
            plot_three_similarity_panels(
                featsim_log,
                save_path=f"{featsim_save_root}/{scene}_sim_all_combined.png"
            )
            cam_save_root = os.path.join(cfg.workspace.work_dir, 'pho_cam_traj_results', data, pose_setting)
            # plot_cam_trajectory(hq_pred_pose[0], lq_pred_pose[0], res_pred_pose[0], visualize_direction=False, save_path=f"{cam_save_root}/{scene}.png")
            plot_cam_trajectory_fair(hq_pred_pose[0], lq_pred_pose[0], res_pred_pose[0], visualize_direction=False, save_path=f"{cam_save_root}/fair_{scene}.png")




            # costvolume_save_root = os.path.join(cfg.workspace.work_dir, 'pho_costvolume_results', data, pose_setting)
            # costvolume_pck(
            #     hq_encoder_out, lq_encoder_out, raw_output,
            #     hq_imgs=imgs[0],
            #     lq_imgs=lq_imgs[0],
            #     save_root=costvolume_save_root,
            #     scene=scene,
            #     ref_b_idx=lq_ref_b_idx,
            #     # vis_layers=[-1],   # [-1] = deepest layer only; change to e.g. [12, 24, 39] or None for all
            #     vis_layers=[18]
            # )
                        
            
            
            
            # plot_all_cam_trajectory_fair(
            #     hq_pred_pose[0], 
            #     lq_pred_pose[0],
            #     ir_pred_pose[0],
            #     res_pred_pose[0],  
            #     labels = ['HQ (GT)', 'LQ', 'Restormer', 'CoRe3D (Ours)'],
            #     line_widths = [3, 2, 2, 2],
            #     colors = ['g', 'r', 'm', 'b'],
            #     visualize_direction=False, 
            #     save_path=f"{cam_save_root}/all_fair_{scene}.png"
            # )

            # freq_save_root = os.path.join(cfg.workspace.work_dir, 'pho_freq_results', data, pose_setting)
            # plot_freq_analysis_per_view(
            #     hq_out=hq_encoder_out,
            #     lq_out=lq_encoder_out,
            #     res_out=raw_output,
            #     save_root=freq_save_root,
            #     scene=scene,
            #     patch_grid=(model_H//14, model_W//14),
            #     avg_embedding=True
            # )
            
            # pca2_save_root = os.path.join(cfg.workspace.work_dir, 'pho_pca2_results', data, pose_setting)
            # plot_pca2_analysis_per_view(
            #     hq_out=hq_encoder_out,
            #     lq_out=lq_encoder_out,
            #     res_out=raw_output,
            #     save_root=pca2_save_root,
            #     scene=scene,
            #     patch_grid=(model_H//14, model_W//14),
            #     device='cuda'
            # )

            # pca3_save_root = os.path.join(cfg.workspace.work_dir, 'pho_pca3_results', data, pose_setting)
            # plot_pca3_analysis_per_view(
            #     hq_out=hq_encoder_out,
            #     lq_out=lq_encoder_out,
            #     res_out=raw_output,
            #     save_root=pca3_save_root,
            #     scene=scene,
            #     patch_grid=(model_H//14, model_W//14),
            #     device='cuda'
            # )
            
            # breakpoint()
            
            

        elif cfg.MVRM_EVAL.eval_method == 'w_mvrm_front_back':       
            print('-'*70)      
            print('APPLYING MVRM_FRONT_BACK O')
            print('-'*70)
            
            

            # (W_MVRM FRONT BACK) LQ FORWARD PASS
            print("LQ FORWARD PASS (EXTRACTING MVRM FRONT BACK FEATS)")
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
                                            ref_b_idx=None
                                        )
            # lq_pred_pose_enc = lq_encoder_out.pose_enc      # 1 v 9
            lq_pred_pose = lq_encoder_out['extrinsics']     # 1 v 3 4
            lq_ref_b_idx = lq_encoder_out.ref_b_idx
            # safety check
            for key in lq_mvrm_out.keys():
                assert key[-1] in cfg.mvrm.train.extract_feat_layers, f"Extracted MVRM feature layer {key[-1]} not in config extract_feat_layers {cfg.mvrm.train.extract_feat_layers}"
            first_extract_layer_idx = cfg.mvrm.train.extract_feat_layers[0]
            second_extract_layer_idx = cfg.mvrm.train.extract_feat_layers[1]
            lq_latent_mvrm1 = lq_mvrm_out[('extract_feat', first_extract_layer_idx)]         # b v n+1 d
            lq_depth = lq_encoder_out.depth.unsqueeze(2)     # 1 v 1 h w
                        




            # (W_MVRM_FRONT_BACK) HQ FORWARD PASS 
            print("HQ FORWARD PASS")
            hq_encoder_out, _ = self._run_model_forward(
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
                                            ref_b_idx=lq_ref_b_idx
                                        )
            # hq_pred_pose_enc = hq_encoder_out.pose_enc      # 1 v 9
            hq_pred_pose = hq_encoder_out['extrinsics']     # 1 v 3 4
            # hq_latent = hq_mvrm_out['extract_feat']         # b v 973 3072
            hq_depth = hq_encoder_out.depth.unsqueeze(2)    # 1 v 1 h w
            
            

            
            # generate pure noise
            noise_generator.manual_seed(42)
            pure_noise1 = torch.randn(lq_latent_mvrm1.shape, generator=noise_generator, device=imgs.device, dtype=torch.float32)
            # lq_latent condition method
            if cfg.mvrm.lq_latent_cond == 'addition':
                xt = pure_noise1 + lq_latent_mvrm1
            model_kwargs={
                'model_img_size': (model_H, model_W)
            }
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.no_grad():
                with torch.autocast(device_type=imgs.device.type, dtype=autocast_dtype):                
                    restored_samples_mvrm1 = eval_sampler(xt, denoiser.forward, **model_kwargs)[-1]     # b v n d
            mvrm_result={}
            mvrm_result[('restored_latent', first_extract_layer_idx)] = restored_samples_mvrm1
            # mvrm_result['restored_latent'] = hq_latent
            
            
  
            lq_latent_mvrm2 = lq_mvrm_out[('extract_feat', second_extract_layer_idx)]         # b v n+1 d
            # generate pure noise
            noise_generator.manual_seed(42)
            pure_noise2 = torch.randn(lq_latent_mvrm2.shape, generator=noise_generator, device=imgs.device, dtype=torch.float32)
            # lq_latent condition method
            if cfg.mvrm.lq_latent_cond == 'addition':
                xt = pure_noise2 + lq_latent_mvrm2
            model_kwargs={
                'model_img_size': (model_H, model_W)
            }
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.no_grad():
                with torch.autocast(device_type=imgs.device.type, dtype=autocast_dtype):                
                    restored_samples_mvrm2 = eval_sampler(xt, denoiser2.forward, **model_kwargs)[-1]     # b v n d

            mvrm_result[('restored_latent', second_extract_layer_idx)] = restored_samples_mvrm2
            # mvrm_result['restored_latent'] = hq_latent
            

            # (W_MVRM_FRONT_BACK) RES FORWARD PASS
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
                                            mode='val'
                                        )
            # res_pred_pose_enc = raw_output.pose_enc      # 1 v 9
            res_pred_pose = raw_output['extrinsics']     # 1 v 3 4
            res_depth = raw_output.depth.unsqueeze(2)    # 1 v 1 h w
            
            scene = scene.replace('/', '_') if '/' in scene else scene
            
            # from torchvision.utils import save_image
            # save_image(imgs.squeeze(0), 'img.jpg', normalize=True)
            # save_image(lq_imgs.squeeze(0), 'img_lq.jpg', normalize=True)


            vis_save_root = os.path.join(cfg.workspace.work_dir, 'pho_vis_results', data, pose_setting)
            vis_all(
                vis_save_root=vis_save_root,
                scene=scene,
                hq_img=imgs[0],
                lq_img=lq_imgs[0],
                hq_depth=hq_depth[0],
                lq_depth=lq_depth[0],
                res_depth=res_depth[0],
            )
            metric_save_root = os.path.join(cfg.workspace.work_dir, 'pho_metric_results', data, pose_setting)
            metric_all(
                metric_save_root=metric_save_root,
                scene=scene,
                poses = (hq_pred_pose[0], lq_pred_pose[0],res_pred_pose[0]),
                depths = (hq_depth, lq_depth, res_depth)
            )
            featsim_log = featsim_all(hq_encoder_out, lq_encoder_out, raw_output)
            featsim_save_root = os.path.join(cfg.workspace.work_dir, 'pho_featsim_results', data, pose_setting)
            plot_three_similarity_panels(
                featsim_log,
                save_path=f"{featsim_save_root}/{scene}_sim_all_combined.png"
            )
            cam_save_root = os.path.join(cfg.workspace.work_dir, 'pho_cam_traj_results', data, pose_setting)
            plot_cam_trajectory(hq_pred_pose[0], lq_pred_pose[0], res_pred_pose[0], visualize_direction=False, save_path=f"{cam_save_root}/{scene}.png")
     
     
     






        elif cfg.MVRM_EVAL.eval_method == 'w_mvrm_front_connect_back':       
            print('-'*70)      
            print('APPLYING W_MVRM_FRONT_CONNECT_BACK O')
            print('-'*70)
            



            # (W_MVRM FRONT_CONNECT_BACK) LQ FORWARD PASS
            print("LQ FORWARD PASS (MVRM FRONT_CONNECT_BACK FEATS)")
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
                                            ref_b_idx=None
                                        )
            # lq_pred_pose_enc = lq_encoder_out.pose_enc      # 1 v 9
            lq_pred_pose = lq_encoder_out['extrinsics']     # 1 v 3 4
            lq_ref_b_idx = lq_encoder_out.ref_b_idx
            lq_latent_mvrm1 = lq_mvrm_out[('extract_feat', cfg.mvrm.train.extract_feat_layers[0])]         # b v n+1 d
            lq_depth = lq_encoder_out.depth.unsqueeze(2)     # 1 v 1 h w
                        


            # (W_MVRM FRONT_CONNECT_BACK) HQ FORWARD PASS 
            print("HQ FORWARD PASS (MVRM FRONT_CONNECT_BACK FEATS)")
            hq_encoder_out, _ = self._run_model_forward(
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
                                            ref_b_idx=lq_ref_b_idx
                                        )
            # hq_pred_pose_enc = hq_encoder_out.pose_enc      # 1 v 9
            hq_pred_pose = hq_encoder_out['extrinsics']     # 1 v 3 4
            # hq_latent = hq_mvrm_out['extract_feat']         # b v 973 3072
            hq_depth = hq_encoder_out.depth.unsqueeze(2)    # 1 v 1 h w
            
            

            
            # generate pure noise
            noise_generator.manual_seed(42)
            pure_noise1 = torch.randn(lq_latent_mvrm1.shape, generator=noise_generator, device=imgs.device, dtype=torch.float32)
            # lq_latent condition method
            if cfg.mvrm.lq_latent_cond == 'addition':
                xt = pure_noise1 + lq_latent_mvrm1
            model_kwargs={
                'model_img_size': (model_H, model_W)
            }
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.no_grad():
                with torch.autocast(device_type=imgs.device.type, dtype=autocast_dtype):                
                    restored_samples_mvrm1 = eval_sampler(xt, denoiser.forward, **model_kwargs)[-1]     # b v n d
            mvrm_result={}
            mvrm_result[('restored_latent', cfg.mvrm.train.extract_feat_layers[0])] = restored_samples_mvrm1
            # mvrm_result['restored_latent'] = hq_latent
            


            # (W_MVRM FRONT_CONNECT_BACK) LQ2 FORWARD PASS
            print("LQ2 FORWARD PASS (MVRM FRONT_CONNECT_BACK FEATS)")
            lq_encoder_out2, lq_mvrm_out2 = self._run_model_forward(
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
                                            ref_b_idx=None,
                                            front_connect_back_mvrm_cfg = cfg.mvrm2
                                        )

            # lq_pred_pose_enc = lq_encoder_out.pose_enc      # 1 v 9
            lq_pred_pose2 = lq_encoder_out2['extrinsics']     # 1 v 3 4
            lq_ref_b_idx2 = lq_encoder_out2.ref_b_idx
            lq_latent_mvrm2 = lq_mvrm_out2[('extract_feat', cfg.mvrm2.train.extract_feat_layers[0])]         # b v n+1 d
            lq_depth2 = lq_encoder_out2.depth.unsqueeze(2)     # 1 v 1 h w
            
            
            
            # generate pure noise
            noise_generator.manual_seed(42)
            pure_noise2 = torch.randn(lq_latent_mvrm2.shape, generator=noise_generator, device=imgs.device, dtype=torch.float32)
            # lq_latent condition method
            if cfg.mvrm.lq_latent_cond == 'addition':
                xt = pure_noise2 + lq_latent_mvrm2
            model_kwargs={
                'model_img_size': (model_H, model_W)
            }
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.no_grad():
                with torch.autocast(device_type=imgs.device.type, dtype=autocast_dtype):                
                    restored_samples_mvrm2 = eval_sampler(xt, denoiser2.forward, **model_kwargs)[-1]     # b v n d

            mvrm_result[('restored_latent', cfg.mvrm2.train.extract_feat_layers[0])] = restored_samples_mvrm2
            # mvrm_result['restored_latent'] = hq_latent
            

            # (W_MVRM FRONT_CONNECT_BACK) RES FORWARD PASS
            print("RES FORWARD PASS (MVRM FRONT_CONNECT_BACK FEATS)")
            raw_output, _ = self._run_model_forward(
                                            lq_imgs, 
                                            ex_t_norm, 
                                            in_t, 
                                            export_feat_layers, 
                                            infer_gs, 
                                            use_ray_pose, 
                                            ref_view_strategy, 
                                            mvrm_cfg=cfg.mvrm2.val, 
                                            mvrm_result=mvrm_result, 
                                            mode='val'
                                        )
            # res_pred_pose_enc = raw_output.pose_enc      # 1 v 9
            res_pred_pose = raw_output['extrinsics']     # 1 v 3 4
            res_depth = raw_output.depth.unsqueeze(2)    # 1 v 1 h w
            
            scene = scene.replace('/', '_') if '/' in scene else scene
            
            # from torchvision.utils import save_image
            # save_image(imgs.squeeze(0), 'img.jpg', normalize=True)
            # save_image(lq_imgs.squeeze(0), 'img_lq.jpg', normalize=True)


            vis_save_root = os.path.join(cfg.workspace.work_dir, 'pho_vis_results', data, pose_setting)
            vis_all(
                vis_save_root=vis_save_root,
                scene=scene,
                hq_img=imgs[0],
                lq_img=lq_imgs[0],
                hq_depth=hq_depth[0],
                lq_depth=lq_depth[0],
                res_depth=res_depth[0],
            )
            metric_save_root = os.path.join(cfg.workspace.work_dir, 'pho_metric_results', data, pose_setting)
            metric_all(
                metric_save_root=metric_save_root,
                scene=scene,
                poses = (hq_pred_pose[0], lq_pred_pose[0],res_pred_pose[0]),
                depths = (hq_depth, lq_depth, res_depth)
            )
            featsim_log = featsim_all(hq_encoder_out, lq_encoder_out, raw_output)
            featsim_save_root = os.path.join(cfg.workspace.work_dir, 'pho_featsim_results', data, pose_setting)
            plot_three_similarity_panels(
                featsim_log,
                save_path=f"{featsim_save_root}/{scene}_sim_all_combined.png"
            )
            cam_save_root = os.path.join(cfg.workspace.work_dir, 'pho_cam_traj_results', data, pose_setting)
            plot_cam_trajectory(hq_pred_pose[0], lq_pred_pose[0], res_pred_pose[0], visualize_direction=False, save_path=f"{cam_save_root}/{scene}.png")
     
            







            
            
            
            

        elif cfg.MVRM_EVAL.eval_method == 'wo_mvrm_IR':             
            print('-'*70)
            print('WO_MVRM_IR')
            print('-'*70)
            

            # (WO_MVRM_IR) LQ FORWARD PASS
            print("LQ FORWARD PASS")
            # lq_encoder_out, _ = self._run_model_forward(
            lq_encoder_out = self._run_model_forward(
                                            # lq_imgs, 
                                            vggt_lq_imgs,
                                            ex_t_norm, 
                                            in_t, 
                                            export_feat_layers, 
                                            infer_gs, 
                                            use_ray_pose, 
                                            ref_view_strategy, 
                                            mvrm_cfg=None, 
                                            mvrm_result=None, 
                                            mode=None,
                                            ref_b_idx=None
                                        )
            lq_pred_pose = lq_encoder_out['extrinsics']     # 1 v 3 4
            lq_depth = lq_encoder_out['depth'].permute(0,1,4,2,3)    # 1 v 1 h w
            # lq_ref_b_idx = lq_encoder_out.ref_b_idx
            lq_ref_b_idx = None
            
            
            # (WO_MVRM_IR) HQ FORWARD PASS
            print("HQ FORWARD PASS")
            # hq_encoder_out, _ = self._run_model_forward(
            hq_encoder_out = self._run_model_forward(
                                            # imgs, 
                                            vggt_imgs,
                                            ex_t_norm, 
                                            in_t, 
                                            export_feat_layers, 
                                            infer_gs, 
                                            use_ray_pose, 
                                            ref_view_strategy, 
                                            mvrm_cfg=None, 
                                            mvrm_result=None, 
                                            mode=None,
                                            ref_b_idx=lq_ref_b_idx
                                            # ref_b_idx=None
                                        )
            hq_pred_pose = hq_encoder_out['extrinsics']     # 1 v 3 4
            hq_depth = hq_encoder_out['depth'].permute(0,1,4,2,3)    # 1 v 1 h w
            


            # (WO_MVRM_IR) IR IMG FORWARD PASS
            print("IR IMG FORWARD PASS")
            # raw_output, _ = self._run_model_forward(
            raw_output = self._run_model_forward(
                                            # res_imgs, 
                                            vggt_res_imgs,
                                            ex_t_norm, 
                                            in_t, 
                                            export_feat_layers, 
                                            infer_gs, 
                                            use_ray_pose, 
                                            ref_view_strategy, 
                                            mvrm_cfg=None, 
                                            mvrm_result=None, 
                                            mode=None,
                                        )
            res_pred_pose = raw_output['extrinsics']     # 1 v 3 4
            res_depth = raw_output['depth'].permute(0,1,4,2,3)    # 1 v 1 h w
            
            scene = scene.replace('/', '_') if '/' in scene else scene
            
            vis_save_root = os.path.join(cfg.workspace.work_dir, 'pho_vis_results', data, pose_setting)
            vis_all(
                vis_save_root=vis_save_root,
                scene=scene,
                hq_img=imgs[0],
                lq_img=res_imgs[0],
                hq_depth=hq_depth[0],
                lq_depth=lq_depth[0],
                res_depth=res_depth[0],
            )
            metric_save_root = os.path.join(cfg.workspace.work_dir, 'pho_metric_results', data, pose_setting)
            metric_all(
                metric_save_root=metric_save_root,
                scene=scene,
                poses = (hq_pred_pose[0], lq_pred_pose[0],res_pred_pose[0]),
                depths = (hq_depth, lq_depth, res_depth)
            )
            featsim_log = featsim_all(hq_encoder_out, lq_encoder_out, raw_output)
            featsim_save_root = os.path.join(cfg.workspace.work_dir, 'pho_featsim_results', data, pose_setting)
            plot_three_similarity_panels(
                featsim_log,
                save_path=f"{featsim_save_root}/{scene}_sim_all_combined.png"
            )
            cam_save_root = os.path.join(cfg.workspace.work_dir, 'pho_cam_traj_results', data, pose_setting)
            # plot_cam_trajectory(hq_pred_pose[0], lq_pred_pose[0], res_pred_pose[0], visualize_direction=False, save_path=f"{cam_save_root}/{scene}.png")
            plot_cam_trajectory_fair(hq_pred_pose[0], lq_pred_pose[0], res_pred_pose[0], visualize_direction=False, save_path=f"{cam_save_root}/fair_{scene}.png")
            
            # breakpoint()



            
        elif cfg.MVRM_EVAL.eval_method == 'wo_mvrm_LQ':       
            print('-'*70)
            print('WO_MVRM_LQ')
            print('-'*70)



            # # (WO_MVRM_LQ) LQ FORWARD PASS
            # print("LQ FORWARD PASS")
            # # lq_encoder_out, _ = self._run_model_forward(
            # lq_encoder_out = self._run_model_forward(
            #                                 lq_imgs, 
            #                                 ex_t_norm, 
            #                                 in_t, 
            #                                 export_feat_layers, 
            #                                 infer_gs, 
            #                                 use_ray_pose, 
            #                                 ref_view_strategy, 
            #                                 mvrm_cfg=None, 
            #                                 mvrm_result=None, 
            #                                 mode=None,
            #                                 ref_b_idx=None
            #                             )
            # # lq_pred_pose = lq_encoder_out['extrinsics']     # 1 v 3 4
            # # lq_depth = lq_encoder_out.depth.unsqueeze(2)     # 1 v 1 h w
            # # lq_ref_b_idx = lq_encoder_out.ref_b_idx
            lq_ref_b_idx = None
            
            
            
            # (WO_MVRM_LQ) HQ FORWARD PASS
            print("HQ FORWARD PASS")
            # hq_encoder_out, _ = self._run_model_forward(
            hq_encoder_out = self._run_model_forward(
                                            # imgs,
                                            vggt_imgs, 
                                            ex_t_norm, 
                                            in_t, 
                                            export_feat_layers, 
                                            infer_gs, 
                                            use_ray_pose, 
                                            ref_view_strategy, 
                                            mvrm_cfg=None, 
                                            mvrm_result=None, 
                                            mode=None,
                                            ref_b_idx=lq_ref_b_idx
                                            # ref_b_idx=None
                                        )
            hq_pred_pose = hq_encoder_out['extrinsics']     # 1 v 3 4
            # hq_depth = hq_encoder_out.depth.unsqueeze(2)    # 1 v 1 h w
            hq_depth = hq_encoder_out['depth'].permute(0,1,4,2,3)    # 1 v 1 h w
            
            
            
            # (WO_MVRM_LQ) LQ FORWARD PASS
            print("LQ FORWARD PASS")
            # raw_output, _ = self._run_model_forward(
            raw_output = self._run_model_forward(
                                            # lq_imgs, 
                                            vggt_lq_imgs,
                                            ex_t_norm, 
                                            in_t, 
                                            export_feat_layers, 
                                            infer_gs, 
                                            use_ray_pose, 
                                            ref_view_strategy, 
                                            mvrm_cfg=None, 
                                            mvrm_result=None, 
                                            mode=None
                                        )
            res_pred_pose = raw_output['extrinsics']     # 1 v 3 4
            # res_depth = raw_output.depth.unsqueeze(2)    # 1 v 1 h w
            res_depth = raw_output['depth'].permute(0,1,4,2,3)    # 1 v 1 h w
            
            
            # same for lq
            lq_encoder_out=raw_output
            lq_pred_pose = res_pred_pose
            lq_depth = res_depth
            
            scene = scene.replace('/', '_') if '/' in scene else scene
            
            vis_save_root = os.path.join(cfg.workspace.work_dir, 'pho_vis_results', data, pose_setting)
            vis_all(
                vis_save_root=vis_save_root,
                scene=scene,
                hq_img=imgs[0],
                lq_img=lq_imgs[0],
                hq_depth=hq_depth[0],
                lq_depth=lq_depth[0],
                res_depth=res_depth[0],
            )
            metric_save_root = os.path.join(cfg.workspace.work_dir, 'pho_metric_results', data, pose_setting)
            metric_all(
                metric_save_root=metric_save_root,
                scene=scene,
                poses = (hq_pred_pose[0], lq_pred_pose[0],res_pred_pose[0]),
                depths = (hq_depth, lq_depth, res_depth)
            )
            featsim_log = featsim_all(hq_encoder_out, lq_encoder_out, raw_output)
            featsim_save_root = os.path.join(cfg.workspace.work_dir, 'pho_featsim_results', data, pose_setting)
            plot_three_similarity_panels(
                featsim_log,
                save_path=f"{featsim_save_root}/{scene}_sim_all_combined.png"
            )
            cam_save_root = os.path.join(cfg.workspace.work_dir, 'pho_cam_traj_results', data, pose_setting)
            # plot_cam_trajectory(hq_pred_pose[0], lq_pred_pose[0], res_pred_pose[0], visualize_direction=False, save_path=f"{cam_save_root}/{scene}.png")
            plot_cam_trajectory_fair(hq_pred_pose[0], lq_pred_pose[0], res_pred_pose[0], visualize_direction=False, save_path=f"{cam_save_root}/fair_{scene}.png")






        elif cfg.MVRM_EVAL.eval_method == 'wo_mvrm_HQ':       
            print('-'*70)
            print('WO_MVRM_HQ')
            print('-'*70)


            # (WO_MVRM_HQ) HQ FORWARD PASS
            print("HQ FORWARD PASS")
            # raw_output, _ = self._run_model_forward(
                
            raw_output = self._run_model_forward(
                                            # imgs,
                                            vggt_imgs,
                                            ex_t_norm, 
                                            in_t, 
                                            export_feat_layers, 
                                            infer_gs, 
                                            use_ray_pose, 
                                            ref_view_strategy, 
                                            mvrm_cfg=None, 
                                            mvrm_result=None, 
                                            mode=None
                                        )
            
            # save_image(imgs.squeeze(0), 'img.jpg', normalize=True)
            # save_image(raw_output['depth'].squeeze(0).permute(0,3,1,2), 'img_depth.jpg')
            
            
            hq_encoder_out = raw_output
            hq_pred_pose = raw_output['extrinsics']     # 1 v 3 4
            hq_depth = raw_output['depth'].permute(0,1,4,2,3)    # 1 v 1 h w
            
            lq_imgs = imgs
            lq_encoder_out = raw_output
            lq_pred_pose = hq_pred_pose
            lq_depth = hq_depth
            
            res_pred_pose = hq_pred_pose
            res_depth = hq_depth            
            
            scene = scene.replace('/', '_') if '/' in scene else scene
            
            vis_save_root = os.path.join(cfg.workspace.work_dir, 'pho_vis_results', data, pose_setting)
            vis_all(
                vis_save_root=vis_save_root,
                scene=scene,
                hq_img=imgs[0],
                lq_img=lq_imgs[0],
                hq_depth=hq_depth[0],
                lq_depth=lq_depth[0],
                res_depth=res_depth[0],
            )
            metric_save_root = os.path.join(cfg.workspace.work_dir, 'pho_metric_results', data, pose_setting)
            metric_all(
                metric_save_root=metric_save_root,
                scene=scene,
                poses = (hq_pred_pose[0], lq_pred_pose[0],res_pred_pose[0]),
                depths = (hq_depth, lq_depth, res_depth)
            )
            featsim_log = featsim_all(hq_encoder_out, lq_encoder_out, raw_output)
            featsim_save_root = os.path.join(cfg.workspace.work_dir, 'pho_featsim_results', data, pose_setting)
            plot_three_similarity_panels(
                featsim_log,
                save_path=f"{featsim_save_root}/{scene}_sim_all_combined.png"
            )
            cam_save_root = os.path.join(cfg.workspace.work_dir, 'pho_cam_traj_results', data, pose_setting)
            # plot_cam_trajectory(hq_pred_pose[0], lq_pred_pose[0], res_pred_pose[0], visualize_direction=False, save_path=f"{cam_save_root}/{scene}.png")
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

            # breakpoint()
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
        
        # PHO - manual false for single gpu inference as it makes it slow
        need_sync=False
        
        if need_sync:
            torch.cuda.synchronize(device)
        start_time = time.time()
        feat_layers = list(export_feat_layers) if export_feat_layers is not None else None
        output = self.forward(imgs, ex_t, in_t, feat_layers, infer_gs, use_ray_pose, ref_view_strategy, mvrm_cfg, mvrm_result, mode, ref_b_idx, front_connect_back_mvrm_cfg, analysis, export_rgb_feat_layers)
        # output, mvrm_out = self.forward(imgs, ex_t, in_t, feat_layers, infer_gs, use_ray_pose, ref_view_strategy, mvrm_cfg, mvrm_result, mode, ref_b_idx, front_connect_back_mvrm_cfg, analysis, export_rgb_feat_layers)
        if need_sync:
            torch.cuda.synchronize(device)
        end_time = time.time()
        logger.info(f"Model Forward Pass Done. Time: {end_time - start_time} seconds")
        # return output, mvrm_out
        return output

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
