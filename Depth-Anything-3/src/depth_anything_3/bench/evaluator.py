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
Main Evaluator class for DepthAnything3 benchmark evaluation.

Supports multiple datasets and evaluation modes:
- pose: Camera pose estimation (AUC metrics)
- recon_unposed: 3D reconstruction with predicted poses
- recon_posed: 3D reconstruction with GT poses
- view_syn: Novel view synthesis (TODO)
"""

import json
import os
import random
import shutil
from typing import Dict as TDict, Iterable, List

import numpy as np
import torch
from addict import Dict
from tqdm import tqdm

from depth_anything_3.bench.print_metrics import MetricsPrinter
from depth_anything_3.utils.parallel_utils import parallel_execution
from depth_anything_3.bench.registries import MV_REGISTRY
from depth_anything_3.utils.constants import EVAL_REF_VIEW_STRATEGY
from PIL import Image

import math
from RAE.src.stage2.transport import create_transport, Sampler
from RAE.src import initialize
from RAE.src.stage2.models import Stage2ModelProtocol
from RAE.src.utils.model_utils import instantiate_from_config
from depth_anything_3.bench.depth_metrics import (
    DepthEvalConfig,
    compute_depth_metrics,
    resize_depth_nearest
)

DEBLUR_IMG_EXTS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')

BENCH_LAYOUTS = {
    # Deblur-NeRF's real captured-blur scenes have no sharp/GT counterpart —
    # only blurry captures + COLMAP poses (see _collect_scene_folder_no_gt_sequences).
    # Inference-only / qualitative: no PSNR/SSIM/LPIPS or pseudo-GT geo-eval.
    'deblur_nerf_bench': dict(layout='scene_folder_no_gt', key='deblur_nerf',
                               image_subdir='images_4'),
}


class Evaluator:
    """
    Main evaluation orchestrator for DepthAnything3 benchmarks.

    Usage:
        evaluator = Evaluator(
            work_dir="./eval_workspace",
            datas=["dtu"],
            modes=["pose", "recon_unposed", "recon_posed"],
        )
        api = DepthAnything3.from_pretrained("...")
        evaluator.infer(api)
        metrics = evaluator.eval()
        evaluator.print_metrics()
    """

    VALID_MODES = {"pose", "recon_unposed", "recon_posed", "view_syn", 'depth'}

    def __init__(
        self,
        work_dir: str = "./eval_workspace",
        datas: List[str] = ("dtu",),
        modes: List[str] = ("recon_unposed",),
        ref_view_strategy: str = EVAL_REF_VIEW_STRATEGY,
        scenes: List[str] = None,
        debug: bool = False,
        num_fusion_workers: int = 4,
        max_frames: int = 100,
        gpu_id: int = 0,
        total_gpus: int = 1,
        full_cfg=None
    ):
        """
        Initialize the evaluator.

        Args:
            work_dir: Base directory for model outputs and metric files
            datas: List of dataset names (must be registered in MV_REGISTRY)
            modes: List of evaluation modes to run
            ref_view_strategy: Reference view selection strategy for inference
                               ("first", "saddle_balanced", etc.)
            scenes: Specific scenes to evaluate (None = all scenes)
            debug: Enable verbose debug output
            num_fusion_workers: Number of parallel workers for TSDF fusion (default: 4)
            max_frames: Maximum number of frames per scene (default: 100).
                        If a scene has more frames, randomly sample to this limit.
                        Set to -1 to disable sampling.
            gpu_id: GPU index for multi-GPU (0-indexed)
            total_gpus: Total number of GPUs for task distribution
        """
        self.work_dir = work_dir
        self.datas = list(datas)
        self.modes = set(modes)
        self.ref_view_strategy = ref_view_strategy
        self.scenes_filter = scenes
        self.debug = debug
        self.num_fusion_workers = num_fusion_workers
        self.max_frames = max_frames
        self.gpu_id = gpu_id
        self.total_gpus = total_gpus
        
        
        self.full_cfg = full_cfg
        

        # Validate modes
        unknown = self.modes - self.VALID_MODES
        if unknown:
            raise ValueError(f"Unknown modes: {unknown}. Valid: {sorted(self.VALID_MODES)}")

        os.makedirs(self.work_dir, exist_ok=True)

        # Initialize datasets
        self.datasets = Dict()
        for data in self.datas:
            if not MV_REGISTRY.has(data):
                available = list(MV_REGISTRY.all().keys())
                raise ValueError(f"Dataset '{data}' not found. Available: {available}")
            self.datasets[data] = MV_REGISTRY.get(data)()


        # Initialize metrics printer
        self._printer = MetricsPrinter()
        
        
        if full_cfg.MVRM_EVAL.eval_method == 'w_mvrm':
            time_dist_shift = math.sqrt(full_cfg.misc.time_dist_shift_dim / full_cfg.misc.time_dist_shift_base)
            # load Transport 
            transport = create_transport(**full_cfg.transport.params, time_dist_shift=time_dist_shift,)
            transport_sampler = Sampler(transport)
            # load sampler 
            self.eval_sampler = initialize.load_sampler(full_cfg, transport_sampler)
            # load denoiser 
            self.denoiser: Stage2ModelProtocol = instantiate_from_config(full_cfg.denoiser)  
            self.denoiser = self.denoiser.eval()

            # if torch.__version__ >= "2.0":
            #     self.denoiser = torch.compile(self.denoiser)
        
        elif full_cfg.MVRM_EVAL.eval_method == 'w_mvrm_front_back' or full_cfg.MVRM_EVAL.eval_method == 'w_mvrm_front_connect_back':

            time_dist_shift = math.sqrt(full_cfg.misc.time_dist_shift_dim / full_cfg.misc.time_dist_shift_base)
            # load Transport 
            transport = create_transport(**full_cfg.transport.params, time_dist_shift=time_dist_shift,)
            transport_sampler = Sampler(transport)
            # load sampler 
            self.eval_sampler = initialize.load_sampler(full_cfg, transport_sampler)
            # load denoiser 
            self.denoiser: Stage2ModelProtocol = instantiate_from_config(full_cfg.denoiser)  
            self.denoiser = self.denoiser.eval()
            
            # load denoiser2 
            self.denoiser2: Stage2ModelProtocol = instantiate_from_config(full_cfg.denoiser2)  
            self.denoiser2 = self.denoiser2.eval()
            
            
                
        else:
            self.eval_sampler = None 
            self.denoiser = None 
            
        
        
        
        
        
        self.rgb_dec_cfg = full_cfg.MVRM_EVAL.get('rgb_decoder', None)    
        if self.rgb_dec_cfg is not None:
            if self.rgb_dec_cfg.stage_1.params.dpt_model_type == 'da3-base':
                self.rae = instantiate_from_config(self.rgb_dec_cfg.stage_1)
                self.rae.eval()
            
            elif self.rgb_dec_cfg.stage_1.params.dpt_model_type == 'da3-giant':
                pass
                
        
    
        


    # -------------------- Public APIs -------------------- #

    def all(self, api) -> TDict[str, dict]:
        """
        Run complete evaluation pipeline: inference + evaluation.

        Args:
            api: DepthAnything3 API instance

        Returns:
            Combined metrics dictionary
        """
        self.infer(api)
        return self.eval()

    def _get_scenes(self, dataset) -> List[str]:
        """Get list of scenes to evaluate, optionally filtered."""
        all_scenes = dataset.SCENES
        if self.scenes_filter:
            scenes = [s for s in all_scenes if s in self.scenes_filter]
            if self.debug:
                print(f"[DEBUG] Filtered scenes: {scenes} (from {len(all_scenes)} total)")
            return scenes
        return all_scenes

    def infer(self, api, model_path, noise_generator, device) -> None:
        """
        Run inference according to requested modes.

        - Unposed export if 'pose' or 'recon_unposed' is in modes
        - Posed export if 'recon_posed' or 'view_syn' is in modes

        Multi-GPU: Use --gpu_id and --total_gpus to distribute tasks.
        Example: Launch 4 processes with gpu_id=0,1,2,3 and total_gpus=4

        Args:
            api: DepthAnything3 API instance
            model_path: Model path (unused, kept for API compatibility)
        """
        
        need_unposed = {"pose", "recon_unposed", "depth"} & self.modes
        need_posed = {"recon_posed", "view_syn"} & self.modes
        export_format = "mini_npz-glb" if self.debug else "mini_npz"    # mini_npz


        # Collect all tasks
        all_tasks = []
        for data in self.datas:
            dataset = self.datasets[data]
            for scene in self._get_scenes(dataset):
                all_tasks.append((data, scene))
        
        
        '''
        (Pdb) all_tasks[:10]
            [('eth3d', 'courtyard'), ('eth3d', 'electro'), ('eth3d', 'kicker'), ('eth3d', 'pipes'), ('eth3d', 'relief'), ('eth3d', 'delivery_area'), ('eth3d', 'facade'), ('eth3d', 'office'), ('eth3d', 'playground'), ('eth3d', 'relief_2')]
        '''

        # Distribute tasks across GPUs
        if self.total_gpus > 1: # f
            tasks = [t for i, t in enumerate(all_tasks) if i % self.total_gpus == self.gpu_id]
            print(f"[INFO] GPU {self.gpu_id}/{self.total_gpus}: {len(tasks)}/{len(all_tasks)} tasks")
        else:   # t
            tasks = all_tasks
            print(f"[INFO] Total inference tasks: {len(tasks)}")
        
        
        if self.full_cfg.MVRM_EVAL.eval_method == 'w_mvrm':
            self.denoiser = self.denoiser.to(device)
            self.denoiser2 = None

        elif self.full_cfg.MVRM_EVAL.eval_method == 'w_mvrm_front_back' or self.full_cfg.MVRM_EVAL.eval_method == 'w_mvrm_front_connect_back':
            self.denoiser = self.denoiser.to(device) 
            self.denoiser2 = self.denoiser2.to(device)
        
        else:
            self.denoiser = None 
            self.denoiser2 = None
            
        
        
        
        # load rgb decoder
        if self.rgb_dec_cfg is not None:
            self.rgb_decoder = self.rae.mae_decoder.to(device)
                        
            if getattr(self.rae, 'adapter', None) is not None:
                self.proj_adapter = self.rae.adapter.to(device)
            else:
                self.proj_adapter = None
            
                
        else:
            self.rgb_decoder = None
            self.proj_adapter = None
            
            
            
        eval_deblur_bench = self.full_cfg.MVRM_EVAL.get('eval_deblur_bench', False)


        if eval_deblur_bench:
            # Use npz (not mini_npz) so predicted images are saved for PSNR/SSIM/LPIPS eval
            export_format = "npz"

            # Which models to actually run inference for, within each selected bench.
            # 'ours' = the own MVRM model; other names are keys from that bench's
            # `model:` dict (e.g. any external baseline restoration methods you
            # want to compare against).
            # Opt-in, same convention as to_eval_deblur_bench: empty/unset = none.
            to_eval_model = self.full_cfg.MVRM_EVAL.get('to_eval_model', None)
            selected_models = set(to_eval_model or [])

            for deblur_ds, sequences, restored_models_for_ds in self._expand_deblur_bench_config():
                deblur_num_seqs = self.full_cfg.MVRM_EVAL.get('deblur_num_seqs', None)
                if deblur_num_seqs is not None:
                    sequences = sequences[:deblur_num_seqs]
                print(f'------------ {deblur_ds.upper()} DEBLUR BENCH: {len(sequences)} sequences ------------ ')

                # model_results/ only ever gets read back for sequences that have GT
                # (see the gt_meta_path gate in _eval_deblur_bench below), so skip
                # creating it at all when this dataset has no GT sequences.
                ds_has_any_gt = any(gt_seq is not None for _, _, gt_seq in sequences)

                if ds_has_any_gt:
                    # Clear the dataset-level summary so re-runs don't append duplicates
                    ds_summary_path = os.path.join(
                        self.work_dir, 'model_results', deblur_ds, 'selected_images_all.txt')
                    os.makedirs(os.path.dirname(ds_summary_path), exist_ok=True)
                    open(ds_summary_path, 'w').close()

                for seq_name, lq_seq, gt_seq in sequences:
                    N_seq = len(lq_seq)
                    # gt_seq is None for GT-free benches (e.g. deblur_nerf_bench's
                    # real captured-blur scenes, see _collect_scene_folder_no_gt_sequences)
                    # — inference still runs, but no deblur_gt_files.npz gets written,
                    # so PSNR/SSIM/LPIPS/geo-eval stay a no-op for those sequences and
                    # only qualitative exports are produced.
                    has_gt = gt_seq is not None

                    # If the sequence is longer than max_frames, take the center window
                    # to maximise overlap while staying within memory
                    if self.max_frames > 0 and N_seq > self.max_frames:
                        start = (N_seq - self.max_frames) // 2
                        lq_seq = lq_seq[start:start + self.max_frames]
                        if has_gt:
                            gt_seq = gt_seq[start:start + self.max_frames]

                    n = len(lq_seq)
                    dummy_ext = np.eye(4, dtype=np.float32)[None].repeat(n, axis=0)
                    dummy_ixt = np.eye(3, dtype=np.float32)[None].repeat(n, axis=0)

                    if 'ours' in selected_models:
                        scene_data = Dict()
                        scene_data.lq_image_files = lq_seq
                        # No separate HQ reference without GT — echo the LQ images
                        # themselves as a placeholder so the model still gets an
                        # image_files input (only used for the diagnostic HQ pass /
                        # vis_all comparisons, not for the exported prediction, which
                        # is overridden by the model's real output under w_mvrm).
                        scene_data.image_files = gt_seq if has_gt else lq_seq
                        scene_data.extrinsics = dummy_ext
                        scene_data.intrinsics = dummy_ixt
                        scene_data.aux = Dict()

                        print(f'  [{deblur_ds}] {seq_name}: {n} frames')
                        # model_results/ (the npz prediction export + its sidecar files
                        # below) only ever gets read back by _eval_deblur_bench for
                        # sequences that have GT (gated on gt_meta_path there), so skip
                        # it entirely for GT-free sequences — the qualitative outputs
                        # (vis_rgb_restored_results, vis_depth_results) are unaffected,
                        # since those key off cfg.workspace.work_dir, not export_dir.
                        export_dir = self._export_dir(deblur_ds, seq_name, posed=False) if has_gt else None
                        api.inference(
                            scene_data,
                            export_dir=export_dir,
                            export_format=export_format,
                            ref_view_strategy=self.ref_view_strategy,
                            eval_sampler=self.eval_sampler,
                            denoiser=self.denoiser,
                            denoiser2=self.denoiser2,
                            noise_generator=noise_generator,
                            cfg=self.full_cfg,
                            # data (first element) becomes a path segment for every pho_*_results/
                            # folder in api.py, so this nests each model's outputs separately
                            # instead of mixing own/pseudo_gt/baseline results in one shared directory.
                            scene_info=(f'{deblur_ds}/ours', seq_name),
                            use_pose=False,
                            use_ray_pose=self.full_cfg.model.use_ray_pose,
                            rgb_decoder=self.rgb_decoder,
                            proj_adapter=self.proj_adapter,
                            has_gt=has_gt
                        )
                        if has_gt:
                            # api.inference()'s npz export runs asynchronously (@async_call),
                            # so it isn't guaranteed to have created export_dir/exports/ yet —
                            # create it here unconditionally before writing either sidecar file.
                            exports_dir = os.path.join(export_dir, 'exports')
                            os.makedirs(exports_dir, exist_ok=True)
                            # Save GT file list for this sequence so eval can load it
                            meta_path = os.path.join(exports_dir, 'deblur_gt_files.npz')
                            np.savez_compressed(meta_path, gt_files=np.array(gt_seq, dtype=object))

                            # Per-sequence text file: one image basename per line
                            seq_txt = os.path.join(exports_dir, 'selected_images.txt')
                            with open(seq_txt, 'w') as f:
                                for p in gt_seq:
                                    f.write(os.path.basename(p) + '\n')

                            # Append to per-dataset summary file (one file covers all sequences)
                            ds_summary_path = os.path.join(
                                self.work_dir, 'model_results', deblur_ds, 'selected_images_all.txt')
                            with open(ds_summary_path, 'a') as f:
                                f.write(f'# {seq_name}\n')
                                for p in gt_seq:
                                    f.write(os.path.basename(p) + '\n')

                    # Run inference on GT clean images to obtain pseudo-GT geometry
                    # (pose, depth, 3D) for geometric consistency evaluation. Needed by
                    # every selected model's geo eval (own or restored), so it runs
                    # whenever at least one model is selected, not just 'ours'. Skipped
                    # entirely for GT-free benches — there's no clean reference image to
                    # derive pseudo-GT geometry from.
                    if selected_models and has_gt:
                        pseudo_gt_scene_data = Dict()
                        pseudo_gt_scene_data.lq_image_files = gt_seq
                        pseudo_gt_scene_data.image_files = gt_seq
                        pseudo_gt_scene_data.extrinsics = dummy_ext
                        pseudo_gt_scene_data.intrinsics = dummy_ixt
                        pseudo_gt_scene_data.aux = Dict()

                        pseudo_gt_export_dir = os.path.join(
                            self.work_dir, 'model_results', deblur_ds, seq_name, 'pseudo_gt')
                        print(f'  [{deblur_ds}] {seq_name}: pseudo-GT inference on {n} clean frames')

                        # Pseudo-GT geometry must be identical regardless of this run's
                        # configured eval_method (w_mvrm for 'ours' runs, wo_mvrm_LQ for
                        # baseline-model runs, etc.) — otherwise 'ours' and baseline geo-
                        # metrics, computed in separate runs/work_dirs against their own
                        # pseudo_gt, would be scored against different reference geometry.
                        # Force a fixed, deterministic, MVRM-free forward pass for this
                        # call only.
                        _orig_eval_method = self.full_cfg.MVRM_EVAL.eval_method
                        self.full_cfg.MVRM_EVAL.eval_method = 'wo_mvrm_HQ'
                        try:
                            api.inference(
                                pseudo_gt_scene_data,
                                export_dir=pseudo_gt_export_dir,
                                export_format=export_format,
                                ref_view_strategy=self.ref_view_strategy,
                                eval_sampler=self.eval_sampler,
                                denoiser=self.denoiser,
                                denoiser2=self.denoiser2,
                                noise_generator=noise_generator,
                                cfg=self.full_cfg,
                                scene_info=(f'{deblur_ds}/pseudo_gt', seq_name),
                                use_pose=False,
                                use_ray_pose=self.full_cfg.model.use_ray_pose,
                                rgb_decoder=self.rgb_decoder,
                                proj_adapter=self.proj_adapter
                            )
                        finally:
                            self.full_cfg.MVRM_EVAL.eval_method = _orig_eval_method

                    # Run inference on pre-restored images from external deblurring models
                    # (restored_models_for_ds is already filtered to selected_models, see
                    # _expand_deblur_bench_config)
                    for model_name, restored_dir in restored_models_for_ds.items():
                        # under <restored_dir>/<seq_name>/, unlike RealBlur's flat layout
                        # where all sequences' restored images sit directly in restored_dir.
                        seq_restored_dir = os.path.join(restored_dir, seq_name)
                        if os.path.isdir(seq_restored_dir):
                            restored_dir = seq_restored_dir
                        if not os.path.isdir(restored_dir):
                            print(f'  [{deblur_ds}] {model_name}: restored dir not found: {restored_dir}')
                            continue
                        # Match LQ basenames to restored images (handles extension mismatch)
                        restored_seq = []
                        matched_gt_seq = [] if has_gt else None
                        gt_iter = gt_seq if has_gt else [None] * len(lq_seq)
                        for lq_path, gt_path in zip(lq_seq, gt_iter):
                            bn   = os.path.basename(lq_path)
                            stem = os.path.splitext(bn)[0]
                            candidate = os.path.join(restored_dir, bn)
                            if not os.path.exists(candidate):
                                for ext in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG'):
                                    alt = os.path.join(restored_dir, stem + ext)
                                    if os.path.exists(alt):
                                        candidate = alt
                                        break
                                else:
                                    continue
                            restored_seq.append(candidate)
                            if has_gt:
                                matched_gt_seq.append(gt_path)
                        if not restored_seq:
                            print(f'  [{deblur_ds}] {model_name}/{seq_name}: no matching restored images in {restored_dir}')
                            continue
                        n_r = len(restored_seq)
                        dummy_ext_r = np.eye(4, dtype=np.float32)[None].repeat(n_r, axis=0)
                        dummy_ixt_r = np.eye(3, dtype=np.float32)[None].repeat(n_r, axis=0)

                        restored_scene_data = Dict()
                        restored_scene_data.lq_image_files = restored_seq
                        # No GT to pair against — echo the restored images themselves
                        # as the placeholder HQ field (see 'ours' branch above).
                        restored_scene_data.image_files = matched_gt_seq if has_gt else restored_seq
                        restored_scene_data.extrinsics = dummy_ext_r
                        restored_scene_data.intrinsics = dummy_ixt_r
                        restored_scene_data.aux = Dict()

                        # see the 'ours' branch above for why this is skipped without GT
                        restored_export_dir = os.path.join(
                            self.work_dir, 'model_results', deblur_ds, model_name, seq_name, 'unposed') if has_gt else None
                        if has_gt:
                            meta_path = os.path.join(restored_export_dir, 'exports', 'deblur_gt_files.npz')
                            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                            np.savez_compressed(meta_path, gt_files=np.array(matched_gt_seq, dtype=object))

                        print(f'  [{deblur_ds}] {model_name}/{seq_name}: {n_r} restored frames')
                        api.inference(
                            restored_scene_data,
                            export_dir=restored_export_dir,
                            export_format=export_format,
                            ref_view_strategy=self.ref_view_strategy,
                            eval_sampler=self.eval_sampler,
                            denoiser=self.denoiser,
                            denoiser2=self.denoiser2,
                            noise_generator=noise_generator,
                            cfg=self.full_cfg,
                            scene_info=(f'{deblur_ds}/{model_name}', seq_name),
                            use_pose=False,
                            use_ray_pose=self.full_cfg.model.use_ray_pose,
                            rgb_decoder=self.rgb_decoder,
                            proj_adapter=self.proj_adapter,
                            has_gt=has_gt,
                        )



        else:

            for data, scene in tqdm(tasks, desc=f"Inference (GPU {self.gpu_id})"):
                
                dataset = self.datasets[data]
                # breakpoint()
                scene_data = dataset.get_data(scene) 
                scene_data = self._sample_frames(scene_data, scene)
                
                # for img_path in scene_data.lq_image_files:
                    
                #     # extract folder name 
                #     folder_name = img_path.split('/')[5]
                #     new_path = img_path.replace(f"/{folder_name}/", f"/filtered_{folder_name}/")

                    
                #     # new_path = img_path.replace("/cam_blur_100_resize_640/", "/filtered_cam_blur_100_resize_640/")
                #     # new_path = img_path.replace("/cam_blur_300_resize_640/", "/filtered_cam_blur_300_resize_640/")
                #     # new_path = img_path.replace("/cam_blur_500_resize_640/", "/filtered_cam_blur_500_resize_640/")
                    
                #     # new_path = img_path.replace("/cam_blur_50_resize_640/", "/filtered_cam_blur_50_resize_640/")
                #     # new_path = img_path.replace("/cam_blur_70_resize_640/", "/filtered_cam_blur_70_resize_640/")
                #     # new_path = img_path.replace("/cam_blur_120_resize_640/", "/filtered_cam_blur_120_resize_640/")
                #     # new_path = img_path.replace("/cam_blur_150_resize_640/", "/filtered_cam_blur_150_resize_640/")
                    
                #     # new_path = img_path.replace("/cam_blur_400/", "/filtered_cam_blur_400/")
                #     # new_path = img_path.replace("/cam_blur_500/", "/filtered_cam_blur_500/")
                #     # new_path = img_path.replace("/cam_blur_600/", "/filtered_cam_blur_600/")
                    
                #     # new_path = img_path.replace("/cam_blur_200/", "/filtered_cam_blur_200/")
                #     # new_path = img_path.replace("/cam_blur_150/", "/filtered_cam_blur_150/")
                #     # new_path = img_path.replace("/cam_blur_250/", "/filtered_cam_blur_250/")
                #     # new_path = img_path.replace("/cam_blur_700/", "/filtered_cam_blur_700/")
                #     # new_path = img_path.replace("/cam_blur_800/", "/filtered_cam_blur_800/")

                #     new_dir = os.path.dirname(new_path)
                #     os.makedirs(new_dir, exist_ok=True)
                #     # Copy image
                #     shutil.copy2(img_path, new_path)  # copy2 preserves metadata
                # continue 
            
                

            
                # for img_path in scene_data.image_files:
                #     new_path = img_path.replace("/clean/", "/filtered_clean/")
                #     # new_path = img_path.replace("/cam_blur_50/", "/filtered_cam_blur_50/")
                #     # new_path = img_path.replace("/cam_blur_100/", "/filtered_cam_blur_100/")
                #     # new_path = img_path.replace("/cam_blur_300/", "/filtered_cam_blur_300/")
                #     new_dir = os.path.dirname(new_path)
                #     os.makedirs(new_dir, exist_ok=True)
                #     # Copy image
                #     shutil.copy2(img_path, new_path)  # copy2 preserves metadata
                # continue 
                
                if self.full_cfg.model.use_ray_pose:
                    print('=========== USING RAY POSE ===========')
                else:
                    print('=========== USING CAMERA POSE ===========')
                    
                
                
                    
                if self.rgb_dec_cfg is not None:
                    print('--- LOADED RGB DECODER ---')
                
                

                if need_unposed:    # t
                    print('------------ UNPOSED!! ------------ ')
                    export_dir = self._export_dir(data, scene, posed=False)
                    
                    api.inference(
                        scene_data,
                        export_dir=export_dir,             
                        export_format=export_format,                # export_format: 'mini_npz-glb'
                        ref_view_strategy=self.ref_view_strategy,
                        eval_sampler=self.eval_sampler,
                        denoiser=self.denoiser,
                        denoiser2 = self.denoiser2,
                        noise_generator=noise_generator,
                        cfg=self.full_cfg,
                        scene_info = (data,scene),
                        use_pose=False,
                        use_ray_pose = self.full_cfg.model.use_ray_pose,
                        rgb_decoder = self.rgb_decoder,
                        proj_adapter = self.proj_adapter,
                        device=device
                    )
                    # breakpoint()
                    self._save_gt_meta(export_dir, scene_data)

                if need_posed:
                    print('------------ POSED!! ------------ ')
                    export_dir = self._export_dir(data, scene, posed=True)
                    api.inference(
                        scene_data,
                        scene_data.extrinsics,      # provide extrinsics
                        scene_data.intrinsics,      # provide intrinsics
                        export_dir=export_dir,
                        export_format=export_format,
                        ref_view_strategy=self.ref_view_strategy,
                        eval_sampler=self.eval_sampler,
                        denoiser=self.denoiser,
                        denoiser2 = self.denoiser2,
                        noise_generator=noise_generator,
                        cfg=self.full_cfg,
                        scene_info = (data,scene),
                        use_pose=True,
                        use_ray_pose = self.full_cfg.model.use_ray_pose,
                        rgb_decoder = self.rgb_decoder,
                        proj_adapter = self.proj_adapter,
                        device=device
                    )
                    self._save_gt_meta(export_dir, scene_data)
            
        # breakpoint()
            
    def eval(self) -> TDict[str, dict]:
        """
        Evaluate for all configured modes and write JSON files.
        
        Evaluation order by mode (all datasets per mode):
        1. pose - all datasets
        2. recon_unposed - all datasets
        3. recon_posed - all datasets

        Returns:
            Summary mapping: {"<data>_<mode>": metrics_dict}
        """
        summary: TDict[str, dict] = {}

        eval_deblur_bench = self.full_cfg.MVRM_EVAL.get('eval_deblur_bench', False)

        # Regular DA3 benchmark evaluation — skipped when eval_deblur_bench is active
        # because inference only ran on deblur datasets in that case
        if not eval_deblur_bench:
            if "pose" in self.modes:
                print(f"\n{'='*60}")
                print(f"📊 Evaluating POSE for all datasets...")
                print(f"{'='*60}")
                for data, result in self._eval_pose():
                    summary[f"{data}_pose"] = result

            if "depth" in self.modes:
                print(f"\n{'='*60}")
                print(f"📊 Evaluating DEPTH for all datasets...")
                print(f"{'='*60}")
                summary.update(self._eval_depth(mode="depth"))

            if "recon_unposed" in self.modes:
                print(f"\n{'='*60}")
                print(f"📊 Evaluating RECON_UNPOSED for all datasets...")
                print(f"{'='*60}")
                for data, result in self._eval_reconstruction("recon_unposed"):
                    summary[f"{data}_recon_unposed"] = result

            if "recon_posed" in self.modes:
                print(f"\n{'='*60}")
                print(f"📊 Evaluating RECON_POSED for all datasets...")
                print(f"{'='*60}")
                for data, result in self._eval_reconstruction("recon_posed"):
                    summary[f"{data}_recon_posed"] = result

            if "view_syn" in self.modes:
                print(f"\n{'='*60}")
                print(f"📊 Evaluating VIEW_SYN for all datasets...")
                print(f"{'='*60}")
                summary.update(self._eval_view_syn())

        if eval_deblur_bench:
            print(f"\n{'='*60}")
            print(f"📊 Evaluating DEBLUR BENCH...")
            print(f"{'='*60}")
            deblur_metrics = self._eval_deblur_bench()
            summary.update({f'{k}_deblur': v for k, v in deblur_metrics.items()})

        return summary

    def _eval_depth(self, mode: str) -> dict[str, dict]:
        assert mode in {"depth"} 
        os.makedirs(self._metric_dir, exist_ok=True)

        depth_datasets = [d for d in self.datas if d != "dtu64"]
        eval_datasets = [d for d in depth_datasets]

        cfg = DepthEvalConfig(delta_thresholds=(1.25, 1.25**2, 1.25**3))

        all_metrics: dict[str, dict] = {}
        for data in eval_datasets:
            dataset = self.datasets[data]
            dataset_results = {}

            scenes = self._get_scenes(dataset)
            success = 0
            for scene in scenes:
                try:
                    out_dir = self._export_dir(data, scene, posed=False)
                    result_path = os.path.join(out_dir, "exports", "mini_npz", "results.npz")
                    if not os.path.exists(result_path):
                        raise FileNotFoundError(result_path)

                    pred_npz = np.load(result_path)
                    if "depth" not in pred_npz:
                        raise KeyError("results.npz missing 'depth'")
                    pred_depths = pred_npz["depth"]  # (N,H,W)

                    full_gt_data = dataset.get_data(scene)
                    scene_data = self._sample_frames(full_gt_data, scene) 

                    sampled_files = [str(p) for p in scene_data.image_files]
                    image_indices = [full_gt_data.image_files.index(f) for f in sampled_files]

                    per_img_metrics = []
                    for i, img_idx in enumerate(image_indices[: pred_depths.shape[0]]):
                        pred_depth = pred_depths[i].astype(np.float32)

                        gt_depth, gt_valid = self._load_gt_depth_and_mask(
                            data=data,
                            dataset=dataset,
                            scene=scene,
                            full_gt_data=full_gt_data,
                            img_idx=img_idx,
                        )

                        pred_depth = resize_depth_nearest(pred_depth, gt_depth.shape[:2])

                        m = compute_depth_metrics(pred_depth, gt_depth, gt_valid, cfg=cfg)
                        if not m:
                            continue
                        per_img_metrics.append(m)

                    if not per_img_metrics:
                        raise RuntimeError("No valid GT depth pixels for evaluation.")

                    scene_metrics = self._mean_of_dicts(per_img_metrics)
                    scene_metrics["num_views"] = float(len(per_img_metrics))
                    scene_metrics["num_eval_imgs"] = float(len(per_img_metrics))

                    dataset_results[scene] = scene_metrics
                    success += 1
                except Exception as e:
                    if self.debug:
                        print(f"[WARN] depth eval failed for {data}/{scene}: {e}")

            if success == 0:
                continue

            scene_vals = [v for k, v in dataset_results.items() if k != "mean"]
            dataset_results["mean"] = self._mean_of_dicts(scene_vals)
            dataset_results["success_rate"] = float(success) / float(len(scenes)) * 100.0
            dataset_results["_meta"] = {
                "mode": mode,
                "delta_thresholds": list(cfg.delta_thresholds),
                "note": "Depth metrics are abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3, "
                        "averaged per sampled image then per-scene.",
            }

            out_path = os.path.join(self._metric_dir, f"{data}_{mode}.json")
            self._dump_json(out_path, dataset_results)
            all_metrics[f"{data}_{mode}"] = dataset_results

        return all_metrics

    @staticmethod
    def _group_deblur_sequences(lq_files, gt_files, ds_name):
        """
        Group flat file lists into per-sequence lists based on filename conventions.

        GoPro:     GOPR0384_11_00-000001.png  →  seq = GOPR0384_11_00   (split on last '-')
        HIDE:      100fromGOPR0950.png        →  seq = GOPR0950          (part after 'from')
        RealBlur:  scene002-10.png            →  seq = scene002          (split on last '-')
        """
        import re
        from collections import defaultdict

        lq_by_seq = defaultdict(list)
        gt_by_seq = defaultdict(list)

        for lq, gt in zip(lq_files, gt_files):
            fname = os.path.basename(lq)
            if ds_name == 'hide':
                m = re.match(r'^\d+from(.+)\.(?:png|jpg|jpeg)$', fname, re.IGNORECASE)
                seq = m.group(1) if m else 'all'
            else:
                # Works for both GoPro (GOPR0384_11_00-000001.png)
                # and RealBlur (scene002-10.png)
                seq = fname.rsplit('-', 1)[0]
            lq_by_seq[seq].append(lq)
            gt_by_seq[seq].append(gt)

        return [
            (seq, sorted(lq_by_seq[seq]), sorted(gt_by_seq[seq]))
            for seq in sorted(lq_by_seq)
        ]

    @staticmethod
    def _collect_per_seq_folder_sequences(split_root, blur_subdir, sharp_subdir):
        sequences = []
        for seq_name in sorted(os.listdir(split_root)):
            lq_dir = os.path.join(split_root, seq_name, blur_subdir)
            gt_dir = os.path.join(split_root, seq_name, sharp_subdir)
            if not os.path.isdir(lq_dir) or not os.path.isdir(gt_dir):
                continue
            lq_files = sorted(os.path.join(lq_dir, f) for f in os.listdir(lq_dir) if f.endswith(DEBLUR_IMG_EXTS))
            gt_files = sorted(os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith(DEBLUR_IMG_EXTS))
            sequences.append((seq_name, lq_files, gt_files))
        return sequences

    @staticmethod
    def _collect_scene_folder_no_gt_sequences(data_path, image_subdir):
        """
        Collect sequences from a dataset with one subdirectory per scene and no
        ground-truth counterpart (e.g. Deblur-NeRF's real captured-blur scenes:
        <data_path>/<scene>/<image_subdir>/*, only blurry captures + COLMAP poses).

        gt_files is always None — infer() treats that as "no GT available" and
        skips writing deblur_gt_files.npz, pseudo-GT inference, and GT-pairing
        for restored-model matching, so these scenes only ever produce qualitative
        (inference-only) exports: _eval_deblur_bench()'s PSNR/SSIM/LPIPS/geo-eval
        already no-op gracefully whenever deblur_gt_files.npz is missing.
        """
        sequences = []
        for seq_name in sorted(os.listdir(data_path)):
            img_dir = os.path.join(data_path, seq_name, image_subdir)
            if not os.path.isdir(img_dir):
                continue
            lq_files = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(DEBLUR_IMG_EXTS))
            if not lq_files:
                continue
            sequences.append((seq_name, lq_files, None))
        return sequences

    def _expand_deblur_bench_config(self):
        """
        Expand every selected `*_bench` entry in MVRM_EVAL config (per BENCH_LAYOUTS)
        into a flat list of (ds_key, sequences, restored_models_for_ds), where
        sequences = [(seq_name, lq_files, gt_files), ...] and restored_models_for_ds
        = {model_name: restored_dir_for_this_ds_key}.

        Stateless / re-derived from config on every call so it stays correct whether
        called from infer() or from _eval_deblur_bench() running standalone
        (eval.eval_only=true, no infer() call in this process).

        to_eval_deblur_bench is opt-in: empty/unset selects NO bench (safe default,
        avoids accidentally running every configured benchmark + all its restored-model
        baselines). List the specific *_bench names you want to actually run.

        to_eval_model is likewise opt-in and filters restored_models_for_ds down to the
        selected baseline names (e.g. to_eval_model: [ours, baseline_a] keeps only
        baseline_a here; 'ours' itself isn't a `model:` entry, it gates the own-model
        call in infer()).
        """
        to_eval = self.full_cfg.MVRM_EVAL.get('to_eval_deblur_bench', None)
        bench_names = [b for b in BENCH_LAYOUTS if b in self.full_cfg.MVRM_EVAL and b in (to_eval or [])]
        to_eval_model = self.full_cfg.MVRM_EVAL.get('to_eval_model', None)
        selected_models = set(to_eval_model or [])

        out = []
        for bench_name in bench_names:
            bench_cfg = self.full_cfg.MVRM_EVAL[bench_name]
            spec = BENCH_LAYOUTS[bench_name]
            model_cfg = {k: v for k, v in bench_cfg.get('model', {}).items() if k in selected_models}

            if spec['layout'] == 'flat':
                lq_dir, gt_dir = bench_cfg.data.lq_path, bench_cfg.data.hq_path
                lq_files = sorted(os.path.join(lq_dir, f) for f in os.listdir(lq_dir) if f.endswith(DEBLUR_IMG_EXTS))
                gt_files = sorted(os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith(DEBLUR_IMG_EXTS))
                sequences = self._group_deblur_sequences(lq_files, gt_files, spec['key'])
                out.append((spec['key'], sequences, dict(model_cfg)))

            elif spec['layout'] == 'per_seq_folder':
                sequences = self._collect_per_seq_folder_sequences(
                    bench_cfg.data_path, spec['blur_subdir'], spec['sharp_subdir'])
                out.append((spec['key'], sequences, dict(model_cfg)))

            elif spec['layout'] == 'scene_folder_no_gt':
                sequences = self._collect_scene_folder_no_gt_sequences(
                    bench_cfg.data_path, spec['image_subdir'])
                out.append((spec['key'], sequences, dict(model_cfg)))
        return out

    def _eval_deblur_bench(self) -> dict[str, dict]:
        """
        Evaluate deblurring quality on the configured deblur benchmarks (whichever
        `*_bench` entries are selected in MVRM_EVAL, see BENCH_LAYOUTS).

        Results are stored per sequence (one dir per sequence name).
        Per-image PSNR/SSIM/LPIPS are collected across all sequences and averaged
        at the dataset level.
        """
        import cv2
        import imageio.v2 as imageio

        os.makedirs(self._metric_dir, exist_ok=True)

        expanded = self._expand_deblur_bench_config()
        deblur_datasets = [ds_key for ds_key, _, _ in expanded]
        restored_models_by_ds = {ds_key: restored for ds_key, _, restored in expanded}
        all_metrics: dict[str, dict] = {}
        # Every bench in the new schema has real GT, so pseudo-GT geo eval applies
        # to all of them. Used by both the own-model and restored-model sections
        # below, so it must not be gated behind own-model results being present.
        pseudo_gt_datasets = set(deblur_datasets)

        # Defined unconditionally (not nested inside the own-model branch) so both
        # the own-model and restored-model sections can use it regardless of
        # whether 'ours' is in to_eval_model — a prior version defined _to_pcd only
        # inside the own-model `else:` block, so when to_eval_model excluded 'ours'
        # that branch never ran and the restored-model recon block silently hit a
        # NameError, swallowed by its own `except Exception: pass`.
        from depth_anything_3.utils.geometry import as_homogeneous as _as_hom
        from depth_anything_3.bench.utils import evaluate_3d_reconstruction

        def _to_pcd(depths, exts, ixts, subsample=10, with_index=False):
            """
            Unproject a (N,H,W) depth stack into world-space points via its
            per-frame extrinsics/intrinsics. If with_index, also returns a
            (frame_idx, pixel_idx) pair per point so points can be matched by
            exact (frame, pixel) identity across two different depth/pose
            predictions built from the same image grid.
            """
            pts_all, frame_idx_all, pix_idx_all = [], [], []
            n_v = min(depths.shape[0], exts.shape[0], ixts.shape[0])
            H, W = depths.shape[1], depths.shape[2]
            ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            flat_idx_full = (ys * W + xs).reshape(-1)
            for i in range(n_v):
                d = depths[i]
                valid = (d > 0) & np.isfinite(d)
                fx, fy = ixts[i, 0, 0], ixts[i, 1, 1]
                cx, cy = ixts[i, 0, 2], ixts[i, 1, 2]
                x3 = (xs[valid] - cx) * d[valid] / fx
                y3 = (ys[valid] - cy) * d[valid] / fy
                z3 = d[valid]
                pts_cam = np.stack([x3, y3, z3, np.ones_like(z3)], axis=-1)
                c2w = np.linalg.inv(_as_hom(exts[i:i+1])[0].astype(np.float64))
                pts_world = (c2w @ pts_cam.T).T[:, :3]
                pts_all.append(pts_world[::subsample])
                if with_index:
                    flat_idx = flat_idx_full[valid.reshape(-1)][::subsample]
                    frame_idx_all.append(np.full(len(flat_idx), i))
                    pix_idx_all.append(flat_idx)
            pts = np.concatenate(pts_all, axis=0) if pts_all else np.zeros((0, 3))
            if not with_index:
                return pts
            frame_idx = np.concatenate(frame_idx_all) if frame_idx_all else np.zeros((0,), dtype=int)
            pix_idx = np.concatenate(pix_idx_all) if pix_idx_all else np.zeros((0,), dtype=int)
            return pts, frame_idx, pix_idx

        def _umeyama(src: np.ndarray, dst: np.ndarray):
            """Closed-form Umeyama: find (scale, R, t) minimizing ||scale*R@src_i+t - dst_i||^2."""
            n, m = src.shape
            mu_src, mu_dst = src.mean(0), dst.mean(0)
            src_c, dst_c = src - mu_src, dst - mu_dst
            cov = (dst_c.T @ src_c) / n
            U, D, Vt = np.linalg.svd(cov)
            S = np.eye(m)
            if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                S[-1, -1] = -1
            R = U @ S @ Vt
            var_src = (src_c ** 2).sum() / n
            scale = np.trace(np.diag(D) @ S) / var_src
            t = mu_dst - scale * R @ mu_src
            return scale, R, t

        def _recon_metrics_vs_pseudo_gt(pred_depth, pred_ext, pred_ixt, pgt_depth, pgt_ext, pgt_ixt):
            """
            Sim(3)-align the predicted point cloud to the pseudo-GT point cloud
            (rotation + translation + SCALE) before computing Chamfer/F-score,
            matching the *intent* of the convention used by the other DA3
            benchmarks (see bench/datasets/{eth3d,scannetpp,hiroom,sevenscenes}.py
            _prep_unposed, which Umeyama-aligns predicted camera trajectories to
            GT before fusion) — predicted depth/pose come from a separate forward
            pass with no guaranteed absolute scale relative to the pseudo-GT pass,
            so comparing raw point clouds without this step conflates scale
            mismatch with actual geometric error.

            Unlike those benchmarks, the alignment here is fit from DENSE per-pixel
            point correspondences rather than from the sparse camera-pose
            trajectory (via evo's PosePath3D.align, what
            depth_anything_3.utils.pose_align.align_poses_umeyama wraps). That
            was tried first and rejected: deblur-bench sequences only have
            max_frames views (10 in the configs this was validated against), and
            fitting a 7-DOF Sim(3) transform from only 10 near-static camera
            centers (minimal handshake motion, little 3D spread) is a
            numerically ill-conditioned SVD problem — verified empirically it
            produced wildly wrong rotations and made Chamfer *worse* than no
            alignment at all across every model tested (ours and baselines) on a
            test sequence. Frame i, pixel (u,v) in the predicted point cloud and frame
            i, pixel (u,v) in the pseudo-GT point cloud are (up to depth/pose
            error) the same physical scene point, giving thousands of exact
            correspondences per sequence instead of 10 — a well-conditioned fit
            that was validated to produce sane, non-degenerate results across
            multiple sequences and models (no blow-ups, sensible relative
            ordering of predicted vs pseudo-GT quality).
            """
            n_v = min(pred_ext.shape[0], pgt_ext.shape[0])
            if n_v < 3:
                return None
            pred_depth, pred_ext, pred_ixt = pred_depth[:n_v], pred_ext[:n_v], pred_ixt[:n_v]
            pgt_depth, pgt_ext, pgt_ixt = pgt_depth[:n_v], pgt_ext[:n_v], pgt_ixt[:n_v]

            # Dense (every-other-pixel) point clouds, with (frame, pixel) identity,
            # used only to find correspondences for the Sim(3) fit.
            pred_pts_d, pred_f, pred_p = _to_pcd(pred_depth, pred_ext, pred_ixt, subsample=2, with_index=True)
            pgt_pts_d, pgt_f, pgt_p = _to_pcd(pgt_depth, pgt_ext, pgt_ixt, subsample=2, with_index=True)
            key_pred = pred_f.astype(np.int64) * 1_000_000 + pred_p.astype(np.int64)
            key_pgt = pgt_f.astype(np.int64) * 1_000_000 + pgt_p.astype(np.int64)
            _, idx_pred, idx_pgt = np.intersect1d(key_pred, key_pgt, return_indices=True)
            if len(idx_pred) < 100:
                return None
            scale, R, t = _umeyama(pred_pts_d[idx_pred], pgt_pts_d[idx_pgt])

            # Coarser point clouds (matching the original subsample=10 convention)
            # for the actual Chamfer/precision/recall/F-score computation.
            pred_pts = _to_pcd(pred_depth, pred_ext, pred_ixt, subsample=10)
            pgt_pts = _to_pcd(pgt_depth, pgt_ext, pgt_ixt, subsample=10)
            if pred_pts.shape[0] == 0 or pgt_pts.shape[0] == 0:
                return None
            pred_pts_aligned = scale * (R @ pred_pts.T).T + t
            return evaluate_3d_reconstruction(pred_pts_aligned, pgt_pts, threshold=0.05, down_sample=None)

        for deblur_ds in deblur_datasets:
            ds_model_dir = os.path.join(self.work_dir, 'model_results', deblur_ds)
            if not os.path.exists(ds_model_dir):
                print(f'[WARN] No inference results dir for {deblur_ds}, skipping.')
                continue

            # Collect sequence subdirs (must have an 'unposed' child to distinguish
            # them from model-name dirs like a baseline's name that sit at the same level)
            seq_dirs = sorted([
                d for d in os.listdir(ds_model_dir)
                if os.path.isdir(os.path.join(ds_model_dir, d, 'unposed'))
            ])

            per_img_psnr, per_img_ssim, per_img_mse = [], [], []
            total_images = 0

            for seq_name in seq_dirs:
                export_dir = os.path.join(ds_model_dir, seq_name, 'unposed')
                result_path = os.path.join(export_dir, 'exports', 'npz', 'results.npz')
                gt_meta_path = os.path.join(export_dir, 'exports', 'deblur_gt_files.npz')

                if not os.path.exists(gt_meta_path):
                    continue
                # export_to_npz is async — wait for file to be fully written
                if not os.path.exists(result_path):
                    continue
                from depth_anything_3.bench.dataset import _wait_for_file_ready
                _wait_for_file_ready(result_path, timeout=60.0)

                try:
                    pred_npz = np.load(result_path, allow_pickle=True)
                    if 'image' not in pred_npz:
                        continue

                    pred_imgs = pred_npz['image']  # (N,H,W,3) uint8
                    gt_files = list(np.load(gt_meta_path, allow_pickle=True)['gt_files'])
                    num_views = min(len(gt_files), pred_imgs.shape[0])
                    pred_imgs = pred_imgs[:num_views]

                    gt_imgs = self._load_rgb_images(
                        gt_files[:num_views],
                        target_hw=pred_imgs.shape[1:3],
                        reader=imageio.imread,
                        resize_fn=lambda img, hw: cv2.resize(
                            img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA),
                    )

                    pred_f = pred_imgs.astype(np.float32) / 255.0
                    gt_f   = gt_imgs.astype(np.float32) / 255.0
                    diff   = pred_f - gt_f
                    mse_per  = np.mean(diff * diff, axis=(1, 2, 3))
                    psnr_per = -10.0 * np.log10(np.maximum(mse_per, 1e-10))
                    per_img_psnr.extend(psnr_per.tolist())
                    per_img_mse.extend(mse_per.tolist())

                    try:
                        from skimage.metrics import structural_similarity
                        for p, g in zip(pred_f, gt_f):
                            per_img_ssim.append(
                                structural_similarity(g, p, win_size=11,
                                                      gaussian_weights=True,
                                                      channel_axis=2, data_range=1.0)
                            )
                    except Exception:
                        pass

                    total_images += num_views

                except Exception as e:
                    if self.debug:
                        print(f'[WARN] deblur eval failed for {deblur_ds}/{seq_name}: {e}')

            if total_images == 0:
                print(f'[WARN] No valid results for {deblur_ds}')
            else:

                # LPIPS — compute in one batched pass over all accumulated images
                lpips_val = float('nan')
                try:
                    all_pred, all_gt = [], []
                    for seq_name in seq_dirs:
                        export_dir = os.path.join(ds_model_dir, seq_name, 'unposed')
                        result_path = os.path.join(export_dir, 'exports', 'npz', 'results.npz')
                        gt_meta_path = os.path.join(export_dir, 'exports', 'deblur_gt_files.npz')
                        if not os.path.exists(gt_meta_path) or not os.path.exists(result_path):
                            continue
                        _wait_for_file_ready(result_path, timeout=60.0)
                        pred_npz = np.load(result_path, allow_pickle=True)
                        if 'image' not in pred_npz:
                            continue
                        pred_imgs = pred_npz['image']
                        gt_files = list(np.load(gt_meta_path, allow_pickle=True)['gt_files'])
                        num_views = min(len(gt_files), pred_imgs.shape[0])
                        pred_imgs = pred_imgs[:num_views].astype(np.float32) / 255.0
                        gt_imgs = self._load_rgb_images(
                            gt_files[:num_views],
                            target_hw=pred_imgs.shape[1:3],
                            reader=imageio.imread,
                            resize_fn=lambda img, hw: cv2.resize(
                                img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA),
                        ).astype(np.float32) / 255.0
                        all_pred.append(pred_imgs)
                        all_gt.append(gt_imgs)
                    if all_pred:
                        lpips_val = self._compute_lpips(
                            np.concatenate(all_pred, axis=0),
                            np.concatenate(all_gt,   axis=0),
                        )
                except Exception:
                    pass

                metrics = {
                    'psnr':       float(np.mean(per_img_psnr)) if per_img_psnr else float('nan'),
                    'ssim':       float(np.mean(per_img_ssim)) if per_img_ssim else float('nan'),
                    'lpips':      lpips_val,
                    'mse':        float(np.mean(per_img_mse))  if per_img_mse  else float('nan'),
                    'num_images': float(total_images),
                    'num_seqs':   float(len(seq_dirs)),
                }

                print(f'  deblur_bench | {deblur_ds} ({total_images} imgs, {len(seq_dirs)} seqs): '
                      f'psnr={metrics["psnr"]:.4f}  ssim={metrics["ssim"]:.4f}  lpips={metrics["lpips"]:.4f}')

                # Geometric evaluation with pseudo-GT (only for datasets that have clean GT images)
                if deblur_ds in pseudo_gt_datasets:
                    geo_pose_list, geo_depth_list, geo_recon_list = [], [], []

                    for seq_name in seq_dirs:
                        lq_result_path = os.path.join(
                            ds_model_dir, seq_name, 'unposed', 'exports', 'npz', 'results.npz')
                        pseudo_gt_result_path = os.path.join(
                            ds_model_dir, seq_name, 'pseudo_gt', 'exports', 'npz', 'results.npz')

                        if not os.path.exists(lq_result_path) or not os.path.exists(pseudo_gt_result_path):
                            continue
                        from depth_anything_3.bench.dataset import _wait_for_file_ready
                        _wait_for_file_ready(lq_result_path, timeout=60.0)
                        _wait_for_file_ready(pseudo_gt_result_path, timeout=60.0)
                        try:
                            lq_npz  = np.load(lq_result_path,       allow_pickle=True)
                            pgt_npz = np.load(pseudo_gt_result_path, allow_pickle=True)

                            # --- Pose ---
                            if 'extrinsics' in lq_npz and 'extrinsics' in pgt_npz:
                                from depth_anything_3.bench.utils import compute_pose
                                from depth_anything_3.utils.geometry import as_homogeneous
                                pose_result = compute_pose(
                                    torch.from_numpy(as_homogeneous(lq_npz['extrinsics'])),
                                    torch.from_numpy(as_homogeneous(pgt_npz['extrinsics'])),
                                )
                                geo_pose_list.append(self._to_float_dict(pose_result))

                            # --- Depth ---
                            if 'depth' in lq_npz and 'depth' in pgt_npz:
                                from depth_anything_3.bench.depth_metrics import (
                                    compute_depth_metrics, resize_depth_nearest, DepthEvalConfig)
                                cfg_d = DepthEvalConfig(delta_thresholds=(1.25, 1.25**2, 1.25**3))
                                pred_depths = lq_npz['depth'].astype(np.float32)   # (N,H,W)
                                gt_depths   = pgt_npz['depth'].astype(np.float32)  # (N,H,W)
                                n_frames = min(pred_depths.shape[0], gt_depths.shape[0])
                                for i in range(n_frames):
                                    pd = resize_depth_nearest(pred_depths[i], gt_depths[i].shape[:2])
                                    gd = gt_depths[i]
                                    valid = (gd > 0) & np.isfinite(gd)
                                    m = compute_depth_metrics(pd, gd, valid, cfg=cfg_d)
                                    if m:
                                        geo_depth_list.append(m)

                            # --- 3D reconstruction (Chamfer / Precision / Recall / F-score) ---
                            # Umeyama Sim(3)-aligned (scale included) against pseudo-GT before
                            # building point clouds — see _recon_metrics_vs_pseudo_gt docstring.
                            if ('depth' in lq_npz and 'depth' in pgt_npz and
                                    'extrinsics' in lq_npz and 'extrinsics' in pgt_npz and
                                    'intrinsics' in lq_npz and 'intrinsics' in pgt_npz):
                                try:
                                    recon = _recon_metrics_vs_pseudo_gt(
                                        lq_npz['depth'].astype(np.float32), lq_npz['extrinsics'], lq_npz['intrinsics'],
                                        pgt_npz['depth'].astype(np.float32), pgt_npz['extrinsics'], pgt_npz['intrinsics'],
                                    )
                                    if recon is not None:
                                        # 'chamfer' kept as an alias of 'overall' for backward compatibility
                                        # with earlier consumers of this JSON field.
                                        geo_recon_list.append({**recon, 'chamfer': recon['overall']})
                                except Exception:
                                    pass

                        except Exception as e:
                            if self.debug:
                                print(f'[WARN] geo eval failed for {deblur_ds}/{seq_name}: {e}')

                    if geo_pose_list:
                        mean_pose = self._mean_of_dicts(geo_pose_list)
                        metrics.update({f'geo_pose_{k}': v for k, v in mean_pose.items()})
                        print(f'  deblur_bench | {deblur_ds} geo pose: {mean_pose}')
                    if geo_depth_list:
                        mean_depth = self._mean_of_dicts(geo_depth_list)
                        metrics.update({f'geo_depth_{k}': v for k, v in mean_depth.items()})
                        print(f'  deblur_bench | {deblur_ds} geo depth: abs_rel={mean_depth.get("abs_rel", float("nan")):.4f}')
                    if geo_recon_list:
                        mean_recon = self._mean_of_dicts(geo_recon_list)
                        metrics.update({f'geo_recon_{k}': v for k, v in mean_recon.items()})
                        print(f'  deblur_bench | {deblur_ds} geo recon: chamfer={mean_recon.get("chamfer", float("nan")):.4f}  fscore={mean_recon.get("fscore", float("nan")):.4f}')

                all_metrics[deblur_ds] = metrics
                self._dump_json(
                    os.path.join(self._metric_dir, f'{deblur_ds}_deblur.json'),
                    {deblur_ds: metrics},
                )

            # Evaluate pre-restored images from external deblurring models
            for model_name, _restored_dir in restored_models_by_ds.get(deblur_ds, {}).items():
                model_ds_dir = os.path.join(self.work_dir, 'model_results', deblur_ds, model_name)
                if not os.path.exists(model_ds_dir):
                    print(f'[WARN] No inference results dir for {deblur_ds} [{model_name}], skipping.')
                    continue
                model_seq_dirs = sorted([
                    d for d in os.listdir(model_ds_dir)
                    if os.path.isdir(os.path.join(model_ds_dir, d, 'unposed'))
                ])
                if not model_seq_dirs:
                    continue

                from depth_anything_3.bench.dataset import _wait_for_file_ready
                m_psnr, m_ssim, m_mse = [], [], []
                m_pred_all, m_gt_all = [], []
                m_total = 0
                m_geo_pose, m_geo_depth, m_geo_recon = [], [], []

                for seq_name in model_seq_dirs:
                    export_dir    = os.path.join(model_ds_dir, seq_name, 'unposed')
                    result_path   = os.path.join(export_dir, 'exports', 'npz', 'results.npz')
                    gt_meta_path  = os.path.join(export_dir, 'exports', 'deblur_gt_files.npz')

                    if not os.path.exists(gt_meta_path) or not os.path.exists(result_path):
                        continue
                    _wait_for_file_ready(result_path, timeout=60.0)

                    # --- Image quality vs GT clean ---
                    try:
                        pred_npz = np.load(result_path, allow_pickle=True)
                        if 'image' not in pred_npz:
                            continue
                        pred_imgs = pred_npz['image']
                        gt_files  = list(np.load(gt_meta_path, allow_pickle=True)['gt_files'])
                        num_views = min(len(gt_files), pred_imgs.shape[0])
                        pred_imgs = pred_imgs[:num_views]
                        gt_imgs   = self._load_rgb_images(
                            gt_files[:num_views],
                            target_hw=pred_imgs.shape[1:3],
                            reader=imageio.imread,
                            resize_fn=lambda img, hw: cv2.resize(
                                img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA),
                        )
                        pred_f = pred_imgs.astype(np.float32) / 255.0
                        gt_f   = gt_imgs.astype(np.float32) / 255.0
                        diff   = pred_f - gt_f
                        mse_per  = np.mean(diff * diff, axis=(1, 2, 3))
                        psnr_per = -10.0 * np.log10(np.maximum(mse_per, 1e-10))
                        m_psnr.extend(psnr_per.tolist())
                        m_mse.extend(mse_per.tolist())
                        try:
                            from skimage.metrics import structural_similarity
                            for p, g in zip(pred_f, gt_f):
                                m_ssim.append(
                                    structural_similarity(g, p, win_size=11,
                                                          gaussian_weights=True,
                                                          channel_axis=2, data_range=1.0)
                                )
                        except Exception:
                            pass
                        m_pred_all.append(pred_f)
                        m_gt_all.append(gt_f)
                        m_total += num_views
                    except Exception as e:
                        if self.debug:
                            print(f'[WARN] restored image eval failed for {model_name}/{deblur_ds}/{seq_name}: {e}')

                    # --- Geometric eval vs pseudo-GT ---
                    if deblur_ds in pseudo_gt_datasets:
                        pseudo_gt_result_path = os.path.join(
                            self.work_dir, 'model_results', deblur_ds,
                            seq_name, 'pseudo_gt', 'exports', 'npz', 'results.npz')
                        if not os.path.exists(result_path) or not os.path.exists(pseudo_gt_result_path):
                            continue
                        _wait_for_file_ready(pseudo_gt_result_path, timeout=60.0)
                        try:
                            res_npz = np.load(result_path,            allow_pickle=True)
                            pgt_npz = np.load(pseudo_gt_result_path,  allow_pickle=True)

                            if 'extrinsics' in res_npz and 'extrinsics' in pgt_npz:
                                from depth_anything_3.bench.utils import compute_pose
                                from depth_anything_3.utils.geometry import as_homogeneous
                                pose_result = compute_pose(
                                    torch.from_numpy(as_homogeneous(res_npz['extrinsics'])),
                                    torch.from_numpy(as_homogeneous(pgt_npz['extrinsics'])),
                                )
                                m_geo_pose.append(self._to_float_dict(pose_result))

                            if 'depth' in res_npz and 'depth' in pgt_npz:
                                from depth_anything_3.bench.depth_metrics import (
                                    compute_depth_metrics, resize_depth_nearest, DepthEvalConfig)
                                cfg_d = DepthEvalConfig(delta_thresholds=(1.25, 1.25**2, 1.25**3))
                                pred_depths = res_npz['depth'].astype(np.float32)
                                gt_depths   = pgt_npz['depth'].astype(np.float32)
                                n_frames = min(pred_depths.shape[0], gt_depths.shape[0])
                                for i in range(n_frames):
                                    pd = resize_depth_nearest(pred_depths[i], gt_depths[i].shape[:2])
                                    gd = gt_depths[i]
                                    valid = (gd > 0) & np.isfinite(gd)
                                    m = compute_depth_metrics(pd, gd, valid, cfg=cfg_d)
                                    if m:
                                        m_geo_depth.append(m)

                            if ('depth' in res_npz and 'depth' in pgt_npz and
                                    'extrinsics' in res_npz and 'extrinsics' in pgt_npz and
                                    'intrinsics' in res_npz and 'intrinsics' in pgt_npz):
                                try:
                                    recon = _recon_metrics_vs_pseudo_gt(
                                        res_npz['depth'].astype(np.float32), res_npz['extrinsics'], res_npz['intrinsics'],
                                        pgt_npz['depth'].astype(np.float32), pgt_npz['extrinsics'], pgt_npz['intrinsics'],
                                    )
                                    if recon is not None:
                                        m_geo_recon.append({**recon, 'chamfer': recon['overall']})
                                except Exception:
                                    pass
                        except Exception as e:
                            if self.debug:
                                print(f'[WARN] restored geo eval failed for {model_name}/{deblur_ds}/{seq_name}: {e}')

                if m_total == 0:
                    print(f'[WARN] No valid results for {deblur_ds} [{model_name}]')
                    continue

                m_lpips_val = float('nan')
                try:
                    if m_pred_all:
                        m_lpips_val = self._compute_lpips(
                            np.concatenate(m_pred_all, axis=0),
                            np.concatenate(m_gt_all,   axis=0),
                        )
                except Exception:
                    pass

                model_metrics = {
                    'psnr':       float(np.mean(m_psnr)) if m_psnr else float('nan'),
                    'ssim':       float(np.mean(m_ssim)) if m_ssim else float('nan'),
                    'lpips':      m_lpips_val,
                    'mse':        float(np.mean(m_mse))  if m_mse  else float('nan'),
                    'num_images': float(m_total),
                    'num_seqs':   float(len(model_seq_dirs)),
                }
                if m_geo_pose:
                    mean_pose = self._mean_of_dicts(m_geo_pose)
                    model_metrics.update({f'geo_pose_{k}': v for k, v in mean_pose.items()})
                    print(f'  deblur_bench | {deblur_ds} [{model_name}] geo pose: {mean_pose}')
                if m_geo_depth:
                    mean_depth = self._mean_of_dicts(m_geo_depth)
                    model_metrics.update({f'geo_depth_{k}': v for k, v in mean_depth.items()})
                    print(f'  deblur_bench | {deblur_ds} [{model_name}] geo depth: abs_rel={mean_depth.get("abs_rel", float("nan")):.4f}')
                if m_geo_recon:
                    mean_recon = self._mean_of_dicts(m_geo_recon)
                    model_metrics.update({f'geo_recon_{k}': v for k, v in mean_recon.items()})
                    print(f'  deblur_bench | {deblur_ds} [{model_name}] geo recon: chamfer={mean_recon.get("chamfer", float("nan")):.4f}  fscore={mean_recon.get("fscore", float("nan")):.4f}')

                print(f'  deblur_bench | {deblur_ds} [{model_name}] ({m_total} imgs, {len(model_seq_dirs)} seqs): '
                      f'psnr={model_metrics["psnr"]:.4f}  ssim={model_metrics["ssim"]:.4f}  lpips={model_metrics["lpips"]:.4f}')

                key = f'{deblur_ds}_{model_name}'
                all_metrics[key] = model_metrics
                self._dump_json(
                    os.path.join(self._metric_dir, f'{deblur_ds}_deblur_{model_name}.json'),
                    {key: model_metrics},
                )

        return all_metrics

    def _eval_view_syn(self) -> dict[str, dict]:
        """
        Evaluate view synthesis quality using image-space metrics when rendered outputs are available.

        Requires per-view rendered RGBs saved in exports/npz/results.npz under key "image".
        If not present, the scene is skipped.
        """
        import cv2
        import imageio.v2 as imageio

        os.makedirs(self._metric_dir, exist_ok=True)

        all_metrics: dict[str, dict] = {}
        for data in self.datas:
            dataset = self.datasets[data]
            scenes = self._get_scenes(dataset)
            dataset_results: dict[str, dict] = {}
            success = 0

            for scene in scenes:
                try:
                    export_dir = self._export_dir(data, scene, posed=True)
                    result_path = os.path.join(export_dir, "exports", "npz", "results.npz")
                    if not os.path.exists(result_path):
                        raise FileNotFoundError(result_path)

                    pred_npz = np.load(result_path, allow_pickle=True)
                    if "image" not in pred_npz:
                        raise KeyError("results.npz missing 'image'")

                    pred_imgs = pred_npz["image"]  # (N,H,W,3) uint8
                    if pred_imgs.ndim != 4 or pred_imgs.shape[-1] != 3:
                        raise ValueError(f"Invalid 'image' shape: {pred_imgs.shape}")

                    gt_files = self._get_gt_image_files_for_eval(dataset, scene, export_dir)
                    if not gt_files:
                        raise RuntimeError("No GT image files found for evaluation.")

                    num_views = min(len(gt_files), pred_imgs.shape[0])
                    if num_views == 0:
                        raise RuntimeError("No overlapping views for evaluation.")

                    pred_imgs = pred_imgs[:num_views]
                    gt_imgs = self._load_rgb_images(
                        gt_files[:num_views],
                        target_hw=pred_imgs.shape[1:3],
                        reader=imageio.imread,
                        resize_fn=lambda img, hw: cv2.resize(
                            img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA
                        ),
                    )

                    metrics = self._compute_view_syn_metrics(pred_imgs, gt_imgs)
                    metrics["num_views"] = float(num_views)
                    dataset_results[scene] = metrics
                    success += 1
                except Exception as e:
                    if self.debug:
                        print(f"[WARN] view_syn eval failed for {data}/{scene}: {e}")

            if success == 0:
                continue

            scene_vals = [v for k, v in dataset_results.items() if k != "mean"]
            dataset_results["mean"] = self._mean_of_dicts(scene_vals)
            dataset_results["success_rate"] = float(success) / float(len(scenes)) * 100.0
            dataset_results["_meta"] = {
                "mode": "view_syn",
                "metrics": ["psnr", "ssim", "lpips", "mse"],
                "note": "Uses rendered RGBs from exports/npz/results.npz (key: image) and GT image files.",
            }

            out_path = os.path.join(self._metric_dir, f"{data}_view_syn.json")
            self._dump_json(out_path, dataset_results)
            all_metrics[f"{data}_view_syn"] = dataset_results

        return all_metrics

    def _get_gt_image_files_for_eval(self, dataset, scene: str, export_dir: str) -> list[str]:
        """Return GT image file paths, respecting sampled frames if gt_meta exists."""
        gt_meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
        if os.path.exists(gt_meta_path):
            gt_meta = np.load(gt_meta_path, allow_pickle=True)
            if "image_files" in gt_meta:
                return [str(p) for p in gt_meta["image_files"]]

        full_gt_data = dataset.get_data(scene)
        if hasattr(full_gt_data, "image_files"):
            return list(full_gt_data.image_files)
        return []

    @staticmethod
    def _load_rgb_images(
        paths: list[str],
        target_hw: tuple[int, int],
        reader,
        resize_fn,
    ) -> np.ndarray:
        """Load RGB images, normalize to 3 channels, and resize to target_hw."""
        imgs = []
        for path in paths:
            img = reader(path)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[-1] == 4:
                img = img[..., :3]
            if img.shape[0] != target_hw[0] or img.shape[1] != target_hw[1]:
                img = resize_fn(img, target_hw)
            imgs.append(img)
        return np.stack(imgs, axis=0)

    def _compute_view_syn_metrics(self, pred_imgs: np.ndarray, gt_imgs: np.ndarray) -> dict[str, float]:
        """Compute fast image-space metrics between predicted and GT RGBs."""
        pred = pred_imgs.astype(np.float32) / 255.0
        gt = gt_imgs.astype(np.float32) / 255.0

        diff = pred - gt
        mse = np.mean(diff * diff, axis=(1, 2, 3))
        psnr = -10.0 * np.log10(np.maximum(mse, 1e-10))

        metrics = {
            "psnr": float(np.mean(psnr)) if psnr.size else float("nan"),
            "mse": float(np.mean(mse)) if mse.size else float("nan"),
            "ssim": self._compute_ssim(pred, gt),
            "lpips": self._compute_lpips(pred, gt),
        }
        return metrics

    @staticmethod
    def _compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
        try:
            from skimage.metrics import structural_similarity
        except Exception:
            return float("nan")

        if pred.shape[0] == 0:
            return float("nan")

        ssim_vals = []
        for p, g in zip(pred, gt):
            val = structural_similarity(
                g,
                p,
                win_size=11,
                gaussian_weights=True,
                channel_axis=2,
                data_range=1.0,
            )
            ssim_vals.append(val)
        return float(np.mean(ssim_vals)) if ssim_vals else float("nan")

    @classmethod
    def _get_lpips_model(cls, device: torch.device):
        if not hasattr(cls, "_lpips_cache"):
            cls._lpips_cache = {}
        key = str(device)
        model = cls._lpips_cache.get(key)
        if model is None:
            from lpips import LPIPS

            model = LPIPS(net="vgg").to(device).eval()
            cls._lpips_cache[key] = model
        return model

    def _compute_lpips(self, pred: np.ndarray, gt: np.ndarray) -> float:
        try:
            import torch
        except Exception:
            return float("nan")

        if not torch.cuda.is_available():
            return float("nan")

        if pred.shape[0] == 0:
            return float("nan")

        device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        try:
            model = self._get_lpips_model(device)
        except Exception:
            return float("nan")

        pred_t = torch.from_numpy(pred).permute(0, 3, 1, 2).to(device)
        gt_t = torch.from_numpy(gt).permute(0, 3, 1, 2).to(device)

        batch_size = 8
        vals = []
        with torch.no_grad():
            for i in range(0, pred_t.shape[0], batch_size):
                p = pred_t[i : i + batch_size]
                g = gt_t[i : i + batch_size]
                val = model(g, p, normalize=True)
                vals.append(val.view(-1))

        if not vals:
            return float("nan")
        return float(torch.cat(vals, dim=0).mean().item())

    def _read_pfm(self, path):
        with open(path, "rb") as f:
            header = f.readline().decode().rstrip()
            color = header == "PF"

            width, height = map(int, f.readline().decode().split())
            scale = float(f.readline().decode().rstrip())

            endian = "<" if scale < 0 else ">"
            data = np.fromfile(f, endian + "f")

            shape = (height, width, 3) if color else (height, width)
            data = np.reshape(data, shape)
            data = np.flipud(data)

            return data
        
    def _load_gt_depth_and_mask(self, data: str, dataset, scene: str, full_gt_data, img_idx: int):
        """Load GT depth (meters) and a boolean valid mask (True = valid)."""
        import cv2
        import imageio

        data = data.lower()

        if data == "eth3d":
            img_path = full_gt_data.image_files[img_idx]
            img_name = os.path.basename(img_path)

            scene_dir = os.path.join(dataset.data_root, scene)
            gt_depth_path = os.path.join(scene_dir, "ground_truth_depth", "dslr_images", img_name)
            mask_name = os.path.splitext(img_name)[0] + ".png"
            mask_candidates = [
                os.path.join(scene_dir, "masks_for_images", "dslr_images", mask_name),
                os.path.join(scene_dir, "ground_truth_masks", "dslr_images", img_name),
            ]
            gt_mask_path = next((p for p in mask_candidates if os.path.exists(p)), None)

            if hasattr(full_gt_data, "aux") and hasattr(full_gt_data.aux, "heights"):
                orig_h = int(full_gt_data.aux.heights[img_idx])
                orig_w = int(full_gt_data.aux.widths[img_idx])
            else:
                im = cv2.imread(img_path)
                if im is None:
                    raise FileNotFoundError(f"Failed to read image for size: {img_path}")
                orig_h, orig_w = im.shape[:2]

            gt_depth = np.fromfile(gt_depth_path, dtype=np.float32).reshape(orig_h, orig_w)
            invalid_from_depth = (gt_depth == 0) | (~np.isfinite(gt_depth))

            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_UNCHANGED) if gt_mask_path else None

            if gt_mask is None:
                invalid_from_mask = np.zeros_like(invalid_from_depth)
            else:
                invalid_from_mask = gt_mask == 1

            valid = (~invalid_from_depth) & (~invalid_from_mask)
            return gt_depth.astype(np.float32), valid.astype(bool)

        if data == "dtu":
            img_file = full_gt_data.image_files[img_idx]
            img_num = int(img_file.split('/')[-1].split('_')[1])
            depth_name = f"depth_map_{img_num:04d}.pfm"
            mask_name = f"depth_visual_{img_num:04d}.png"

            gt_depth_path = os.path.join(dataset.data_root, "depth_raw", "Depths", scene, depth_name)
            gt_depth = self._read_pfm(gt_depth_path).astype(np.float32)
            mask_path = os.path.join(dataset.data_root, "depth_raw", "Depths", scene, mask_name)
            # breakpoint()
            mask = Image.open(mask_path)
            mask = np.array(mask, dtype=np.float32)
            valid_mask = mask > 10

            invalid_from_depth = (gt_depth <= 0) | (~np.isfinite(gt_depth))
            valid = valid_mask & (~invalid_from_depth)

            return gt_depth, valid.astype(bool)
        
        aux = getattr(full_gt_data, "aux", None)
        if aux is None or not hasattr(aux, "gt_depth_files"):
            raise RuntimeError(f"Dataset '{data}' does not expose gt_depth_files for depth eval.")

        depth_path = aux.gt_depth_files[img_idx]
        # breakpoint()

        if data == "7scenes":
            raw = imageio.imread(depth_path).astype(np.float32)
            invalid = raw == 65535
            gt_depth = raw / 1000.0
            gt_depth[invalid] = 0.0
            valid = gt_depth > 0
            return gt_depth.astype(np.float32), valid.astype(bool)

        if data == "hiroom":
            raw = imageio.imread(depth_path).astype(np.float32)
            gt_depth = raw / 65535.0 * 100.0
            valid = gt_depth > 0
            if hasattr(aux, "aliasing_mask_files"):
                alias_path = aux.aliasing_mask_files[img_idx]
                if alias_path and os.path.exists(alias_path):
                    alias = imageio.imread(alias_path)
                    valid &= alias == 0
            return gt_depth.astype(np.float32), valid.astype(bool)

        if data == "scannetpp":
            raw = imageio.imread(depth_path).astype(np.float32)
            gt_depth = raw / 1000.0
            valid = (gt_depth > 0) & np.isfinite(gt_depth)
            return gt_depth.astype(np.float32), valid.astype(bool)

        raw = imageio.imread(depth_path).astype(np.float32)
        gt_depth = raw / 1000.0
        valid = (gt_depth > 0) & np.isfinite(gt_depth)
        return gt_depth.astype(np.float32), valid.astype(bool)

    def print_metrics(self, metrics: TDict[str, dict] = None) -> None:
        """
        Print evaluation metrics in a beautiful tabular format.

        Args:
            metrics: Metrics dictionary. If None, loads from saved JSON files.
        """
        if metrics is None:
            metrics = self._load_metrics()

        self._printer.print_results(metrics)

    # -------------------- Evaluation Methods -------------------- #

    def _eval_pose(self) -> Iterable[tuple]:
        """Compute pose-estimation metrics for each dataset and scene."""
        os.makedirs(self._metric_dir, exist_ok=True)

        for data in tqdm(self.datas, desc="Datasets (pose eval)"):
            dataset = self.datasets[data]
            dataset_results = Dict()
            scenes = self._get_scenes(dataset)

            for scene in tqdm(scenes, desc=f"{data} scenes", leave=False):
                export_dir = self._export_dir(data, scene, posed=False)                             # model predictions
                result_path = os.path.join(export_dir, "exports", "mini_npz", "results.npz")        # model predictions
                
                # Check if result file exists and is valid
                if not os.path.exists(result_path):
                    print(f"\n[ERROR] Result file not found: {result_path}")
                    print(f"[ERROR] CWD: {os.getcwd()}")
                    print(f"[ERROR] Please run inference first (remove --eval_only)")
                    continue
                try:
                    # Use saved GT meta (handles frame sampling correctly)
                    gt_meta = self._load_gt_meta(export_dir)                            # get gt_meta used for evaluation (handles frame sampling)
                    if gt_meta is not None:
                        result = self._compute_pose_with_gt(result_path, gt_meta)
                    else:
                        # Fallback to dataset GT (no sampling was done)
                        result = dataset.eval_pose(scene, result_path)
                    dataset_results[scene] = self._to_float_dict(result)
                except Exception as e:
                    print(f"\n[ERROR] Failed to evaluate pose for {data}/{scene}: {e}")
                    print(f"[ERROR] File path: {os.path.abspath(result_path)}")
                    if self.debug:
                        import traceback
                        traceback.print_exc()
                    continue

            if not dataset_results:
                print(f"[WARNING] No valid results for {data}")
                continue
                
            dataset_results["mean"] = self._mean_of_dicts(dataset_results.values())
            out_path = os.path.join(self._metric_dir, f"{data}_pose.json")
            self._dump_json(out_path, dataset_results)
            yield data, dataset_results

    def _eval_reconstruction(self, mode: str) -> Iterable[tuple]:
        """
        Compute reconstruction metrics for each dataset and scene.

        Args:
            mode: "recon_unposed" or "recon_posed"
        """
        assert mode in {"recon_unposed", "recon_posed"}
        os.makedirs(self._metric_dir, exist_ok=True)

        posed_flag = mode == "recon_posed"
        
        # Filter out datasets that don't support reconstruction (e.g., dtu64)
        recon_datas = [d for d in self.datas if d != "dtu64"]

        for data in tqdm(recon_datas, desc=f"Datasets ({mode} eval)"):
            dataset = self.datasets[data]
            dataset_results = Dict()
            scenes = self._get_scenes(dataset)

            # Prepare paths for all scenes
            scene_list = []
            result_paths = []
            fuse_paths = []
            for scene in scenes:
                export_dir = self._export_dir(data, scene, posed=posed_flag)
                result_path = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
                fuse_path = os.path.join(export_dir, "exports", "fuse", "pcd.ply")
                scene_list.append(scene)
                result_paths.append(result_path)
                fuse_paths.append(fuse_path)

            # Parallel fusion (default 4 workers)
            # DTU uses CUDA operations in fusion, which doesn't work well with ThreadPool
            use_sequential = (data == "dtu")
            parallel_execution(
                scene_list,
                result_paths,
                fuse_paths,
                action=lambda s, rp, fp: dataset.fuse3d(s, rp, fp, mode),
                num_processes=self.num_fusion_workers,
                print_progress=True,
                desc=f"{data} fusion",
                sequential=use_sequential,
            )

            # breakpoint()
            # Sequential evaluation (fast, no need to parallelize)
            for scene, fuse_path in zip(scene_list, fuse_paths):
                # DTU supports CPU-based evaluation
                if data == "dtu" and hasattr(dataset, "eval3d"):
                    result = dataset.eval3d(scene, fuse_path)
                else:
                    result = dataset.eval3d(scene, fuse_path)
                dataset_results[scene] = self._to_float_dict(result)
                print(f"  {mode} | {data} | {scene}: {result}")

            dataset_results["mean"] = self._mean_of_dicts(dataset_results.values())
            out_path = os.path.join(self._metric_dir, f"{data}_{mode}.json")
            self._dump_json(out_path, dataset_results)
            yield data, dataset_results

    # -------------------- Helpers -------------------- #

    def _save_gt_meta(self, export_dir: str, scene_data: Dict) -> None:
        """
        Save GT extrinsics/intrinsics/image_files for evaluation.

        This is needed when frames are sampled, so eval_pose and fuse3d can use
        the correct (sampled) GT instead of full dataset GT.

        Args:
            export_dir: Export directory for the scene
            scene_data: Sampled scene data
        """
        meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        np.savez_compressed(
            meta_path,
            extrinsics=scene_data.extrinsics,
            intrinsics=scene_data.intrinsics,
            image_files=np.array(scene_data.image_files, dtype=object),
        )

    def _load_gt_meta(self, export_dir: str) -> Dict:
        """
        Load saved GT extrinsics/intrinsics for evaluation.

        Returns:
            Dict with extrinsics and intrinsics, or None if not found
        """
        meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
        if os.path.exists(meta_path):
            data = np.load(meta_path)
            return Dict({
                "extrinsics": data["extrinsics"],
                "intrinsics": data["intrinsics"],
            })
        return None

    def _compute_pose_with_gt(self, result_path: str, gt_meta: Dict) -> TDict[str, float]:
        """
        Compute pose metrics using saved GT meta (handles frame sampling).

        Args:
            result_path: Path to npz with predicted extrinsics
            gt_meta: Dict with GT extrinsics from saved meta

        Returns:
            Dict with pose metrics
        """
        from depth_anything_3.bench.dataset import _wait_for_file_ready
        from depth_anything_3.bench.utils import compute_pose
        from depth_anything_3.utils.geometry import as_homogeneous

        _wait_for_file_ready(result_path)
        pred = np.load(result_path)
        return compute_pose(
            torch.from_numpy(as_homogeneous(pred["extrinsics"])),
            torch.from_numpy(as_homogeneous(gt_meta["extrinsics"])),
        )

    def _sample_frames(self, scene_data: Dict, scene: str) -> Dict:
        """
        Sample frames if scene has more than max_frames.

        Uses fixed random seed (42) for reproducibility.

        Args:
            scene_data: Scene data dict with image_files, extrinsics, intrinsics, aux
            scene: Scene name (for logging)

        Returns:
            Sampled scene_data if num_frames > max_frames, otherwise original
        """
        
        
        if self.max_frames <= 0:
            return scene_data
        
                
        num_frames = len(scene_data.image_files)

        if num_frames <= self.max_frames:
            return scene_data

        # When load_lq is on, restrict sampling pool to frames that have a valid LQ image.
        # lq_image_files is aligned with image_files (None where LQ is absent).
        load_lq = self.full_cfg.MVRM_EVAL.get('load_lq', False)
        if load_lq and hasattr(scene_data, 'lq_image_files'):
            candidate_indices = [i for i, p in enumerate(scene_data.lq_image_files) if p is not None]
        else:
            candidate_indices = list(range(num_frames))

        random.seed(42)
        random.shuffle(candidate_indices)
        sampled_indices = sorted(candidate_indices[:self.max_frames])

        print(f"  [Sampling] {scene}: {num_frames} ({len(candidate_indices)} valid) -> {self.max_frames} frames")

        # Create new scene_data with sampled frames
        sampled = Dict()

        # PHO: lq/res are aligned with image_files via None placeholders
        if hasattr(scene_data, 'lq_image_files'):
            sampled.lq_image_files = [scene_data.lq_image_files[i] for i in sampled_indices]

        if hasattr(scene_data, 'res_image_files'):
            sampled.res_image_files = [scene_data.res_image_files[i] for i in sampled_indices]

        sampled.image_files = [scene_data.image_files[i] for i in sampled_indices]



        sampled.extrinsics = scene_data.extrinsics[sampled_indices]
        sampled.intrinsics = scene_data.intrinsics[sampled_indices]



        # Copy aux data, sampling lists if needed
        sampled.aux = Dict()
        for key, val in scene_data.aux.items():
            if isinstance(val, list) and len(val) == num_frames:
                sampled.aux[key] = [val[i] for i in sampled_indices]
            elif isinstance(val, np.ndarray) and len(val) == num_frames:
                sampled.aux[key] = val[sampled_indices]
            else:
                sampled.aux[key] = val

        return sampled

    @property
    def _metric_dir(self) -> str:
        """Directory for storing metric JSON files."""
        return os.path.join(self.work_dir, "metric_results")

    def _export_dir(self, data: str, scene: str, posed: bool) -> str:
        """
        Get export directory path.

        Structure: .../model_results/{data}/{scene}/{posed|unposed}
        """
        suffix = "posed" if posed else "unposed"
        export_dir = os.path.join(self.work_dir, "model_results", data, scene, suffix)
        os.makedirs(export_dir, exist_ok=True)
        return export_dir

    @staticmethod
    def _to_float_dict(d: TDict[str, float]) -> dict:
        """Convert numpy scalars to plain Python floats for JSON safety."""
        return {k: float(v) for k, v in d.items()}

    # @staticmethod
    # def _mean_of_dicts(dicts: Iterable[dict]) -> dict:
    #     """Compute elementwise mean across a list of homogeneous metric dicts."""
    #     dicts = list(dicts)
    #     if not dicts:
    #         return {}
    #     keys = dicts[0].keys()
    #     return {k: float(np.mean([d[k] for d in dicts]).item()) for k in keys}


    @staticmethod
    def _mean_of_dicts(dicts: Iterable[dict]) -> dict:
        """Compute elementwise mean across a list of homogeneous metric dicts.
        Ignores NaN values (failed scenes)."""
        dicts = list(dicts)
        if not dicts:
            return {}

        keys = dicts[0].keys()
        out = {}

        for k in keys:
            values = np.array([d[k] for d in dicts], dtype=np.float64)
            if np.all(np.isnan(values)):
                out[k] = float("nan")
            else:
                out[k] = float(np.nanmean(values))

        return out


    @staticmethod
    def _dump_json(path: str, obj: dict, indent: int = 4) -> None:
        """Write JSON with UTF-8 and pretty indentation."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)

    def _load_metrics(self) -> TDict[str, dict]:
        """Load evaluation metrics from JSON files."""
        metrics = {}
        metric_dir = self._metric_dir

        if not os.path.exists(metric_dir):
            return metrics

        for filename in os.listdir(metric_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(metric_dir, filename)
                try:
                    with open(filepath, encoding="utf-8") as f:
                        data = json.load(f)
                    key = filename[:-5]  # Remove .json extension
                    metrics[key] = data
                except Exception as e:
                    print(f"Warning: Failed to read metrics file: {filename} - {e}")

        return metrics


# -------------------- CLI Entry Point -------------------- #


if __name__ == "__main__":
    import sys
    import argparse
    from omegaconf import OmegaConf
    from depth_anything_3.cfg import load_config


    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config

    # # Get default config path (relative to this file)
    # _default_config = os.path.join(
    #     os.path.dirname(__file__), "configs", "eval_bench.yaml"
    # )


    # # Check for help flag first (we need to handle this before OmegaConf)
    # if "--help" in sys.argv or "-h" in sys.argv:
    #     pass  # Will handle after config loading

    # Set up argv for OmegaConf processing
    argv = sys.argv[1:]

    # # Check if user provides custom config
    # config_path = _default_config
    # breakpoint()
    # if "--config" in argv:
    #     config_idx = argv.index("--config")
    #     if config_idx + 1 < len(argv):
    #         config_path = argv[config_idx + 1]
    #         # Remove --config and its value
    #         argv = argv[:config_idx] + argv[config_idx + 2:]
    
    
    # Print help if requested
    if "--help" in sys.argv or "-h" in sys.argv:
        print("""
DepthAnything3 Benchmark Evaluation

Usage:
  python -m depth_anything_3.bench.evaluator [OPTIONS] [KEY=VALUE ...]

Configuration:
  --config PATH                      Config YAML file (default: bench/configs/eval_bench.yaml)

Config Overrides (using dotlist notation):
  model.path=VALUE                   Model path or HuggingFace ID
  workspace.work_dir=VALUE           Working directory for outputs
  eval.datasets=[dataset1,dataset2]  Datasets to evaluate (eth3d,7scenes,scannetpp,hiroom,dtu,dtu64)
  eval.modes=[mode1,mode2]           Evaluation modes (pose,recon_unposed,recon_posed)
  eval.scenes=[scene1,scene2]        Specific scenes to evaluate (null=all)
  eval.max_frames=VALUE              Max frames per scene (-1=no limit, default: 100)
  eval.ref_view_strategy=VALUE       Reference view strategy (default: first)
  eval.eval_only=VALUE               Only run evaluation (skip inference) (true/false)
  eval.print_only=VALUE              Only print saved metrics (true/false)
  inference.num_fusion_workers=VALUE Number of parallel workers (default: 4)
  inference.debug=VALUE              Enable debug mode (true/false)

Special Flags:
  --help, -h                         Show this help message

Multi-GPU:
  Use CUDA_VISIBLE_DEVICES to specify GPUs (auto-detected and distributed)

Examples:
  # Use default config
  python -m depth_anything_3.bench.evaluator

  # Override model path
  python -m depth_anything_3.bench.evaluator model.path=depth-anything/DA3-LARGE

  # Evaluate specific datasets and modes
  python -m depth_anything_3.bench.evaluator \\
      eval.datasets=[eth3d,hiroom] \\
      eval.modes=[pose]

  # Use custom config with overrides
  python -m depth_anything_3.bench.evaluator \\
      --config my_config.yaml \\
      model.path=/path/to/model \\
      eval.max_frames=50

  # Multi-GPU inference (auto-distributed)
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m depth_anything_3.bench.evaluator

  # Debug specific scenes
  python -m depth_anything_3.bench.evaluator \\
      eval.datasets=[eth3d] \\
      eval.scenes=[courtyard] \\
      inference.debug=true

  # Only evaluate (skip inference)
  python -m depth_anything_3.bench.evaluator eval.eval_only=true

  # Only print saved metrics
  python -m depth_anything_3.bench.evaluator eval.print_only=true

          """)
        sys.exit(0)

    # Load config with CLI overrides using OmegaConf dotlist
    # Example: python evaluator.py model.path=/path/to/model eval.datasets=[eth3d,dtu]
    config = load_config(config_path, argv=argv)

    # Extract config values
    work_dir = config.workspace.work_dir
    model_path = config.model.path
    datasets = config.eval.datasets
    modes = config.eval.modes
    ref_view_strategy = config.eval.ref_view_strategy
    scenes = config.eval.scenes
    max_frames = config.eval.max_frames
    eval_only = config.eval.eval_only
    print_only = config.eval.print_only
    debug = config.inference.debug
    num_fusion_workers = config.inference.num_fusion_workers
    

    # GPU settings: parse from CLI dotlist args (gpu_id=X total_gpus=Y)
    # These are passed by the main process when spawning workers
    gpu_id = 0
    total_gpus = 1
    for arg in argv:
        if arg.startswith("gpu_id="):
            gpu_id = int(arg.split("=")[1])
        elif arg.startswith("total_gpus="):
            total_gpus = int(arg.split("=")[1])

    # Override dataset scenes if specified
    if scenes:
        print(f"[INFO] Running on specific scenes: {scenes}")

    evaluator = Evaluator(
        work_dir=work_dir,
        datas=datasets,
        modes=modes,
        ref_view_strategy=ref_view_strategy,
        scenes=scenes,
        debug=debug,
        num_fusion_workers=num_fusion_workers,
        max_frames=max_frames,
        gpu_id=gpu_id,
        total_gpus=total_gpus,
        full_cfg=config,
    )

    if print_only:
        evaluator.print_metrics()
    elif eval_only:
        metrics = evaluator.eval()
        evaluator.print_metrics(metrics)
    else:
        # Parse CUDA_VISIBLE_DEVICES to get GPU list
        # If not set, use all available GPUs
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_devices is not None and cuda_devices.strip():
            gpu_list = [g.strip() for g in cuda_devices.split(",") if g.strip()]
        else:
            # CUDA_VISIBLE_DEVICES not set, use all available GPUs
            num_available = torch.cuda.device_count()
            gpu_list = [str(i) for i in range(num_available)] if num_available > 0 else ["0"]

        # Auto multi-GPU: if multiple GPUs and not a worker process
        is_worker = os.environ.get("_DA3_WORKER") == "1"

        # if len(gpu_list) > 1 and not is_worker: # f
        #     # Launch worker processes
        #     import subprocess

        #     num_gpus = len(gpu_list)
        #     print(f"[INFO] Detected {num_gpus} GPUs: {gpu_list}")
        #     print(f"[INFO] Launching {num_gpus} workers...")

        #     # Build base command
        #     base_cmd = [sys.executable, "-m", "depth_anything_3.bench.evaluator"]
        #     # Pass config via dotlist instead of CLI args
        #     if config_path != _default_config:
        #         base_cmd += ["--config", config_path]
        #     base_cmd += [f"model.path={model_path}"]
        #     base_cmd += [f"workspace.work_dir={work_dir}"]
        #     base_cmd += [f"eval.datasets=[{','.join(datasets)}]"]
        #     base_cmd += [f"eval.modes=[{','.join(modes)}]"]
        #     if scenes:
        #         base_cmd += [f"eval.scenes=[{','.join(scenes)}]"]
        #     base_cmd += [f"eval.max_frames={max_frames}"]
        #     base_cmd += [f"eval.ref_view_strategy={ref_view_strategy}"]
        #     base_cmd += [f"inference.debug={str(debug).lower()}"]
        #     base_cmd += [f"inference.num_fusion_workers={num_fusion_workers}"]

        #     # Launch workers
        #     processes = []
        #     for idx, gpu_id in enumerate(gpu_list):
        #         env = os.environ.copy()
        #         env["CUDA_VISIBLE_DEVICES"] = gpu_id
        #         env["_DA3_WORKER"] = "1"  # Mark as worker process

        #         cmd = base_cmd.copy()
        #         # GPU-specific worker config
        #         cmd += [f"gpu_id={idx}", f"total_gpus={num_gpus}"]

        #         print(f"[INFO] Starting worker {idx} on GPU {gpu_id}")
        #         p = subprocess.Popen(cmd, env=env)
        #         processes.append(p)

        #     # Wait for all workers
        #     for p in processes:
        #         p.wait()

        #     print(f"[INFO] All {num_gpus} workers completed")

        #     # Run evaluation after all inference is done
        #     metrics = evaluator.eval()
        #     evaluator.print_metrics(metrics)
        # else:   # t
        

        if 'DA3' in model_path:
            print('Loading DA3')
            # Single GPU or worker process
            from depth_anything_3.api import DepthAnything3
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            api = DepthAnything3.from_pretrained(model_path)
            api = api.to(device)
        
        
        elif 'VGGT' in model_path:
            print('Loading VGGT')
            # Single GPU or worker process
            from depth_anything_3.vggt_api import VGGT
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            api = VGGT.from_pretrained(model_path)
            api = api.to(device)
        
        
        else:
            print('check da3 or vggt model ckpt path ! ')


        if torch.__version__ >= "2.0":
            api = torch.compile(api)
        
        
        # generator
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(42)  # any fixed seed you like

        evaluator.infer(api, model_path=model_path, noise_generator=noise_generator, device=device)

        # Only run eval if single GPU mode (workers don't eval)
        if not is_worker:
            metrics = evaluator.eval()
            evaluator.print_metrics(metrics)
