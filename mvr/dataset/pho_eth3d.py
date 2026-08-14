import os 
import csv 
import glob
import h5py
import cv2 
import numpy as np 
import torch 
import random 
from PIL import Image
from torch.utils.data import Dataset 
from depth_anything_3.utils.io.input_processor import InputProcessor
from motionblur.motionblur import Kernel 
from collections import defaultdict


def group_by_scene(paths, scene_idx):
    scene_dict = defaultdict(list)
    for p in paths:
        scene = p.split('/')[scene_idx]
        scene_dict[scene].append(p)
    return scene_dict


class PhoETH3D(Dataset):
    def __init__(self, data_cfg, mode='train'):
        
        self.ds_name = 'eth3d'
        self.data_cfg = data_cfg 
        self.mode = mode 
        self.data = {}
        
        self.process_res = data_cfg.get('process_res', 504)
        self.patch_size = data_cfg.get('patch_size', 14)

        
        eval_scene = data_cfg.eval_scene
        lq_paths = glob.glob(f'{data_cfg.lq_root_path}/*/images/dslr_images/*')
        hq_paths = glob.glob(f'{data_cfg.hq_root_path}/*/images/dslr_images/*.JPG')
        depth_paths = glob.glob(f'{data_cfg.hq_root_path}/*/ground_truth_depth/dslr_images/*.JPG')

            
        # group paths by scene
        lq_by_scene = group_by_scene(lq_paths, scene_idx=-4)
        hq_by_scene = group_by_scene(hq_paths, scene_idx=-4)
        depth_by_scene = group_by_scene(depth_paths, scene_idx=-4)

        lq_paths, hq_paths, depth_paths = [], [], []
        for scene in eval_scene:
            assert scene in hq_by_scene, f"Scene {scene} not found in HQ paths"
            assert scene in lq_by_scene, f"Scene {scene} not found in LQ paths"
            assert scene in depth_by_scene, f"Scene {scene} not found in depth paths"
            # sort to ensure alignment
            lq_scene = sorted(lq_by_scene[scene])[:data_cfg.num_eval_img]
            hq_scene = sorted(hq_by_scene[scene])[:data_cfg.num_eval_img]
            depth_scene = sorted(depth_by_scene[scene])[:data_cfg.num_eval_img]
            assert len(lq_scene) == len(hq_scene) == len(depth_scene), f"Mismatch in scene {scene}"
            lq_paths.extend(lq_scene)
            hq_paths.extend(hq_scene)
            depth_paths.extend(depth_scene)
        # safety check
        assert len(lq_paths) != 0 
        assert len(hq_paths) != 0
        assert len(depth_paths) != 0
        assert len(lq_paths) == len(hq_paths) == len(depth_paths)
        # append
        self.data['hq_img'] = hq_paths 
        self.data['lq_img'] = lq_paths 
        self.data['gt_depth'] = depth_paths 
        

        self.view_sel = data_cfg.view_selection
        self.input_processor = InputProcessor()
        


    def convert_imgpath(self, img_path: str) -> np.ndarray:
        """
        Load a TartanAir PNG image and return RGB uint8 numpy array.

        Args:
            img_path: path to *.png image

        Returns:
            img: np.ndarray [H, W, 3], uint8, RGB
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # cv2 loads as BGR uint8
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        return img_rgb
    
    
    def resize(self, img: np.ndarray):
        """
        Resize image using:
        1) longest-side resize to process_res
        2) make divisible by patch_size via small resize
        Args:
            img: np.ndarray [H, W, 3], uint8
        Returns:
            resized_img: np.ndarray [H', W', 3], uint8
        """
        process_res = self.process_res
        patch_size = self.patch_size
        # -------------------------
        # 1. resize longest side
        # -------------------------
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest != process_res:
            scale = process_res / float(longest)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            interpolation = (cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA)
            img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        # -------------------------
        # 2. make divisible by patch_size (resize version)
        # -------------------------
        h, w = img.shape[:2]
        def nearest_multiple(x, p):
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down
        new_w = max(1, nearest_multiple(w, patch_size))
        new_h = max(1, nearest_multiple(h, patch_size))
        if new_w != w or new_h != h:
            upscale = (new_w > w) or (new_h > h)
            interpolation = (cv2.INTER_CUBIC if upscale else cv2.INTER_AREA)
            img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        return img

    def resize_depth(self, depth: np.ndarray):
        process_res = self.process_res
        patch_size = self.patch_size
        h, w = depth.shape
        longest = max(h, w)
        if longest != process_res:
            scale = process_res / float(longest)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            depth = cv2.resize(
                depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )
        h, w = depth.shape
        def nearest_multiple(x, p):
            return int(round(x / p)) * p
        new_w = max(1, nearest_multiple(w, patch_size))
        new_h = max(1, nearest_multiple(h, patch_size))
        if new_w != w or new_h != h:
            depth = cv2.resize(
                depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )
        return depth

            
    def load_depth(self, depth_path, rgb_path):
        """
        ETH3D depth loader (official behavior).
        """
        # Read RGB ONLY to get shape
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Failed to load RGB image: {rgb_path}")
        h, w = rgb.shape[:2]
        # Read raw float32 depth
        depth = np.fromfile(depth_path, dtype=np.float32)
        if depth.size != h * w:
            raise ValueError(
                f"Depth size mismatch: {depth.size} vs expected {h*w}"
            )
        depth = depth.reshape(h, w)
        return depth


    def mask_invalid_depth(self, depth: np.ndarray):
        depth = depth.copy()
        invalid = np.isnan(depth) | np.isinf(depth) | (depth <= 0)
        depth[invalid] = 0.0
        return depth


    def get_nearby_ids_random(
        self,
        anchor,
        num_frames,
        expand_ratio=2.0,
    ):
        expand_range = int(num_frames * expand_ratio)
        low = max(0, anchor - expand_range)
        high = min(len(self.data['hq_img']), anchor + expand_range + 1)
        candidates = np.arange(low, high)
        sampled = np.random.choice(
            candidates,
            size=num_frames - 1,
            replace = (len(candidates) < num_frames - 1),
        )
        return np.concatenate([[anchor], sampled])


    def __getitem__(self, items):
        
        idx, num_input_view = items

        # view selection strategy        
        if self.view_sel.strategy == 'near_random':
            frame_ids = self.get_nearby_ids_random(anchor=idx, num_frames=num_input_view, expand_ratio=self.view_sel.expand_ratio)
        
        outputs={}
        outputs['frame_ids'] = frame_ids
        

        hq_view_id, hq_view_list = [], []
        lq_view_id, lq_view_list = [], []
        depth_view_id, depth_view_list = [], []

        for i in frame_ids:
            # HQ
            if 'hq_img' in self.data:
                hq_view = self.data['hq_img'][i]
                scene_id = hq_view.split('/')[-4]
                view_id = hq_view.split('/')[-1].split('.')[0]
                hq_view_id.append(f'eth3d_{scene_id}_{view_id}')
                hq_view_list.append(self.resize(self.convert_imgpath(hq_view)))

            # LQ
            if 'lq_img' in self.data:
                lq_view = self.data['lq_img'][i]
                scene_id = lq_view.split('/')[-2]
                view_id = lq_view.split('/')[-1].split('.')[0]
                lq_view_id.append(f'eth3d_{scene_id}_{view_id}')
                lq_view_list.append(self.resize(self.convert_imgpath(lq_view)))

            # Depth
            if 'gt_depth' in self.data:
                depth_view = self.data['gt_depth'][i]
                scene_id = depth_view.split('/')[-4]
                view_id = depth_view.split('/')[-1].split('.')[0]
                depth_view_id.append(f'eth3d_{scene_id}_{view_id}')
                depth = self.load_depth(depth_view, self.data['hq_img'][i])
                depth = self.mask_invalid_depth(depth)
                depth_view_list.append(self.resize_depth(depth))


        # store outputs
        if 'hq_img' in self.data:
            outputs['hq_ids'] = hq_view_id
            outputs['hq_views'] = hq_view_list

        if 'lq_img' in self.data:
            outputs['lq_ids'] = lq_view_id
            outputs['lq_views'] = lq_view_list

        if 'gt_depth' in self.data:
            outputs['gt_depth_ids'] = depth_view_id
            outputs['gt_depths'] = depth_view_list

        return outputs


    def __len__(self):
        return len(self.data['hq_img'])
