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
from mvr.dataset.view_sel_near_camera import compute_ranking


class PhoHypersim(Dataset):
    def __init__(self, data_cfg, mode='train'):
        
        self.ds_name = 'hypersim'
        self.data_cfg = data_cfg 
        self.mode = mode 
        self.data = {}
        
        self.num_eval_img = data_cfg.get('num_eval_img', -1)

        self.process_res = data_cfg.get('process_res', 504)
        self.patch_size = data_cfg.get('patch_size', 14)

        
        # train/val/test annotations
        with open(data_cfg.ann_path) as f:
            anns = list(csv.DictReader(f))  # len: 82,900
            
            
        if mode == 'train':
            sampled_anns = [ann for ann in anns if ann['split_partition_name']=='train']      # 59,543        
            sampled_scenes = set([ann['scene_name'] for ann in anns if ann['split_partition_name'] == 'train'])
        
        elif mode == 'val':
            sampled_anns = [ann for ann in anns if ann['split_partition_name']=='val']      # 59,543        
            sampled_scenes = set([ann['scene_name'] for ann in anns if ann['split_partition_name'] == 'val'])
        

        # camera annotations (for view_sel)
        cam_ori_paths = sorted(glob.glob(f'{data_cfg.hq_root_path}/*/_detail/cam_*/camera_keyframe_orientations.hdf5')) # camera rotation
        cam_pos_paths = sorted(glob.glob(f'{data_cfg.hq_root_path}/*/_detail/cam_*/camera_keyframe_positions.hdf5'))    # camera translation
        cam_ori_paths = sorted([path for path in cam_ori_paths if path.split('/')[-4] in sampled_scenes])
        cam_pos_paths = sorted([path for path in cam_pos_paths if path.split('/')[-4] in sampled_scenes])
        assert len(cam_ori_paths) == len(cam_pos_paths)


        def cam_key(p):
            return (p.split('/')[-4], p.split('/')[-2])  # (scene_id, cam_xx)
        ori_map = {cam_key(p): p for p in cam_ori_paths}
        pos_map = {cam_key(p): p for p in cam_pos_paths}
        

        self.camera_rankings = {}
        self.camera_num_frames = {}
        common_keys = ori_map.keys() & pos_map.keys()
        
        self.camera_extrinsics = {}
        
        for scene_id, camera_id in sorted(common_keys):
            # parse identifiers
            cam_ori_path = ori_map[(scene_id, camera_id)]
            cam_pos_path = pos_map[(scene_id, camera_id)]

            cache_key = (scene_id, camera_id)
            if cache_key in self.camera_rankings:
                continue  # already computed

            # load camera data
            with h5py.File(cam_ori_path, 'r') as f_ori:
                ext_r = f_ori['dataset'][:]   # (N,3,3)

            with h5py.File(cam_pos_path, 'r') as f_pos:
                ext_t = f_pos['dataset'][:]   # (N,3)

            assert ext_r.shape[0] == ext_t.shape[0]
            N = ext_r.shape[0]

            # build extrinsics
            extrinsics = np.zeros((N, 4, 4), dtype=np.float32)
            extrinsics[:, :3, :3] = ext_r
            extrinsics[:, :3, 3]  = ext_t
            extrinsics[:, 3, 3]   = 1.0
            
            self.camera_extrinsics[(scene_id, camera_id)] = extrinsics

            # compute ranking
            ranking, _ = compute_ranking(
                extrinsics,
                lambda_t=1.0,
                normalize=True,
                batched=True
            )

            # cache
            self.camera_rankings[cache_key] = ranking
            self.camera_num_frames[cache_key] = N        


        # HQ 
        hq_paths = sorted([f"{data_cfg.hq_root_path}/{ann['scene_name']}/images/scene_{ann['camera_name']}_final_hdf5/frame.{int(ann['frame_id']):04}.color.hdf5" for ann in sampled_anns])


        # depth 
        depth_paths = sorted([f"{data_cfg.hq_root_path}/{ann['scene_name']}/images/scene_{ann['camera_name']}_geometry_hdf5/frame.{int(ann['frame_id']):04}.depth_meters.hdf5" for ann in sampled_anns])

  
        self.data['hq_img'] = sorted([path for path in hq_paths if os.path.exists(path)])
        self.data['gt_depth'] = sorted([path for path in depth_paths if os.path.exists(path)])
      
      
        # for validation
        if self.num_eval_img != -1:                                                                              
            self.data['hq_img']   = self.data['hq_img'][:self.num_eval_img]                                      
            self.data['gt_depth'] = self.data['gt_depth'][:self.num_eval_img]
              

        # ----------------------------------------
        # build index mapping: global idx <-> camera frame idx
        # ----------------------------------------
        self.global_to_camera_idx = {}   # global_idx -> (scene_id, camera_id, cam_frame_idx)
        self.camera_to_global_idx = {}   # (scene_id, camera_id, cam_frame_idx) -> global_idx

        for global_idx, hq_path in enumerate(self.data['hq_img']):
            scene_id  = hq_path.split('/')[-4]
            camera_id = hq_path.split('/')[-2]   # cam_00
            frame_id  = int(hq_path.split('.')[-3])  # frame.XXXX

            key = (scene_id, camera_id)

            if key not in self.camera_to_global_idx:
                self.camera_to_global_idx[key] = {}

            self.camera_to_global_idx[key][frame_id] = global_idx
            self.global_to_camera_idx[global_idx] = (scene_id, camera_id, frame_id)



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


    def convert_hdf5_img(self, hdf5_path):
        gamma = 1.0 / 2.2
        inv_gamma = 1.0 / gamma
        percentile = 90
        brightness_nth_percentile_desired = 0.8
        eps = 0.0001
        try:
            with h5py.File(hdf5_path, "r") as f:
                rgb_color = f["dataset"][:].astype(np.float32) # [H, W, 3]
            entity_path = hdf5_path.replace("final_hdf5", "geometry_hdf5").replace(".color.hdf5", ".render_entity_id.hdf5")
            if os.path.exists(entity_path):
                with h5py.File(entity_path, "r") as f:
                    render_entity_id = f["dataset"][:].astype(np.int32)
                valid_mask = render_entity_id != -1
            else:
                valid_mask = np.ones(rgb_color.shape[:2], dtype=bool)
        except Exception as e:
            print(f"Error loading {hdf5_path}: {e}")
            return None
        brightness = 0.3 * rgb_color[:, :, 0] + 0.59 * rgb_color[:, :, 1] + 0.11 * rgb_color[:, :, 2]
        brightness_valid = brightness[valid_mask]
        if len(brightness_valid) == 0 or np.percentile(brightness_valid, percentile) < eps:
            scale = 1.0
        else:
            current_p = np.percentile(brightness_valid, percentile)
            scale = np.power(brightness_nth_percentile_desired, inv_gamma) / current_p
        rgb_tm = np.power(np.maximum(scale * rgb_color, 0), gamma)
        rgb_tm = np.clip(rgb_tm, 0.0, 1.0)
        rgb_tm = (rgb_tm * 255.0).round().astype(np.uint8)
        return rgb_tm
    
    
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
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = depth.shape
        def nearest_multiple(x, p):
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down
        new_w = max(1, nearest_multiple(w, patch_size))
        new_h = max(1, nearest_multiple(h, patch_size))
        if new_w != w or new_h != h:
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return depth

    def convert_hdf5_depth(self, hdf5_path):
        W, H, FOCAL = 1024, 768, 886.81
        with h5py.File(hdf5_path, "r") as f:
            dist = f["dataset"][:]
        x = np.linspace(-W / 2 + 0.5, W / 2 - 0.5, W)
        y = np.linspace(-H / 2 + 0.5, H / 2 - 0.5, H)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, FOCAL)
        plane_norm = np.sqrt(X**2 + Y**2 + Z**2)
        gt_depth = dist / plane_norm * FOCAL
        gt_depth = np.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)
        return gt_depth.astype(np.float32)


    def get_random_ids(self, anchor, num_frames):
        """
        Global random sampling baseline.
        """
        if num_frames == 1:
            return np.array([anchor], dtype=np.int64)
        N = len(self.data['hq_img'])
        candidates = np.arange(N)
        candidates = np.delete(candidates, anchor)
        K = num_frames - 1
        sampled = np.random.choice(candidates, size=K,replace=(len(candidates) < K),)
        return np.concatenate([[anchor], sampled]).astype(np.int64)


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
        sampled = np.random.choice(candidates,size=num_frames - 1,replace = (len(candidates) < num_frames - 1),)
        return np.concatenate([[anchor], sampled])


    def get_nearby_ids_camera(self, anchor, num_frames, expand_ratio=2.0):
        if num_frames == 1:
            return np.array([anchor], dtype=np.int64)
        # ----------------------------
        # map global idx -> camera frame
        # ----------------------------
        scene_id, camera_id, cam_frame_idx = self.global_to_camera_idx[anchor]
        tmp_camera_id = f"cam_{camera_id.split('_')[-3]}"
        # get ranking
        ranking = self.camera_rankings[(scene_id, tmp_camera_id)]
        cam_neighbors = ranking[cam_frame_idx]
        # skip itself
        cam_neighbors = cam_neighbors[1:]
        # only keep frames that exist in dataset
        valid_frames = self.camera_to_global_idx[(scene_id, camera_id)]
        cam_neighbors = [i for i in cam_neighbors if i in valid_frames]
        # optionally expand candidate pool
        max_candidates = int(num_frames * expand_ratio)
        cam_neighbors = cam_neighbors[:max_candidates]
        # pick K-1 neighbors
        K = num_frames - 1
        
        # no randomness and only sample the closest views
        # selected_cam_idxs = cam_neighbors[:K]
        
        # add randomness when sampling from neighboring views 
        selected_cam_idxs = np.random.choice(cam_neighbors, size=K, replace=len(cam_neighbors) < K)
        # ----------------------------
        # map camera frame idx -> global idx
        # ----------------------------
        global_indices = [anchor]
        for cam_i in selected_cam_idxs:
            global_i = self.camera_to_global_idx[(scene_id, camera_id)][cam_i]
            global_indices.append(global_i)
        return np.array(global_indices, dtype=np.int64)



    def load_camera_pose(self, scene_id, camera_id, frame_idx):
        camera_id = camera_id.split('_')[1:3]
        camera_id = '_'.join(camera_id)
        return self.camera_extrinsics[(scene_id, camera_id)][frame_idx]



    def __getitem__(self, items):
        
        idx, num_input_view = items
        # print(f'hypersim - {num_input_view}')
        

        # view selection strategy        
        if self.view_sel.strategy == 'random':
            frame_ids = self.get_random_ids(anchor=idx, num_frames=num_input_view)
        elif self.view_sel.strategy == 'near_random':
            frame_ids = self.get_nearby_ids_random(anchor=idx, num_frames=num_input_view, expand_ratio=self.view_sel.expand_ratio)
        elif self.view_sel.strategy == 'near_camera':
            frame_ids = self.get_nearby_ids_camera(anchor=idx, num_frames=num_input_view, expand_ratio=self.view_sel.expand_ratio)
    
        
        outputs={}
        outputs['frame_ids'] = frame_ids
        

        # =====================================================
        #  camera poses
        # =====================================================
        pose_list = []
        for global_i in frame_ids:
            scene_id, camera_id, frame_id = self.global_to_camera_idx[global_i]
            pose = self.load_camera_pose(scene_id, camera_id, frame_id)
            pose_list.append(pose)

        outputs['poses'] = np.stack(pose_list, axis=0)  # (V,4,4)
        
        
        # ----------------------
        #       process hq
        # ----------------------
        hq_view_id=[] 
        hq_view_list=[]
        if 'hq_img' in self.data.keys():
            hq_views = [self.data['hq_img'][i] for i in frame_ids]
            for hq_view in hq_views:
                volume = hq_view.split('/')[-4].split('_')[-2]
                scene = hq_view.split('/')[-4].split('_')[-1]
                camera = hq_view.split('/')[-2].split('_')[-3]
                view_id = hq_view.split('/')[-1].split('.')[-3]
                hq_view_id.append(f'hypersim_{volume}_{scene}_{camera}_{view_id}')
                hq_view_list.append(self.resize(self.convert_hdf5_img(hq_view)))
                # hq_view_list.append({f'hypersim_{volume}_{scene}_{camera}_{view_id}': self.convert_hdf5(hq_view)})
            outputs['hq_ids'] = hq_view_id
            outputs['hq_views'] = hq_view_list


        # ----------------------------------
        #       get lq on the fly
        # ----------------------------------
        lq_view_id=[]
        lq_view_list=[]
        

        hq_views = [self.data['hq_img'][i] for i in frame_ids]
        for global_i, hq_view in zip(frame_ids, hq_views):
            volume = hq_view.split('/')[-4].split('_')[-2]
            scene = hq_view.split('/')[-4].split('_')[-1]
            camera = hq_view.split('/')[-2].split('_')[-3]
            view_id = hq_view.split('/')[-1].split('.')[-3]
            lq_view_id.append(f'hypersim_{volume}_{scene}_{camera}_{view_id}')
            img_pil = self.convert_hdf5_img(hq_view)

            if isinstance(self.data_cfg.lq_kernel_size, int):
                KERNEL_SIZE = self.data_cfg.lq_kernel_size
            else:
                range_start, range_end = self.data_cfg.lq_kernel_size
                mean = (range_start + range_end) // 2
                standard_deviation = (range_end - mean) / 1.96
                KERNEL_SIZE = int(np.clip(np.random.normal(mean, standard_deviation), range_start, range_end))

            if isinstance(self.data_cfg.lq_intensity_range, (int, float)):
                BLUR_INTENSITY = float(self.data_cfg.lq_intensity_range)
            else:
                BLUR_INTENSITY = random.uniform(*self.data_cfg.lq_intensity_range)            
            # print(f"Using kernel size {KERNEL_SIZE} and intensity {BLUR_INTENSITY}")

            if self.mode == 'val':
                np.random.seed(int(global_i))
                random.seed(int(global_i))
            kernel = Kernel(size=(KERNEL_SIZE, KERNEL_SIZE), intensity=BLUR_INTENSITY)
            blurred = kernel.applyTo(img_pil, keep_image_dim=True)
            blurred = np.array(blurred)
            lq_view_list.append(self.resize(blurred))
        outputs['lq_ids'] = lq_view_id 
        outputs['lq_views'] = lq_view_list


        # # -------------------------
        # #       process depth
        # # -------------------------
        # depth_view_id=[]
        # depth_view_list=[]
        # if 'gt_depth' in self.data.keys():
        #     depth_views = [self.data['gt_depth'][i] for i in frame_ids]
        #     for depth_view in depth_views:
        #         volume = depth_view.split('/')[-4].split('_')[-2]
        #         scene = depth_view.split('/')[-4].split('_')[-1]
        #         camera = depth_view.split('/')[-2].split('_')[-3]
        #         view_id = depth_view.split('/')[-1].split('.')[-3]
        #         depth_view_id.append(f'hypersim_{volume}_{scene}_{camera}_{view_id}')
        #         depth_view_list.append(self.resize_depth(self.convert_hdf5_depth(depth_view)))
        #     outputs['gt_depth_ids'] = depth_view_id
        #     outputs['gt_depths'] = depth_view_list


        return outputs
       

    def __len__(self):
        return len(self.data['hq_img'])
