import torch 
import torchvision.transforms as T 
from torch.utils.data import Dataset 
import bisect 


class PhoConcatDataset(Dataset):
    """
    Meta-style composed dataset.
    Routes (idx, image_num, aspect_ratio) to the correct sub-dataset.
    """

    def __init__(self, datasets, cfg, mode='train'):

        self.datasets = datasets
        self.cumulative_sizes = self._compute_cumulative_sizes()
        
        # if mode == 'train':
        #     self.num_input_view = cfg.data.train.num_input_view 
        # elif mode == 'val':
        #     self.num_input_view = cfg.data.val.num_input_view 


    def _compute_cumulative_sizes(self):
        sizes = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            sizes.append(total)
        return sizes

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _get_dataset_index(self, global_idx):
        """
        Convert global idx -> (dataset_id, local_idx)
        """
        dataset_id = bisect.bisect_right(self.cumulative_sizes, global_idx)
        prev_size = 0 if dataset_id == 0 else self.cumulative_sizes[dataset_id - 1]
        local_idx = global_idx - prev_size
        return dataset_id, local_idx


    def __getitem__(self, items):
        """
        items = (global_idx, num_input_view)
        """
        global_idx, num_input_view = items

        dataset_id, local_idx = self._get_dataset_index(global_idx)
        return self.datasets[dataset_id][(local_idx, num_input_view)]


    # def __getitem__(self, items):
    #     """
    #     items = (global_idx, image_num, aspect_ratio)
    #     """
        
    #     breakpoint()
    #     # global_idx, image_num, aspect_ratio = items
    #     global_idx = items 
    #     # print('concat dataset (global): ', global_idx, image_num, aspect_ratio)

    #     dataset_id, local_idx = self._get_dataset_index(global_idx)
    #     # print('concat dataset (local): ', dataset_id, local_idx)

    #     # breakpoint()
    #     # return self.datasets[dataset_id][(local_idx, image_num, aspect_ratio)]
    #     return self.datasets[dataset_id][(local_idx, self.num_input_view)]
    



IMAGENET_NORMALIZE = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def to_tensor(img):
    """
    img: np.ndarray (H,W,3) uint8 RGB
    returns: torch.FloatTensor (3,H,W) ImageNet-normalized
    """
    x = (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .contiguous()
        .float() / 255.0
    )
    return x
    # return IMAGENET_NORMALIZE(x)
    
    

def multiview_collate_fn(batch):
    """
    batch: List[dict], length = B
    """

    outputs={}

    # ------------------------
    # frame ids (tensor)
    # ------------------------
    frame_ids = torch.stack([torch.as_tensor(b["frame_ids"]) for b in batch],dim=0)  # (B, V)
    outputs['frame_ids'] = frame_ids

    # ------------------------
    # ids (keep Python lists)
    # ------------------------
    hq_ids = [b["hq_ids"] for b in batch if 'hq_ids' in b.keys()]                              # [B][V]
    hq_latent_ids = [b["hq_latent_ids"] for b in batch if 'hq_latent_ids' in b.keys() ]        # [B][V]
    lq_ids = [b["lq_ids"] for b in batch if 'lq_ids' in b.keys()]                              # [B][V]
    gt_depth_ids = [b["gt_depth_ids"] for b in batch if 'gt_depth_ids' in b.keys()]            # [B][V]
    
    
    
    # ------------------------
    # HQ images
    # ------------------------
    if 'hq_views' in batch[0].keys():
        hq_views = []
        for b in batch:
            v_imgs = [to_tensor(img) for img in b["hq_views"]]     # [V,3,H,W]
            hq_views.append(torch.stack(v_imgs, dim=0))
        hq_views = torch.stack(hq_views, dim=0)                    # (B,V,3,H,W)
        outputs['hq_views'] = hq_views
        outputs['hq_ids'] = hq_ids


    # ------------------------
    # LQ images
    # ------------------------
    if 'lq_views' in batch[0].keys():
        lq_views = []
        for b in batch:
            v_imgs = [to_tensor(img) for img in b["lq_views"]]     # [V,3,H,W]
            lq_views.append(torch.stack(v_imgs, dim=0))
        lq_views = torch.stack(lq_views, dim=0)                    # (B,V,3,H,W)
        outputs['lq_views'] = lq_views
        outputs['lq_ids'] = lq_ids

    # ------------------------
    # Depth
    # ------------------------
    if 'gt_depths' in batch[0].keys():
        gt_depths = []
        for b in batch:
            v_depths = [
                torch.from_numpy(d).unsqueeze(0)                   # (1,H,W)
                for d in b["gt_depths"]
            ]
            gt_depths.append(torch.stack(v_depths, dim=0))         # (V,1,H,W)
        gt_depths = torch.stack(gt_depths, dim=0)                  # (B,V,1,H,W)
        outputs['gt_depths'] = gt_depths
        outputs['gt_depth_ids'] = gt_depth_ids
        


    # ------------------------
    # Poses
    # ------------------------
    if 'poses' in batch[0].keys():
        poses = torch.stack(
            [torch.from_numpy(b['poses']).float() for b in batch],
            dim=0
        )  # (B, V, 4, 4)
        outputs['poses'] = poses


    # ------------------------
    # return
    # ------------------------
    return outputs
    return {
        "frame_ids": frame_ids,

        "hq_ids": hq_ids,
        "hq_views": hq_views,

        "hq_latent_ids": hq_latent_ids,
        "hq_latent_views": hq_latent_views,

        "lq_ids": lq_ids,
        "lq_views": lq_views,

        "gt_depth_ids": gt_depth_ids,
        "gt_depths": gt_depths,
    }

