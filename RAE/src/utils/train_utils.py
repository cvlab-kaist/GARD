import os 
import sys 
sys.path.append(os.getcwd())
from omegaconf import OmegaConf, DictConfig
from typing import List, Tuple, Union
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from copy import deepcopy
from .dist_utils import setup_distributed


def parse_configs(full_cfg: Union[DictConfig, str]) -> Tuple[DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig]:
    """Load a config file and return component sections as DictConfigs."""
    if isinstance(full_cfg, str):
        full_cfg = OmegaConf.load(full_cfg)

    # load stage1 config
    if full_cfg.stage_1.model == 'rae':
        print('stage1_config: RAE')
        stage1_cfg = full_cfg.stage_1.get("rae", None)
    elif full_cfg.stage_1.model == 'da3':
        print('stage1_config: DA3')
        stage1_cfg = full_cfg.stage_1.get("da3", None)
    elif full_cfg.stage_1.model == 'vggt':
        print('stage1_config: VGGT')
        stage1_cfg = full_cfg.stage_1.get("vggt", None)

    stage2_cfg = full_cfg.get("stage_2", None)
    transport_config = full_cfg.get("transport", None)
    sampler_config = full_cfg.get("sampler", None)
    guidance_config = full_cfg.get("guidance", None)
    misc = full_cfg.get("misc", None)
    training_config = full_cfg.get("training", None)
    eval_config = full_cfg.get("eval", None)
    return stage1_cfg, stage2_cfg, transport_config, sampler_config, guidance_config, misc, training_config, eval_config

def none_or_str(value):
    if value == 'None':
        return None
    return value

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if not param.requires_grad:
            continue
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def prepare_dataloader(
    cfg,
    batch_size: int,
    workers: int,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, DistributedSampler]:
    
    # load dataset
    if 'imagenet' in cfg.data.train.list:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, cfg.data.train.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        dataset = ImageFolder(str(cfg.data.train.imagenet.hq_root_path), transform=transform)
    
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=workers, pin_memory=True, drop_last=False)
        
    else:
        from mvr.dataset.pho_sampler import PhoSampler, PhoBatchSampler
        from mvr.dataset.pho_concat_ds import PhoConcatDataset, multiview_collate_fn
        
        datasets=[]
        if 'hypersim' in cfg.data.train.list:
            from mvr.dataset.pho_hypersim import PhoHypersim
            hypersim_ds = PhoHypersim(cfg.data.train.hypersim)
            datasets.append(hypersim_ds)
        
        if 'tartanair' in cfg.data.train.list:
            from mvr.dataset.pho_tartanair import PhoTartanAir
            tartanair_ds = PhoTartanAir(cfg.data.train.tartanair)
            datasets.append(tartanair_ds)
            
        train_ds = PhoConcatDataset(datasets, cfg)
        train_sampler = PhoSampler(train_ds, shuffle=cfg.training.shuffle)
        train_batchsampler = PhoBatchSampler(sampler=train_sampler, batch_size=batch_size)
        
        train_loader = DataLoader(train_ds, batch_sampler=train_batchsampler, num_workers=workers, pin_memory=True, drop_last=False, collate_fn=multiview_collate_fn)
        
    return train_loader, train_sampler


def get_autocast_scaler(cfg) -> Tuple[dict, torch.cuda.amp.GradScaler | None]:
    if cfg.training.precision == "fp16":
        scaler = GradScaler()
        autocast_kwargs = dict(enabled=True, dtype=torch.float16)
    elif cfg.training.precision == "bf16":
        scaler = None
        autocast_kwargs = dict(enabled=True, dtype=torch.bfloat16)
    else:
        scaler = None
        autocast_kwargs = dict(enabled=False)
    
    return scaler, autocast_kwargs
