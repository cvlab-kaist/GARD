import os 
import cv2 
import torch 
from torchvision import transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from copy import deepcopy
from RAE.src.utils.model_utils import instantiate_from_config

import numpy as np 
from PIL import Image 

from mvr.dataset.pho_sampler import PhoSampler, PhoBatchSampler
from mvr.dataset.pho_concat_ds import PhoConcatDataset, multiview_collate_fn


def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_denoiser: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None
):
    # state = {
    #     "step": step,
    #     "epoch": epoch,
    #     "model": model.module.state_dict(),
    #     "ema": ema_denoiser.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    #     "scheduler": scheduler.state_dict() if scheduler is not None else None,
    # }
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_denoiser.state_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    
    


# def load_checkpoint(
#     path: str,
#     model: DDP,
#     ema_denoiser: torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     scheduler=None,
# ):
#     checkpoint = torch.load(path, map_location="cpu")
#     model.module.load_state_dict(checkpoint["model"], strict=False)
#     ema_denoiser.load_state_dict(checkpoint["ema"], strict=False)
#     optimizer.load_state_dict(checkpoint["optimizer"])
#     if scheduler is not None and checkpoint.get("scheduler") is not None:
#         scheduler.load_state_dict(checkpoint["scheduler"])
#     return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def load_checkpoint(
    path: str,
    model: DDP,
    ema_denoiser: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    verbose: bool = True,
):
    checkpoint = torch.load(path, map_location="cpu")

    # --- load states ---
    # Release/eval-only checkpoints (e.g. release_weights/*.pt) only carry
    # "ema" and drop the raw "model" key. Fall back to seeding the trainable
    # model from the EMA weights in that case -- this is a fresh fine-tune
    # init, not a bit-exact resume (the optimizer is never restored either,
    # see below), so starting both model and EMA from the same EMA weights
    # is the correct behavior here.
    is_fresh_finetune_init = "model" not in checkpoint
    if not is_fresh_finetune_init:
        init_state = checkpoint["model"]
    elif "ema" in checkpoint:
        print(f"[load_checkpoint] No 'model' key in {path}; initializing trainable "
              f"model from 'ema' weights instead (fresh fine-tune init: fresh "
              f"optimizer, epoch/step reset to 0).")
        init_state = checkpoint["ema"]
    else:
        raise KeyError(f"Checkpoint {path} has neither 'model' nor 'ema' weights.")
    model_missing, model_unexpected = model.module.load_state_dict(
        init_state, strict=False
    )
    ema_missing, ema_unexpected = ema_denoiser.load_state_dict(
        checkpoint["ema"], strict=False
    )

    # TODO: pho - comment for now as it raises error when model params and optim params differ
    #optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    # Fresh fine-tune init (ema-only release checkpoint) always starts the
    # epoch/step counters at 0, even if the source checkpoint carries stale
    # step/epoch metadata from the original training run. A true resume
    # (raw "model" key present) still restores the saved epoch/step.
    if is_fresh_finetune_init:
        epoch, step = 0, 0
    else:
        epoch = checkpoint.get("epoch", 0)
        step = checkpoint.get("step", 0)

    # --- neat logging ---
    if verbose:
        print("=" * 60)
        print(f"  Loaded checkpoint: {path}")
        print("-" * 60)
        print(f"  Epoch: {epoch}")
        print(f"  Step : {step}")
        print()

        def _print_keys(title, missing, unexpected):
            print(f"🔹 {title}")
            if not missing and not unexpected:
                print("   ✓ All parameters matched\n")
                return

            if missing:
                print(f"   ⚠ Missing keys ({len(missing)}):")
                for k in missing:
                    print(f"     - {k}")
            if unexpected:
                print(f"   ⚠ Unexpected keys ({len(unexpected)}):")
                for k in unexpected:
                    print(f"     - {k}")
            print()

        _print_keys("Model", model_missing, model_unexpected)
        _print_keys("EMA Model", ema_missing, ema_unexpected)

        print("🔹 Optimizer state: loaded ✓")
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            print("🔹 Scheduler state: loaded ✓")
        print("=" * 60)

    return epoch, step



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


def load_train_data(
    cfg,
    batch_size: int,
    rank: int,
    world_size: int,
):
    
    # load dataset
    if 'imagenet' in cfg.data.train.list:
        from torchvision.datasets import ImageFolder
        transform = T.Compose([
            T.Lambda(lambda pil_image: center_crop_arr(pil_image, cfg.data.train.image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        dataset = ImageFolder(str(cfg.data.train.imagenet.hq_root_path), transform=transform)
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=cfg.training.num_workers, pin_memory=True, drop_last=False)
        
    else:
        datasets=[]
        if 'hypersim' in cfg.data.train.list:
            from mvr.dataset.pho_hypersim import PhoHypersim
            hypersim_ds = PhoHypersim(cfg.data.train.hypersim, mode='train')
            datasets.append(hypersim_ds)
        if 'tartanair' in cfg.data.train.list:
            from mvr.dataset.pho_tartanair import PhoTartanAir
            tartanair_ds = PhoTartanAir(cfg.data.train.tartanair, mode='train')
            datasets.append(tartanair_ds)
        train_ds = PhoConcatDataset(datasets, cfg, mode='train')
        train_sampler = PhoSampler(train_ds, shuffle=cfg.training.shuffle)
        train_batchsampler = PhoBatchSampler(sampler=train_sampler, batch_size=batch_size, max_num_input_view=cfg.data.train.max_num_input_view)
        train_loader = DataLoader(train_ds, batch_sampler=train_batchsampler, num_workers=cfg.training.num_workers, pin_memory=True, drop_last=False, collate_fn=multiview_collate_fn)
    return train_loader, train_sampler



def load_val_data(
    cfg,
    batch_size: int,
    rank: int,
    world_size: int,
):
    datasets=[]
    if 'hypersim' in cfg.data.val.list:
        from mvr.dataset.pho_hypersim import PhoHypersim
        hypersim_ds = PhoHypersim(cfg.data.val.hypersim, mode='val')
        datasets.append(hypersim_ds)
    if 'tartanair' in cfg.data.val.list:
        from mvr.dataset.pho_tartanair import PhoTartanAir
        tartanair_ds = PhoTartanAir(cfg.data.val.tartanair, mode='val')
        datasets.append(tartanair_ds)
    if 'eth3d' in cfg.data.val.list:
        from mvr.dataset.pho_eth3d import PhoETH3D 
        eth3d_ds = PhoETH3D(cfg.data.val.eth3d, mode='val')
        datasets.append(eth3d_ds)
        
    val_ds = PhoConcatDataset(datasets, cfg, mode='val')
    val_sampler = PhoSampler(val_ds, shuffle=True)
    val_batchsampler = PhoBatchSampler(sampler=val_sampler, batch_size=batch_size, max_num_input_view=cfg.data.val.max_num_input_view)
    val_loader = DataLoader(val_ds, batch_sampler=val_batchsampler, num_workers=cfg.training.num_workers, pin_memory=True, drop_last=False, collate_fn=multiview_collate_fn)
    return val_loader, val_sampler



def load_model(cfg, rank, device):


    models={}
    processors={}

        
    ##### Encoder initalization 
    if cfg.stage_1.model == 'da3':
        from depth_anything_3.api import DepthAnything3
        from depth_anything_3.utils.io.input_processor import InputProcessor
        from depth_anything_3.utils.io.output_processor import OutputProcessor
        encoder_input_processor = InputProcessor()
        encoder_output_processor = OutputProcessor()
        encoder = DepthAnything3.from_pretrained(cfg.stage_1.da3.ckpt).to(device)
        encoder.eval()
    elif cfg.stage_1.model == 'vggt':
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        encoder = VGGT.from_pretrained(cfg.stage_1.vggt.ckpt).to(device)
        encoder.eval()
    if cfg.stage_1.model == 'vae':
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(cfg.stage_1.vae.ckpt, subfolder="vae").to(device)
        vae.requires_grad_(False)
        models['vae'] = vae
        # # load da3 for eval
        # from depth_anything_3.api import DepthAnything3
        # from depth_anything_3.utils.io.input_processor import InputProcessor
        # from depth_anything_3.utils.io.output_processor import OutputProcessor
        # encoder_input_processor = InputProcessor()
        # encoder_output_processor = OutputProcessor()
        # encoder = DepthAnything3.from_pretrained(cfg.stage_1.da3.ckpt).to(device)
        # encoder.eval()
        encoder=None
        encoder_input_processor=None
        encoder_output_processor=None
        
                
    models['encoder'] = encoder
    processors['encoder_input_processor'] = encoder_input_processor
    processors['encoder_output_processor'] = encoder_output_processor


    ##### Denoiser initialization 
    from stage2.models import Stage2ModelProtocol
    denoiser: Stage2ModelProtocol = instantiate_from_config(cfg.stage_2).to(device)         
    ema_denoiser = deepcopy(denoiser).to(device)
    ema_denoiser.requires_grad_(False)
    ema_denoiser.eval()
    denoiser.requires_grad_(True) # train stage2 model
    # ddp_denoiser = DDP(denoiser, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=True)
    
    
    ddp_denoiser = DDP(
        denoiser,
        device_ids=[device.index],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    
    ddp_denoiser._set_static_graph()
    

    ddp_denoiser.train()
    
    models['ema_denoiser'] = ema_denoiser
    models['ddp_denoiser'] = ddp_denoiser
    models['denoiser'] = ddp_denoiser.module
    
    return models, processors




def load_sampler(cfg, transport_sampler):
    
    sampler_mode = cfg.sampler.mode
    # sampler_params = dict(cfg.sampler.get("params", {}))
    sampler_params = cfg.sampler.params
    
    if sampler_mode == "ODE":
        eval_sampler = transport_sampler.sample_ode(**sampler_params)   # return _ode.sample from transport.py sample_ode()
    elif sampler_mode == "SDE":
        eval_sampler = transport_sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")
    
    
    return eval_sampler 