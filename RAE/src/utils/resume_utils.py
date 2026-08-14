import os
import sys
from omegaconf import OmegaConf
import logging
from typing import Optional, Tuple
import shutil
from .wandb_utils import initialize_wandb, create_logger
import logging

def configure_experiment_dirs(cfg, rank) -> Tuple[str, str, logging.Logger]:
    experiment_name = f'TRAIN__{cfg.training.precision}__{"-".join(cfg.data.train.list)}__{cfg.data.train.hypersim.view_selection.strategy}-{cfg.data.train.tartanair.view_selection.strategy}__{cfg.stage_1.model}-{cfg.stage_1.model_size}-extractfeat{cfg.mvrm.train.extract_feat_layers[0]}-{cfg.stage_2.model}__bs{cfg.training.global_batch_size}-maxview{cfg.data.train.max_num_input_view}-accum{cfg.training.grad_accum_steps}__lr-{cfg.training.optimizer.lr:.0e}__msg-{cfg.log.tracker.wandb.msg}'
    # experiment_name = os.environ.get("EXPERIMENT_NAME")
    # assert experiment_name is not None, "Please set the EXPERIMENT_NAME environment variable."
    experiment_dir = os.path.join(cfg.log.result_root_dir, "-".join(cfg.data.train.list), experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints") 
    if rank == 0:
        os.makedirs(cfg.log.result_root_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, 'rae')
        logger.info(f"Experiment directory created at {experiment_dir}")
        if cfg.log.tracker.name == 'wandb':
            gpu = os.environ.get("CUDA", "")
            entity = cfg.log.tracker.wandb.entity
            project = cfg.log.tracker.wandb.project
            wandb_exp_name = f'g{gpu}__{experiment_name}'
            initialize_wandb(cfg, entity, wandb_exp_name, project)
    else:
        logger = create_logger(None, 'rae')
    return experiment_dir, checkpoint_dir, logger
def find_resume_checkpoint(resume_dir) -> Optional[str]:
    """
    Find the latest checkpoint file in the resume directory.
    Args:
        resume_dir (str): Path to the resume directory.
    Returns:
        str: Path to the latest checkpoint file.
    """
    if not os.path.exists(resume_dir):
        raise ValueError(f"Resume directory {resume_dir} does not exist.")
    checkpoint_dir = os.path.join(resume_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist.")
    checkpoints = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt") or f.endswith(".ckpt") or f.endswith(".safetensor")
    ]
    if len(checkpoints) == 0:
        return None
    # sort via epoch, saved as 'ep-{epoch:07d}.pt'
    checkpoints = sorted(
        checkpoints,
        key=lambda x: int(os.path.basename(x).split("-")[1].split(".")[0]),
    )
    return checkpoints[-1]

def save_worktree(
    path: str,
    config: OmegaConf,  
) -> None:
    # breakpoint()
    OmegaConf.save(config, os.path.join(path, "config.yaml"))
    # worktree_path = os.path.join(os.getcwd(), "RAE/src")
    # shutil.copytree(worktree_path, os.path.join(path, "src/"), dirs_exist_ok=True)
    # print(f'Worktree {worktree_path} saved to {os.path.join(path, "src/")}')

if __name__ == "__main__":
    experiment_dir = sys.argv[1]
    ckpt_path = find_resume_checkpoint(experiment_dir)
    print(f"Latest checkpoint found at: {ckpt_path}")