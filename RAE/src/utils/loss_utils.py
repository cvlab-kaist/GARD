

import logging
import torch
import torch.nn.functional as F

def velocity_direction_loss(pred_vel, gt_vel, eps=1e-8):
    """
    Cosine direction alignment loss for velocity field.

    Args:
        pred_vel: predicted velocity from model
                  shape: (b, v, n, d)
        gt_vel: ground-truth velocity ut from transport
                shape: (b, v, n, d)
        eps: small constant for numerical stability

    Returns:
        scalar loss
    """

    b, v, n, d = pred_vel.shape

    # Flatten spatial dimensions
    pred_flat = pred_vel.reshape(b * v, -1)
    gt_flat   = gt_vel.reshape(b * v, -1)

    # Normalize vectors (safe normalization)
    pred_norm = pred_flat / (pred_flat.norm(dim=1, keepdim=True) + eps)
    gt_norm   = gt_flat   / (gt_flat.norm(dim=1, keepdim=True) + eps)

    cos_sim = F.cosine_similarity(pred_flat, gt_flat, dim=1, eps=eps)
    loss = (1 - cos_sim).mean()

    return loss



def directional_alignment_loss(restored_latent, hq_latent, lq_latent):
    # all shape: (b, v, n, d)

    # compute direction vectors
    direction_pred = restored_latent - lq_latent
    direction_gt   = hq_latent - lq_latent

    # flatten spatial dims for cosine
    b, v, n, d = direction_pred.shape
    direction_pred = direction_pred.view(b*v, -1)
    direction_gt   = direction_gt.view(b*v, -1)

    cos_sim = F.cosine_similarity(direction_pred, direction_gt, dim=-1)

    # maximize cosine similarity → minimize (1 - cosine)
    loss = (1 - cos_sim).mean()

    return loss



def cross_view_latent_consistency(restored_latent):
    # restored_latent: (b, v, n, d)

    b, v, n, d = restored_latent.shape

    # pairwise difference between adjacent views
    loss = 0.0
    count = 0

    for i in range(v - 1):
        zi = restored_latent[:, i]     # (b, n, d)
        zj = restored_latent[:, i+1]

        loss += ((zi - zj) ** 2).mean()
        count += 1

    return loss / count


# from vggt 
def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL



def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.
    
    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside 
                                  [-hard_max, hard_max] will be clamped. If None, 
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor
    
    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        logging.warning(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor