import os
import numpy as np
import torch
import matplotlib.pyplot as plt


from depth_anything_3.bench.utils import align_to_first_camera


# -----------------------------------------------------------
# Helper: convert (V,3,4) → (V,4,4)
# -----------------------------------------------------------
def to_4x4(se3_3x4: torch.Tensor):
    if se3_3x4.shape[-2:] == (4, 4):
        return se3_3x4
    V = se3_3x4.shape[0]
    bottom = torch.tensor([0,0,0,1], device=se3_3x4.device).view(1,1,4).repeat(V,1,1)
    return torch.cat([se3_3x4, bottom], dim=1)


def plot_all_cam_trajectory_fair(
    pose1,
    pose2,
    pose3,
    pose4,
    labels,
    line_widths,
    colors,
    save_path,
    only_pred=False,
    visualize_direction=True,
    arrow_len_3d=0.2,
    arrow_scale_2d=20
):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    poses = [pose1, pose2, pose3, pose4]
    labels = labels
    colors = colors
    styles = ['-o', ':o', '--x', '-.^']

    centers = []
    directions = []

    # ----------------------------
    # Preprocess poses
    # ----------------------------
    for pose in poses:

        pose = torch.tensor(pose) if not isinstance(pose, torch.Tensor) else pose
        pose = to_4x4(pose)
        pose = align_to_first_camera(pose)

        c, d = extract_view_directions(pose)

        centers.append(c.cpu().numpy())
        directions.append(d.cpu().numpy())

    # ----------------------------
    # Plotting
    # ----------------------------
    fig = plt.figure(figsize=(20, 10))
    step = max(len(centers[0]) // 50, 1)

    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')

    for i, (c, d) in enumerate(zip(centers, directions)):

        if only_pred and i == 0:
            continue

        ax1.plot(
            c[:,0], c[:,1], c[:,2],
            styles[i],
            color=colors[i],
            label=labels[i],
            markersize=5,
            linewidth=line_widths[i]
        )

        if visualize_direction:
            ax1.quiver(
                c[::step,0], c[::step,1], c[::step,2],
                d[::step,0], d[::step,1], d[::step,2],
                length=arrow_len_3d,
                color=colors[i],
                normalize=True
            )

    # start point
    ax1.scatter(
        centers[0][0,0], centers[0][0,1], centers[0][0,2],
        color='black', s=100, label='Start', zorder=10
    )

    ax1.set_title("3D Camera Trajectory")
    ax1.legend()

    # ----------------------------
    # Top-down plot
    # ----------------------------
    ax2 = fig.add_subplot(122)

    for i, (c, d) in enumerate(zip(centers, directions)):

        if only_pred and i == 0:
            continue

        ax2.plot(
            c[:,0], c[:,2],
            styles[i],
            color=colors[i],
            label=labels[i],
            markersize=5,
            linewidth=line_widths[i]
        )

        if visualize_direction:
            ax2.quiver(
                c[::step,0], c[::step,2],
                d[::step,0], d[::step,2],
                color=colors[i],
                scale=arrow_scale_2d
            )

    ax2.scatter(centers[0][0,0], centers[0][0,2], color='black', s=80)

    ax2.set_title("Top-down View")
    ax2.set_aspect('equal')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()






def plot_cam_trajectory_fair(
    hq_pred_pose,
    lq_pred_pose,
    res_pred_pose,
    save_path,
    only_pred=False,
    visualize_direction=True,
    arrow_len_3d=0.2,
    arrow_scale_2d=20,
):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ----------------------------
    # Convert to torch tensors
    # ----------------------------
    hq_pred_pose = torch.tensor(hq_pred_pose) if not isinstance(hq_pred_pose, torch.Tensor) else hq_pred_pose
    lq_pred_pose = torch.tensor(lq_pred_pose) if not isinstance(lq_pred_pose, torch.Tensor) else lq_pred_pose
    res_pred_pose = torch.tensor(res_pred_pose) if not isinstance(res_pred_pose, torch.Tensor) else res_pred_pose

    # ----------------------------
    # Convert to 4x4
    # ----------------------------
    hq_pred_pose = to_4x4(hq_pred_pose)
    lq_pred_pose = to_4x4(lq_pred_pose)
    res_pred_pose = to_4x4(res_pred_pose)

    # ----------------------------
    # Align exactly like metric
    # ----------------------------
    hq_pred_pose = align_to_first_camera(hq_pred_pose)
    lq_pred_pose = align_to_first_camera(lq_pred_pose)
    res_pred_pose = align_to_first_camera(res_pred_pose)

    # ----------------------------
    # Extract centers + directions
    # ----------------------------
    hq_pred_c, hq_pred_d = extract_view_directions(hq_pred_pose)
    lq_pred_c, lq_pred_d = extract_view_directions(lq_pred_pose)
    res_pred_c, res_pred_d = extract_view_directions(res_pred_pose)

    hq_pred_c, hq_pred_d = hq_pred_c.cpu().numpy(), hq_pred_d.cpu().numpy()
    lq_pred_c, lq_pred_d = lq_pred_c.cpu().numpy(), lq_pred_d.cpu().numpy()
    res_pred_c, res_pred_d = res_pred_c.cpu().numpy(), res_pred_d.cpu().numpy()

    # ----------------------------
    # Plotting
    # ----------------------------
    fig = plt.figure(figsize=(20, 10))
    step = max(len(hq_pred_c) // 50, 1)

    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')

    if not only_pred:
        ax1.plot(hq_pred_c[:,0], hq_pred_c[:,1], hq_pred_c[:,2],
                 'g-o', label='HQ', markersize=4, linewidth=2)

    ax1.plot(lq_pred_c[:,0], lq_pred_c[:,1], lq_pred_c[:,2],
             'r:o', label='LQ', markersize=3, linewidth=1.5)

    ax1.plot(res_pred_c[:,0], res_pred_c[:,1], res_pred_c[:,2],
             'b--x', label='Restored', markersize=3, alpha=0.6)

    ax1.scatter(hq_pred_c[0,0], hq_pred_c[0,1], hq_pred_c[0,2],
                color='black', s=100, label='Start', zorder=10)

    if visualize_direction:

        if not only_pred:
            ax1.quiver(hq_pred_c[::step,0], hq_pred_c[::step,1], hq_pred_c[::step,2],
                       hq_pred_d[::step,0], hq_pred_d[::step,1], hq_pred_d[::step,2],
                       length=arrow_len_3d, color='g', normalize=True)

        ax1.quiver(lq_pred_c[::step,0], lq_pred_c[::step,1], lq_pred_c[::step,2],
                   lq_pred_d[::step,0], lq_pred_d[::step,1], lq_pred_d[::step,2],
                   length=arrow_len_3d, color='r', normalize=True)

        ax1.quiver(res_pred_c[::step,0], res_pred_c[::step,1], res_pred_c[::step,2],
                   res_pred_d[::step,0], res_pred_d[::step,1], res_pred_d[::step,2],
                   length=arrow_len_3d, color='b', normalize=True)

    ax1.set_title("3D Camera Trajectory (metric-aligned)")
    ax1.legend()

    # Top-down view
    ax2 = fig.add_subplot(122)

    if not only_pred:
        ax2.plot(hq_pred_c[:,0], hq_pred_c[:,2], 'g-o', markersize=6, linewidth=4, label='HQ')

    ax2.plot(lq_pred_c[:,0],  lq_pred_c[:,2],  'r:o', markersize=5, linewidth=3, label='LQ')
    ax2.plot(res_pred_c[:,0], res_pred_c[:,2], 'b--x', markersize=5, linewidth=3, alpha=0.6, label='Restored')

    ax2.scatter(hq_pred_c[0,0], hq_pred_c[0,2], color='black', s=80, zorder=5)

    if visualize_direction:

        if not only_pred:
            ax2.quiver(hq_pred_c[::step,0], hq_pred_c[::step,2],
                       hq_pred_d[::step,0], hq_pred_d[::step,2],
                       color='green', scale=arrow_scale_2d)

        ax2.quiver(lq_pred_c[::step,0], lq_pred_c[::step,2],
                   lq_pred_d[::step,0], lq_pred_d[::step,2],
                   color='red', scale=arrow_scale_2d)

        ax2.quiver(res_pred_c[::step,0], res_pred_c[::step,2],
                   res_pred_d[::step,0], res_pred_d[::step,2],
                   color='blue', scale=arrow_scale_2d)

    ax2.set_title("Top-down View (metric-aligned)")
    ax2.set_aspect('equal')
    all_x = np.concatenate([hq_pred_c[:,0], lq_pred_c[:,0], res_pred_c[:,0]])
    all_z = np.concatenate([hq_pred_c[:,2], lq_pred_c[:,2], res_pred_c[:,2]])
    pad = 0.05
    x_span = all_x.max() - all_x.min()
    z_span = all_z.max() - all_z.min()
    half_span = max(x_span, z_span) / 2  # keep square for equal aspect
    cx = (all_x.min() + all_x.max()) / 2
    cz = (all_z.min() + all_z.max()) / 2
    ax2.set_xlim(cx - half_span * (1 + pad), cx + half_span * (1 + pad))
    ax2.set_ylim(cz - half_span * (1 + pad), cz + half_span * (1 + pad))
    ax2.set_facecolor('#e8e8e8')
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.grid(True, color='white', linewidth=1.0)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # -------------------------------------------------------
    # Helper: save a top-down-only figure for two trajectories
    # -------------------------------------------------------
    base, ext = os.path.splitext(save_path)

    def _save_topdown(c1, label1, color1, ls1, c2, label2, color2, ls2, suffix):
        fig_td, ax_td = plt.subplots(figsize=(8, 8))
        ax_td.plot(c1[:,0], c1[:,2], ls1, markersize=6, linewidth=2, label=label1, color=color1)
        ax_td.plot(c2[:,0], c2[:,2], ls2, markersize=5, linewidth=2, label=label2, color=color2)
        ax_td.scatter(c1[0,0], c1[0,2], color='black', s=80, zorder=5)
        ax_td.set_title(f"Top-down View — {label1} vs {label2}")
        ax_td.set_aspect('equal')
        all_x = np.concatenate([c1[:,0], c2[:,0]])
        all_z = np.concatenate([c1[:,2], c2[:,2]])
        x_span = all_x.max() - all_x.min()
        z_span = all_z.max() - all_z.min()
        half_span = max(x_span, z_span) / 2
        cx = (all_x.min() + all_x.max()) / 2
        cz = (all_z.min() + all_z.max()) / 2
        ax_td.set_xlim(cx - half_span * (1 + pad), cx + half_span * (1 + pad))
        ax_td.set_ylim(cz - half_span * (1 + pad), cz + half_span * (1 + pad))
        ax_td.set_facecolor('#e8e8e8')
        for spine in ax_td.spines.values():
            spine.set_visible(False)
        ax_td.grid(True, color='white', linewidth=1.0)
        ax_td.legend()
        plt.tight_layout()
        plt.savefig(f"{base}_{suffix}{ext}")
        plt.close()

    _save_topdown(hq_pred_c, 'HQ', 'black', 'k:',
                  lq_pred_c, 'Prediction', 'red',   'r-',  'topdown_hq_lq')
    _save_topdown(hq_pred_c, 'HQ', 'black', 'k:',
                  res_pred_c, 'Prediction', 'red', 'r-', 'topdown_hq_res')


def plot_cam_trajectory(
    hq_pred_pose,
    lq_pred_pose,
    res_pred_pose,
    save_path,
    only_pred=False,
    visualize_direction=True,  
    arrow_len_3d=0.2,
    arrow_scale_2d=20
):
    """
    hq_pred_pose: (V,4,4)
    lq_pred_pose: (V,3,4) or (V,4,4)
    res_pred_pose: (V,3,4) or (V,4,4)

    visualize_direction: whether to draw viewing direction arrows
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # =========================
    # Extract centers + directions
    # =========================
    hq_pred_c, hq_pred_d = extract_view_directions(hq_pred_pose)
    lq_pred_c, lq_pred_d = extract_view_directions(lq_pred_pose)
    res_pred_c, res_pred_d = extract_view_directions(res_pred_pose)

    hq_pred_c, hq_pred_d = hq_pred_c.cpu().numpy(), hq_pred_d.cpu().numpy()
    lq_pred_c, lq_pred_d = lq_pred_c.cpu().numpy(), lq_pred_d.cpu().numpy()
    res_pred_c, res_pred_d = res_pred_c.cpu().numpy(), res_pred_d.cpu().numpy()

    # =========================
    # Align predictions to GT
    # =========================
    lq_pred_c, R_hq = sim3_align(lq_pred_c, hq_pred_c)
    res_pred_c, R_lq = sim3_align(res_pred_c, hq_pred_c)

    lq_pred_d = (R_hq @ lq_pred_d.T).T
    res_pred_d = (R_lq @ res_pred_d.T).T

    # =========================
    # Plotting
    # =========================
    fig = plt.figure(figsize=(20, 10))
    step = max(len(hq_pred_c) // 50, 1)

    # -------------------------
    # 3D view
    # -------------------------
    ax1 = fig.add_subplot(121, projection='3d')

    if not only_pred:
        ax1.plot(hq_pred_c[:, 0], hq_pred_c[:, 1], hq_pred_c[:, 2],
                 'g-o', label='HQ', markersize=4, linewidth=2)

    ax1.plot(lq_pred_c[:, 0], lq_pred_c[:, 1], lq_pred_c[:, 2],
             'r:o', label='LQ', markersize=3, linewidth=1.5)

    ax1.plot(res_pred_c[:, 0], res_pred_c[:, 1], res_pred_c[:, 2],
             'b--x', label='Restored', markersize=3, alpha=0.6)

    ax1.scatter(hq_pred_c[0, 0], hq_pred_c[0, 1], hq_pred_c[0, 2],
                color='black', s=100, label='Start', zorder=10)

    # ---- Draw viewing directions (3D) ----
    if visualize_direction:
        if not only_pred:
            ax1.quiver(hq_pred_c[::step, 0], hq_pred_c[::step, 1], hq_pred_c[::step, 2],
                       hq_pred_d[::step, 0], hq_pred_d[::step, 1], hq_pred_d[::step, 2],
                       length=arrow_len_3d, color='g', normalize=True)

        ax1.quiver(lq_pred_c[::step, 0], lq_pred_c[::step, 1], lq_pred_c[::step, 2],
                   lq_pred_d[::step, 0], lq_pred_d[::step, 1], lq_pred_d[::step, 2],
                   length=arrow_len_3d, color='r', normalize=True)

        ax1.quiver(res_pred_c[::step, 0], res_pred_c[::step, 1], res_pred_c[::step, 2],
                   res_pred_d[::step, 0], res_pred_d[::step, 1], res_pred_d[::step, 2],
                   length=arrow_len_3d, color='b', normalize=True)

    ax1.set_title("3D Camera Trajectory")
    ax1.legend()

    # -------------------------
    # Top-down (X-Z)
    # -------------------------
    ax2 = fig.add_subplot(122)

    if not only_pred:
        ax2.plot(hq_pred_c[:, 0], hq_pred_c[:, 2],
                 'g-o', label='HQ', markersize=4, linewidth=2)

    ax2.plot(lq_pred_c[:, 0], lq_pred_c[:, 2],
             'r:o', label='LQ', markersize=3)

    ax2.plot(res_pred_c[:, 0], res_pred_c[:, 2],
             'b--x', label='Restored', alpha=0.5)

    ax2.scatter(hq_pred_c[0, 0], hq_pred_c[0, 2], color='black', s=80)

    # ---- Draw viewing directions (2D) ----
    if visualize_direction:
        if not only_pred:
            ax2.quiver(hq_pred_c[::step, 0], hq_pred_c[::step, 2],
                       hq_pred_d[::step, 0], hq_pred_d[::step, 2],
                       color='g', scale=arrow_scale_2d)

        ax2.quiver(lq_pred_c[::step, 0], lq_pred_c[::step, 2],
                   lq_pred_d[::step, 0], lq_pred_d[::step, 2],
                   color='r', scale=arrow_scale_2d,
                   headwidth=3, headlength=4, headaxislength=3.5)

        ax2.quiver(res_pred_c[::step, 0], res_pred_c[::step, 2],
                   res_pred_d[::step, 0], res_pred_d[::step, 2],
                   color='b', scale=arrow_scale_2d,
                   headwidth=2.5, headlength=3.5, headaxislength=3)

    ax2.set_title("Top-down View (X-Z)")
    ax2.set_aspect('equal')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    
    
    
    
# ==========================================================
# Utilities
# ==========================================================

def extract_view_directions(T):
    """
    T: (V,4,4) or (V,3,4)
    Returns:
        centers: (V,3)
        directions: (V,3) normalized
    """

    if isinstance(T, torch.Tensor):
        T = T.detach()

    if T.shape[-2:] == (3, 4):
        R = T[..., :3, :3]
        t = T[..., :3, 3]
    elif T.shape[-2:] == (4, 4):
        R = T[..., :3, :3]
        t = T[..., :3, 3]
    else:
        raise ValueError("Invalid extrinsic shape")

    # Detect cam2world vs world2cam
    if torch.norm(t.mean(dim=0)) > 10:
        # cam2world
        C = t
        R_world = R
    else:
        # world2cam
        C = -torch.matmul(R.transpose(-1, -2), t.unsqueeze(-1)).squeeze(-1)
        R_world = R.transpose(-1, -2)

    # Forward direction (camera z-axis in world)
    # z_cam = torch.tensor([0., 0., 1.], device=R.device)
    # d = torch.matmul(R_world, z_cam)
    d = R_world[..., :, 0]

    # Normalize
    d = d / (torch.norm(d, dim=-1, keepdim=True) + 1e-8)

    return C, d


def sim3_align(pred, gt):
    """
    Align pred trajectory to gt trajectory using Sim3
    pred, gt: (V,3) numpy arrays
    Returns:
        pred_aligned (V,3)
        R (3,3) rotation used for alignment
    """

    pred_mean = pred.mean(axis=0)
    gt_mean = gt.mean(axis=0)

    pred_centered = pred - pred_mean
    gt_centered = gt - gt_mean

    # Scale
    scale = np.linalg.norm(gt_centered) / (np.linalg.norm(pred_centered) + 1e-8)
    pred_centered *= scale

    # Rotation (Umeyama)
    H = pred_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    pred_aligned = (R @ pred_centered.T).T + gt_mean

    return pred_aligned, R

