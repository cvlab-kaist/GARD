import os 
import cv2 
import torch 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def tensor_to_uint8_image(img):  # img: (3, H, W), torch tensor
    img = img.permute(1, 2, 0).cpu().numpy()   # (H, W, 3)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def load_depth(depth_path, H, W):
    depth = np.fromfile(depth_path, dtype=np.float32)
    assert depth.size == H * W, f"Size mismatch: {depth_path}"
    depth = depth.reshape(H, W)
    depth[np.isinf(depth)] = np.nan
    return depth


@torch.no_grad()
def align_scale_median(gt, pred, eps=1e-6):
    valid = torch.isfinite(gt) & torch.isfinite(pred) & (gt > eps) & (pred > eps)
    if valid.sum() == 0:
        return pred
    scale = torch.median(gt[valid]) / (torch.median(pred[valid]) + eps)
    return pred * scale


@torch.no_grad()
def compute_depth_metrics(gt, pred, eps=1e-6):
    valid = torch.isfinite(gt) & torch.isfinite(pred) & (gt > eps) & (pred > eps)
    if valid.sum() == 0:
        return [float("nan")] * 7

    gt = torch.clamp(gt[valid], min=eps)
    pred = torch.clamp(pred[valid], min=eps)

    thresh = torch.maximum(gt / pred, pred / gt)

    d1 = (thresh < 1.25).float().mean()
    d2 = (thresh < 1.25 ** 2).float().mean()
    d3 = (thresh < 1.25 ** 3).float().mean()

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean((gt - pred) ** 2 / gt)
    rmse = torch.sqrt(torch.mean((gt - pred) ** 2))
    rmse_log = torch.sqrt(torch.mean((torch.log(gt) - torch.log(pred)) ** 2))

    return [
        abs_rel.item(),
        sq_rel.item(),
        rmse.item(),
        rmse_log.item(),
        d1.item(),
        d2.item(),
        d3.item(),
    ]


def depth_to_colormap(depth, invalid_color=(0, 0, 0), resize=None):
    valid = np.isfinite(depth) & (depth > 0)
    if valid.sum() == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    vis = depth.copy()
    vmin, vmax = np.percentile(vis[valid], [2, 98])
    vis = np.clip(vis, vmin, vmax)
    vis = (vis - vmin) / (vmax - vmin + 1e-8)
    vis = (vis * 255).astype(np.uint8)

    # color = cv2.applyColorMap(vis, cv2.COLORMAP_PLASMA)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
    
    color[~valid] = invalid_color
    
    if resize is not None:
        h, w = resize
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return color


def depth_error_to_colormap_thresholded(
    gt,
    pred,
    thr=0.5,                 # AbsRel threshold
    invalid_color=(0, 0, 0),
    resize=None,
):
    """
    Visualize ONLY high-error regions using TURBO.
    Low-error pixels are blacked out.

    gt, pred: (H, W) numpy arrays
    thr: AbsRel threshold
    """

    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > 0)
    if valid.sum() == 0:
        h, w = gt.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        if resize is not None:
            vis = cv2.resize(vis, (resize[1], resize[0]), interpolation=cv2.INTER_NEAREST)
        return vis

    # AbsRel
    absrel = np.zeros_like(gt, dtype=np.float32)
    absrel[valid] = np.abs(pred[valid] - gt[valid]) / gt[valid]

    # High-error mask
    high = valid & (absrel >= thr)
    if high.sum() == 0:
        return np.zeros((*gt.shape, 3), dtype=np.uint8)

    # Optional log compression (recommended)
    err = np.zeros_like(gt, dtype=np.float32)
    err[high] = np.log10(absrel[high] + 1e-4)

    # Normalize only on high-error pixels
    vmin, vmax = np.percentile(err[high], [5, 95])
    err = np.clip(err, vmin, vmax)
    err = (err - vmin) / (vmax - vmin + 1e-8)
    err = (err * 255).astype(np.uint8)

    vis = cv2.applyColorMap(err, cv2.COLORMAP_TURBO)

    # Mask everything else
    vis[~high] = invalid_color

    if resize is not None:
        h, w = resize
        vis = cv2.resize(vis, (w, h), interpolation=cv2.INTER_LINEAR)

    return vis


# ============================================================
# Formatting helpers
# ============================================================
def fmt(val, width=8, prec=4):
    return f"{val:{width}.{prec}f}"

def fmt_int(val, width=12):
    return f"{val:{width},}"

def write_scene_header(f, scene):
    f.write("\n" + "=" * 90 + "\n")
    f.write(f"Scene: {scene}\n")
    f.write("-" * 90 + "\n")
    f.write(
        f"{'Image':<12}"
        f"{'AbsRel':>8}"
        f"{'SqRel':>8}"
        f"{'RMSE':>9}"
        f"{'RMSElog':>10}"
        f"{'δ1':>8}"
        f"{'δ2':>8}"
        f"{'δ3':>8}"
        f"{'Pixels':>12}\n"
    )
    f.write("-" * 90 + "\n")


def vis_all(vis_save_root, scene, hq_img, lq_img, hq_depth, lq_depth, res_depth):
    """
    hq_img:   (V, 3, H, W), or None for GT-free scenes (HQ panels are omitted)
    lq_img:   (V, 3, H, W)
    hq_depth: (V, 1, H, W), or None for GT-free scenes (HQ panels are omitted)
    lq_depth: (V, 1, H, W)
    res_depth:(V, 1, H, W)

    """

    os.makedirs(vis_save_root, exist_ok=True)

    has_hq = hq_img is not None and hq_depth is not None
    V = lq_img.shape[0]
    all_rows = []

    for i in range(V):

        # -------------------------
        # RGB images
        # -------------------------
        lq_rgb = tensor_to_uint8_image(lq_img[i])

        H, W, _ = lq_rgb.shape

        # -------------------------
        # Depth maps → numpy
        # -------------------------
        lq_d = lq_depth[i, 0].detach().cpu().numpy()
        res_d = res_depth[i, 0].detach().cpu().numpy()

        # -------------------------
        # Depth → colormap
        # -------------------------
        lq_d_vis = depth_to_colormap(lq_d, resize=(H, W))
        res_d_vis = depth_to_colormap(res_d, resize=(H, W))

        # -------------------------
        # One row: 5 panels (3 when there's no HQ reference to show)
        # -------------------------
        panels = [lq_rgb[:, :, ::-1], lq_d_vis, res_d_vis]
        if has_hq:
            hq_rgb = tensor_to_uint8_image(hq_img[i])
            hq_d = hq_depth[i, 0].detach().cpu().numpy()
            hq_d_vis = depth_to_colormap(hq_d, resize=(H, W))
            panels = [hq_rgb[:, :, ::-1], hq_d_vis] + panels
        row = np.concatenate(panels, axis=1)

        all_rows.append(row)

    # -------------------------
    # Stack all rows vertically
    # -------------------------
    final_vis = np.concatenate(all_rows, axis=0)

    save_path = os.path.join(vis_save_root, f"{scene}.png")
    cv2.imwrite(save_path, final_vis)

    print(f"Saved visualization to {save_path}")
    




def vis_attn_maps(vis_save_root, num_view, model_img_size, scene, imgs_list, attn_maps_list, ref_b_idx, mvrm_timestep=None, mvrm_step=None):

    model_H, model_W = model_img_size
    patch_size = 14
    ph = model_H // patch_size   # e.g. 504 // 14 = 36
    pw = model_W // patch_size   # e.g. 378 // 14 = 27
    
    
    to_vis = ['cls_tkn', 'patch_tkn']

    # ── normalization modes to save: choose any subset of ['global_log1p', 'perv_p99'] ──
    norm_modes = ['global_log1p', 'perv_p99']
    


    # build view reorder indices to match what the model saw (ref view placed first)
    if ref_b_idx is not None:
        b = int(ref_b_idx[0].item())
        view_order = [b] + list(range(0, b)) + list(range(b + 1, num_view))
    else:
        view_order = list(range(num_view))

    # build imgs lookup: {name: (v, H, W, 3) uint8 numpy}
    imgs_dict = {}
    for imgs_name, imgs_tensor in imgs_list:
        # imgs_tensor: (1, v, c, H, W), imagenet-normalised
        imgs_np = imgs_tensor[0].permute(0, 2, 3, 1).cpu().numpy()   # (v, H, W, 3)
        imgs_np = imgs_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        imgs_np = np.clip(imgs_np * 255, 0, 255).astype(np.uint8)
        imgs_np = imgs_np[view_order]   # reorder views to match model's reference-first ordering
        imgs_dict[imgs_name] = imgs_np   # BGR conversion handled by cv2 below


    for attn_map_name, attn_map in attn_maps_list:


        # pick corresponding images (lq_maps -> lq_imgs, hq_maps -> hq_imgs, res_maps -> res_imgs)
        img_key = attn_map_name.replace('_maps', '_imgs')
        imgs_np = imgs_dict.get(img_key, list(imgs_dict.values())[0])  # fallback to first
        v0_bgr = cv2.cvtColor(imgs_np[0], cv2.COLOR_RGB2BGR)

        # ── cls_tkn: one image per layer + one combined image (layers as rows) ─
        if 'cls_tkn' in to_vis:
            
            if mvrm_step is not None:
                cls_attn_save_root = os.path.join(vis_save_root, f'iter{mvrm_step}', 'cls_tkn', attn_map_name, scene)
            elif mvrm_timestep is not None:
                cls_attn_save_root = os.path.join(vis_save_root, f'timestep{mvrm_timestep}', 'cls_tkn', attn_map_name, scene)
            else:
                cls_attn_save_root = os.path.join(vis_save_root, 'cls_tkn', attn_map_name, scene)
            os.makedirs(cls_attn_save_root, exist_ok=True)

            cls_layer_strips = {m: [] for m in norm_modes}  # {mode: [(layer_idx, strip)]}
            cls_bar_figs = []                                # (layer_idx, fig) for combined bar graph

            for map_info, map in attn_map.items():
                model_name, layer_idx, attn_type = map_info
                if attn_type != 'global':
                    continue

                tokens_per_view = map.shape[1] // num_view
                assert tokens_per_view == ph * pw + 1, \
                    f"tokens_per_view {tokens_per_view} != ph*pw+1 {ph*pw+1}"

                cls_row = map[0, 0, :]   # (V * tokens_per_view,)

                # ── bar graph of cls_row magnitudes ──────────────────────────
                cls_row_np = cls_row.cpu().float().numpy()
                fig, ax = plt.subplots(figsize=(max(8, len(cls_row_np) // 50), 4))
                ax.bar(np.arange(len(cls_row_np)), cls_row_np, width=1.0, linewidth=0)
                ax.set_xlabel('token index')
                ax.set_ylabel('attention weight')
                ax.set_title(f'{attn_map_name}  layer{layer_idx}  cls→all tokens')
                ax.set_xlim(0, len(cls_row_np))
                ax.set_yscale('log')
                for vi in range(1, num_view):
                    ax.axvline(vi * tokens_per_view, color='red', linewidth=0.8, linestyle='--')
                fig.tight_layout()
                cls_bar_figs.append((layer_idx, fig))
                # ─────────────────────────────────────────────────────────────
                
                raw_patch_attns = []
                for vi in range(num_view):
                    start = vi * tokens_per_view
                    pa = cls_row[start + 1 : start + tokens_per_view].cpu().numpy()
                    raw_patch_attns.append(pa)

                def _make_cls_strip(patch_attns_list):
                    label_img = v0_bgr.copy()
                    cv2.putText(label_img, f'layer{layer_idx}', (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cols = [label_img]
                    for vi in range(num_view):
                        patch_attn = patch_attns_list[vi].reshape(ph, pw)
                        heatmap = cv2.resize((patch_attn * 255).astype(np.uint8), (model_W, model_H), interpolation=cv2.INTER_LINEAR)
                        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        img_bgr = cv2.cvtColor(imgs_np[vi], cv2.COLOR_RGB2BGR)
                        overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)
                        cv2.putText(overlay, f'view{vi+1}', (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cols.append(overlay)
                    return np.concatenate(cols, axis=1)

                if 'global_log1p' in norm_modes:
                    all_log = [np.log1p(pa) for pa in raw_patch_attns]
                    g_min = min(p.min() for p in all_log)
                    g_max = max(p.max() for p in all_log)
                    all_patch_attns_global = [(pa - g_min) / (g_max - g_min + 1e-8) for pa in all_log]
                    cls_layer_strips['global_log1p'].append((layer_idx, _make_cls_strip(all_patch_attns_global)))

                if 'perv_p99' in norm_modes:
                    all_patch_attns_perv = []
                    for pa in raw_patch_attns:
                        pa = np.clip(pa, 0, np.percentile(pa, 99))
                        pa = (pa - pa.min()) / (pa.max() - pa.min() + 1e-8)
                        all_patch_attns_perv.append(pa)
                    cls_layer_strips['perv_p99'].append((layer_idx, _make_cls_strip(all_patch_attns_perv)))

                # # per-layer image
                # save_path = os.path.join(cls_attn_save_root, f'layer{layer_idx}__{attn_type}__v0cls.png')
                # cv2.imwrite(save_path, strip)
                # print(f"[CLS attn map] Saved: {save_path}")

            # combined heatmap: one file per norm mode, each in its own subfolder
            for mode, strips in cls_layer_strips.items():
                if strips:
                    mode_dir = os.path.join(cls_attn_save_root, mode)
                    os.makedirs(mode_dir, exist_ok=True)
                    combined = np.concatenate([s for _, s in sorted(strips)], axis=0)
                    save_path = os.path.join(mode_dir, f'all_layers__{attn_type}__v0cls.png')
                    cv2.imwrite(save_path, combined)
                    print(f"[CLS attn map {mode}] Saved: {save_path}")

            # combined bar graph: all layers stacked vertically into one image
            if cls_bar_figs:
                from io import BytesIO
                from PIL import Image
                bar_imgs = []
                for _, fig in sorted(cls_bar_figs):
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=100)
                    plt.close(fig)
                    buf.seek(0)
                    bar_imgs.append(np.array(Image.open(buf)))
                combined_bar = np.concatenate(bar_imgs, axis=0)
                bar_combined_path = os.path.join(cls_attn_save_root, f'all_layers__{attn_type}__v0cls_bar.png')
                combined_bar_bgr = cv2.cvtColor(combined_bar, cv2.COLOR_RGBA2BGR) if combined_bar.shape[2] == 4 else cv2.cvtColor(combined_bar, cv2.COLOR_RGB2BGR)
                cv2.imwrite(bar_combined_path, combined_bar_bgr)
                print(f"[CLS bar graph combined] Saved: {bar_combined_path}")

        # ── patch_tkn: one image per query patch, layers as rows ─────────────
        if 'patch_tkn' in to_vis:

            # determine tokens_per_view and sampled patches from the first global map
            first_global_map = next(m for (_, _, t), m in attn_map.items() if t == 'global')
            tokens_per_view = first_global_map.shape[1] // num_view
            n_patches = tokens_per_view - 1
            
            # sample a 3x3 grid evenly spread across the image (at 1/4, 1/2, 3/4 positions)
            grid_rows, grid_cols = 3, 3
            row_positions = [int(round(ph * f)) for f in [0.25, 0.50, 0.75]]
            col_positions = [int(round(pw * f)) for f in [0.25, 0.50, 0.75]]
            sampled_patch_indices = []
            for r in row_positions:
                for c in col_positions:
                    r = np.clip(r, 0, ph - 1)
                    c = np.clip(c, 0, pw - 1)
                    sampled_patch_indices.append(r * pw + c)
            sampled_patch_indices = np.array(sorted(set(sampled_patch_indices)))

            if mvrm_step is not None:
                patch_attn_save_root = os.path.join(vis_save_root, f'iter{mvrm_step}', 'patch_tkn', attn_map_name, scene)
            elif mvrm_timestep is not None:
                patch_attn_save_root = os.path.join(vis_save_root, f'timestep{mvrm_timestep}', 'patch_tkn', attn_map_name, scene)
            else:
                patch_attn_save_root = os.path.join(vis_save_root, 'patch_tkn', attn_map_name, scene)
            os.makedirs(patch_attn_save_root, exist_ok=True)

            # collect strips: {mode: {pi: [(layer_idx, strip)]}}
            strips_per_patch = {m: {pi: [] for pi in sampled_patch_indices} for m in norm_modes}

            for map_info, map in attn_map.items():
                model_name, layer_idx, attn_type = map_info
                if attn_type != 'global':
                    continue

                for pi in sampled_patch_indices:
                    query_row = map[0, 1 + pi, :]   # (V * tokens_per_view,)

                    # self + cross attention for all views (vi=0 is self-attention)
                    all_view_attns = []
                    for vi in range(num_view):
                        start = vi * tokens_per_view
                        pa = query_row[start + 1 : start + tokens_per_view].cpu().numpy()
                        if vi == 0:
                            pa = pa.copy()
                            pa[pi] = 0.0   # mask diagonal (query patch attending to itself)
                        all_view_attns.append((vi, pa))
                    # query patch pixel center
                    pi_h, pi_w = pi // pw, pi % pw
                    cx = int((pi_w + 0.5) * patch_size)
                    cy = int((pi_h + 0.5) * patch_size)

                    def _make_patch_strip(view_attns_list):
                        ref_img = v0_bgr.copy()
                        cv2.circle(ref_img, (cx, cy), radius=8, color=(0, 255, 0), thickness=-1)
                        cv2.circle(ref_img, (cx, cy), radius=8, color=(0, 0, 0), thickness=1)
                        cv2.putText(ref_img, f'layer{layer_idx}', (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cols = [ref_img]
                        for vi, pa in view_attns_list:
                            patch_attn = pa.reshape(ph, pw)
                            heatmap = cv2.resize((patch_attn * 255).astype(np.uint8), (model_W, model_H), interpolation=cv2.INTER_LINEAR)
                            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                            img_bgr = cv2.cvtColor(imgs_np[vi], cv2.COLOR_RGB2BGR)
                            overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)
                            label = f'view{vi+1}(self)' if vi == 0 else f'view{vi+1}'
                            cv2.putText(overlay, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cols.append(overlay)
                        return np.concatenate(cols, axis=1)

                    if 'global_log1p' in norm_modes:
                        all_log = [(vi, np.log1p(pa)) for vi, pa in all_view_attns]
                        g_min = min(pa.min() for _, pa in all_log)
                        g_max = max(pa.max() for _, pa in all_log)
                        all_view_attns_global = [(vi, (pa - g_min) / (g_max - g_min + 1e-8)) for vi, pa in all_log]
                        strips_per_patch['global_log1p'][pi].append((layer_idx, _make_patch_strip(all_view_attns_global)))

                    if 'perv_p99' in norm_modes:
                        all_view_attns_perv = []
                        for vi, pa in all_view_attns:
                            pa = np.clip(pa, 0, np.percentile(pa, 99))
                            pa = (pa - pa.min()) / (pa.max() - pa.min() + 1e-8)
                            all_view_attns_perv.append((vi, pa))
                        strips_per_patch['perv_p99'][pi].append((layer_idx, _make_patch_strip(all_view_attns_perv)))

            # save one image per query patch per norm mode, each mode in its own subfolder
            for mode, patch_strips in strips_per_patch.items():
                mode_dir = os.path.join(patch_attn_save_root, mode)
                os.makedirs(mode_dir, exist_ok=True)
                for pi in sampled_patch_indices:
                    grid_img = np.concatenate([s for _, s in sorted(patch_strips[pi])], axis=0)
                    save_path = os.path.join(mode_dir, f'v0patch{pi:04d}.png')
                    cv2.imwrite(save_path, grid_img)
                    print(f"[Patch attn map {mode}] Saved: {save_path}")
                

        
        
def vis_pointcloud_correspondence_maps(vis_save_root, scene, num_view, model_img_size, imgs_list, attn_maps_list, ref_b_idx):
    """
    Visualize patch-level geometric correspondence maps derived from 3D point clouds.
    Mirrors vis_attn_maps patch_tkn visualization but handles no-CLS-token layout:
      geo_corr shape: (b, v*n, v*n)  — purely spatial, no CLS tokens.
    """
    model_H, model_W = model_img_size
    patch_size = 14
    ph = model_H // patch_size
    pw = model_W // patch_size
    n  = ph * pw   # spatial tokens per view (no CLS)

    norm_modes = ['global_log1p', 'perv_p99']

    # ── view reorder (ref view first) ──
    if ref_b_idx is not None:
        b = int(ref_b_idx[0].item())
        view_order = [b] + list(range(0, b)) + list(range(b + 1, num_view))
    else:
        view_order = list(range(num_view))

    # ── build imgs lookup ──
    imgs_dict = {}
    for imgs_name, imgs_tensor in imgs_list:
        imgs_np = imgs_tensor[0].permute(0, 2, 3, 1).cpu().numpy()   # (v, H, W, 3)
        imgs_np = imgs_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        imgs_np = np.clip(imgs_np * 255, 0, 255).astype(np.uint8)
        imgs_np = imgs_np[view_order]
        imgs_dict[imgs_name] = imgs_np

    # ── sampled 3x3 grid of query patches ──
    row_positions = [int(round(ph * f)) for f in [0.25, 0.50, 0.75]]
    col_positions = [int(round(pw * f)) for f in [0.25, 0.50, 0.75]]
    sampled_patch_indices = []
    for r in row_positions:
        for c in col_positions:
            r = np.clip(r, 0, ph - 1)
            c = np.clip(c, 0, pw - 1)
            sampled_patch_indices.append(r * pw + c)
    sampled_patch_indices = np.array(sorted(set(sampled_patch_indices)))

    for pc_map_name, pc_map in attn_maps_list:

        # pick corresponding images: 'lq_pc_maps' -> 'lq_imgs'
        img_key = pc_map_name.replace('_pc_maps', '_imgs')
        imgs_np = imgs_dict.get(img_key, list(imgs_dict.values())[0])
        v0_bgr  = cv2.cvtColor(imgs_np[0], cv2.COLOR_RGB2BGR)

        # pc_map: {('pc', 0, 'global'): geo_corr (b, v*n, v*n)}
        for map_info, geo_corr in pc_map.items():
            if map_info[2] != 'global':
                continue

            geo_corr_np = geo_corr[0].cpu().float().numpy()  # (v*n, v*n)

            # reorder view blocks to match view_order (ref view first)
            perm = np.concatenate([np.arange(vi * n, vi * n + n) for vi in view_order])
            geo_corr_np = geo_corr_np[np.ix_(perm, perm)]

            patch_save_root = os.path.join(vis_save_root, 'patch_tkn', pc_map_name, scene)
            os.makedirs(patch_save_root, exist_ok=True)

            strips_per_patch = {m: {} for m in norm_modes}

            for pi in sampled_patch_indices:
                # query patch pi is in ref view (view 0 after reorder) — no CLS offset
                query_row = geo_corr_np[pi, :]  # (v*n,)

                all_view_attns = []
                for vi in range(num_view):
                    pa = query_row[vi * n : vi * n + n].copy()
                    if vi == 0:
                        pa[pi] = 0.0   # mask self-correspondence on diagonal
                    all_view_attns.append((vi, pa))

                pi_h, pi_w = pi // pw, pi % pw
                cx = int((pi_w + 0.5) * patch_size)
                cy = int((pi_h + 0.5) * patch_size)

                def _make_strip(view_attns_list):
                    ref_img = v0_bgr.copy()
                    cv2.circle(ref_img, (cx, cy), radius=8, color=(0, 255, 0), thickness=-1)
                    cv2.circle(ref_img, (cx, cy), radius=8, color=(0, 0, 0), thickness=1)
                    cols = [ref_img]
                    for vi, pa in view_attns_list:
                        patch_attn    = pa.reshape(ph, pw)
                        heatmap       = cv2.resize((patch_attn * 255).astype(np.uint8), (model_W, model_H), interpolation=cv2.INTER_LINEAR)
                        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        img_bgr       = cv2.cvtColor(imgs_np[vi], cv2.COLOR_RGB2BGR)
                        overlay       = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)
                        label         = f'view{vi+1}(self)' if vi == 0 else f'view{vi+1}'
                        cv2.putText(overlay, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cols.append(overlay)
                    return np.concatenate(cols, axis=1)

                if 'global_log1p' in norm_modes:
                    all_log    = [(vi, np.log1p(pa)) for vi, pa in all_view_attns]
                    g_min      = min(pa.min() for _, pa in all_log)
                    g_max      = max(pa.max() for _, pa in all_log)
                    normalized = [(vi, (pa - g_min) / (g_max - g_min + 1e-8)) for vi, pa in all_log]
                    strips_per_patch['global_log1p'][pi] = _make_strip(normalized)

                if 'perv_p99' in norm_modes:
                    perv = []
                    for vi, pa in all_view_attns:
                        p99 = np.percentile(pa, 99)
                        if p99 > 0:  # skip clip when distribution is too sparse (e.g. hard one-hot)
                            pa = np.clip(pa, 0, p99)
                        pa = (pa - pa.min()) / (pa.max() - pa.min() + 1e-8)
                        perv.append((vi, pa))
                    strips_per_patch['perv_p99'][pi] = _make_strip(perv)

            for mode, patch_strips in strips_per_patch.items():
                mode_dir = os.path.join(patch_save_root, mode)
                os.makedirs(mode_dir, exist_ok=True)
                for pi in sampled_patch_indices:
                    strip = patch_strips.get(pi)
                    if strip is not None:
                        save_path = os.path.join(mode_dir, f'v0patch{pi:04d}.png')
                        cv2.imwrite(save_path, strip)
                        print(f"[PC corr {mode}] Saved: {save_path}")


def vis_pointcloud_cycle_correspondence_maps(vis_save_root, scene, num_view, model_img_size, imgs_list, attn_maps_list, ref_b_idx):
    """
    Same as vis_pointcloud_correspondence_maps but for cycle-consistency-masked geo_corr.
    Expects pc_map keys with map_info[2] == 'global_cycle'.
    Zero values indicate patches that failed cycle consistency (not visible in that view);
    they appear as blue in JET, distinguishing them from low-but-visible patches.
    """
    model_H, model_W = model_img_size
    patch_size = 14
    ph = model_H // patch_size
    pw = model_W // patch_size
    n  = ph * pw

    norm_modes = ['global_log1p', 'perv_p99']

    if ref_b_idx is not None:
        b = int(ref_b_idx[0].item())
        view_order = [b] + list(range(0, b)) + list(range(b + 1, num_view))
    else:
        view_order = list(range(num_view))

    imgs_dict = {}
    for imgs_name, imgs_tensor in imgs_list:
        imgs_np = imgs_tensor[0].permute(0, 2, 3, 1).cpu().numpy()
        imgs_np = imgs_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        imgs_np = np.clip(imgs_np * 255, 0, 255).astype(np.uint8)
        imgs_np = imgs_np[view_order]
        imgs_dict[imgs_name] = imgs_np

    row_positions = [int(round(ph * f)) for f in [0.25, 0.50, 0.75]]
    col_positions = [int(round(pw * f)) for f in [0.25, 0.50, 0.75]]
    sampled_patch_indices = []
    for r in row_positions:
        for c in col_positions:
            r = np.clip(r, 0, ph - 1)
            c = np.clip(c, 0, pw - 1)
            sampled_patch_indices.append(r * pw + c)
    sampled_patch_indices = np.array(sorted(set(sampled_patch_indices)))

    for pc_map_name, pc_map in attn_maps_list:

        img_key = pc_map_name.replace('_pc_maps_cycle', '_imgs')
        imgs_np = imgs_dict.get(img_key, list(imgs_dict.values())[0])
        v0_bgr  = cv2.cvtColor(imgs_np[0], cv2.COLOR_RGB2BGR)

        for map_info, geo_corr in pc_map.items():
            if map_info[2] != 'global_cycle':
                continue

            geo_corr_np = geo_corr[0].cpu().float().numpy()  # (v*n, v*n)

            perm = np.concatenate([np.arange(vi * n, vi * n + n) for vi in view_order])
            geo_corr_np = geo_corr_np[np.ix_(perm, perm)]

            patch_save_root = os.path.join(vis_save_root, 'patch_tkn', pc_map_name, scene)
            os.makedirs(patch_save_root, exist_ok=True)

            strips_per_patch = {m: {} for m in norm_modes}

            for pi in sampled_patch_indices:
                query_row = geo_corr_np[pi, :]  # (v*n,)

                all_view_attns = []
                for vi in range(num_view):
                    pa = query_row[vi * n : vi * n + n].copy()
                    if vi == 0:
                        pa[pi] = 0.0
                    all_view_attns.append((vi, pa))

                pi_h, pi_w = pi // pw, pi % pw
                cx = int((pi_w + 0.5) * patch_size)
                cy = int((pi_h + 0.5) * patch_size)

                def _make_strip(view_attns_list):
                    ref_img = v0_bgr.copy()
                    cv2.circle(ref_img, (cx, cy), radius=8, color=(0, 255, 0), thickness=-1)
                    cv2.circle(ref_img, (cx, cy), radius=8, color=(0, 0, 0), thickness=1)
                    cols = [ref_img]
                    for vi, pa in view_attns_list:
                        patch_attn    = pa.reshape(ph, pw)
                        heatmap       = cv2.resize((patch_attn * 255).astype(np.uint8), (model_W, model_H), interpolation=cv2.INTER_LINEAR)
                        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        img_bgr       = cv2.cvtColor(imgs_np[vi], cv2.COLOR_RGB2BGR)
                        overlay       = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)
                        label         = f'view{vi+1}(self)' if vi == 0 else f'view{vi+1}'
                        cv2.putText(overlay, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cols.append(overlay)
                    return np.concatenate(cols, axis=1)

                if 'global_log1p' in norm_modes:
                    all_log = [(vi, np.log1p(pa)) for vi, pa in all_view_attns]
                    # compute range over visible (non-zero) values only so masked zeros stay at 0 (blue)
                    all_nonzero = np.concatenate([pa[pa > 0] for _, pa in all_log], axis=0) \
                                  if any((pa > 0).any() for _, pa in all_log) else np.array([0.0, 1.0])
                    g_min = float(all_nonzero.min())
                    g_max = float(all_nonzero.max())
                    normalized = [(vi, np.clip((pa - g_min) / (g_max - g_min + 1e-8), 0, 1))
                                  for vi, pa in all_log]
                    strips_per_patch['global_log1p'][pi] = _make_strip(normalized)

                if 'perv_p99' in norm_modes:
                    perv = []
                    for vi, pa in all_view_attns:
                        visible_vals = pa[pa > 0]
                        p99 = float(np.percentile(visible_vals, 99)) if len(visible_vals) > 0 else 0.0
                        pa  = np.clip(pa, 0, p99)
                        pa  = (pa - pa.min()) / (pa.max() - pa.min() + 1e-8)
                        perv.append((vi, pa))
                    strips_per_patch['perv_p99'][pi] = _make_strip(perv)

            for mode, patch_strips in strips_per_patch.items():
                mode_dir = os.path.join(patch_save_root, mode)
                os.makedirs(mode_dir, exist_ok=True)
                for pi in sampled_patch_indices:
                    strip = patch_strips.get(pi)
                    if strip is not None:
                        save_path = os.path.join(mode_dir, f'v0patch{pi:04d}.png')
                        cv2.imwrite(save_path, strip)
                        print(f"[PC cycle corr {mode}] Saved: {save_path}")


def vis_pointcloud_reproj_correspondence_maps(vis_save_root, scene, num_view, model_img_size, imgs_list, attn_maps_list, ref_b_idx):
    """
    Same as vis_pointcloud_cycle_correspondence_maps but for reprojection-masked geo_corr.
    Expects pc_map keys with map_info[2] == 'global_reproj'.
    Zero values indicate patches that failed reprojection visibility (out of bounds or
    depth-inconsistent); they appear as blue in JET.
    """
    model_H, model_W = model_img_size
    patch_size = 14
    ph = model_H // patch_size
    pw = model_W // patch_size
    n  = ph * pw

    norm_modes = ['global_log1p', 'perv_p99']

    if ref_b_idx is not None:
        b = int(ref_b_idx[0].item())
        view_order = [b] + list(range(0, b)) + list(range(b + 1, num_view))
    else:
        view_order = list(range(num_view))

    imgs_dict = {}
    for imgs_name, imgs_tensor in imgs_list:
        imgs_np = imgs_tensor[0].permute(0, 2, 3, 1).cpu().numpy()
        imgs_np = imgs_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        imgs_np = np.clip(imgs_np * 255, 0, 255).astype(np.uint8)
        imgs_np = imgs_np[view_order]
        imgs_dict[imgs_name] = imgs_np

    row_positions = [int(round(ph * f)) for f in [0.25, 0.50, 0.75]]
    col_positions = [int(round(pw * f)) for f in [0.25, 0.50, 0.75]]
    sampled_patch_indices = []
    for r in row_positions:
        for c in col_positions:
            r = np.clip(r, 0, ph - 1)
            c = np.clip(c, 0, pw - 1)
            sampled_patch_indices.append(r * pw + c)
    sampled_patch_indices = np.array(sorted(set(sampled_patch_indices)))

    for pc_map_name, pc_map in attn_maps_list:

        img_key = pc_map_name.replace('_pc_maps_reproj', '_imgs')
        imgs_np = imgs_dict.get(img_key, list(imgs_dict.values())[0])
        v0_bgr  = cv2.cvtColor(imgs_np[0], cv2.COLOR_RGB2BGR)

        for map_info, geo_corr in pc_map.items():
            if map_info[2] != 'global_reproj':
                continue

            geo_corr_np = geo_corr[0].cpu().float().numpy()  # (v*n, v*n)

            perm = np.concatenate([np.arange(vi * n, vi * n + n) for vi in view_order])
            geo_corr_np = geo_corr_np[np.ix_(perm, perm)]

            patch_save_root = os.path.join(vis_save_root, 'patch_tkn', pc_map_name, scene)
            os.makedirs(patch_save_root, exist_ok=True)

            strips_per_patch = {m: {} for m in norm_modes}

            for pi in sampled_patch_indices:
                query_row = geo_corr_np[pi, :]

                all_view_attns = []
                for vi in range(num_view):
                    pa = query_row[vi * n : vi * n + n].copy()
                    if vi == 0:
                        pa[pi] = 0.0
                    all_view_attns.append((vi, pa))

                pi_h, pi_w = pi // pw, pi % pw
                cx = int((pi_w + 0.5) * patch_size)
                cy = int((pi_h + 0.5) * patch_size)

                def _make_strip(view_attns_list):
                    ref_img = v0_bgr.copy()
                    cv2.circle(ref_img, (cx, cy), radius=8, color=(0, 255, 0), thickness=-1)
                    cv2.circle(ref_img, (cx, cy), radius=8, color=(0, 0, 0), thickness=1)
                    cols = [ref_img]
                    for vi, pa in view_attns_list:
                        patch_attn    = pa.reshape(ph, pw)
                        heatmap       = cv2.resize((patch_attn * 255).astype(np.uint8), (model_W, model_H), interpolation=cv2.INTER_LINEAR)
                        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        img_bgr       = cv2.cvtColor(imgs_np[vi], cv2.COLOR_RGB2BGR)
                        overlay       = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)
                        label         = f'view{vi+1}(self)' if vi == 0 else f'view{vi+1}'
                        cv2.putText(overlay, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cols.append(overlay)
                    return np.concatenate(cols, axis=1)

                if 'global_log1p' in norm_modes:
                    all_log = [(vi, np.log1p(pa)) for vi, pa in all_view_attns]
                    all_nonzero = np.concatenate([pa[pa > 0] for _, pa in all_log], axis=0) \
                                  if any((pa > 0).any() for _, pa in all_log) else np.array([0.0, 1.0])
                    g_min = float(all_nonzero.min())
                    g_max = float(all_nonzero.max())
                    normalized = [(vi, np.clip((pa - g_min) / (g_max - g_min + 1e-8), 0, 1))
                                  for vi, pa in all_log]
                    strips_per_patch['global_log1p'][pi] = _make_strip(normalized)

                if 'perv_p99' in norm_modes:
                    perv = []
                    for vi, pa in all_view_attns:
                        visible_vals = pa[pa > 0]
                        p99 = float(np.percentile(visible_vals, 99)) if len(visible_vals) > 0 else 0.0
                        pa  = np.clip(pa, 0, p99)
                        pa  = (pa - pa.min()) / (pa.max() - pa.min() + 1e-8)
                        perv.append((vi, pa))
                    strips_per_patch['perv_p99'][pi] = _make_strip(perv)

            for mode, patch_strips in strips_per_patch.items():
                mode_dir = os.path.join(patch_save_root, mode)
                os.makedirs(mode_dir, exist_ok=True)
                for pi in sampled_patch_indices:
                    strip = patch_strips.get(pi)
                    if strip is not None:
                        save_path = os.path.join(mode_dir, f'v0patch{pi:04d}.png')
                        cv2.imwrite(save_path, strip)
                        print(f"[PC reproj corr {mode}] Saved: {save_path}")

