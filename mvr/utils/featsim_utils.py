import os 
import torch 
import torch.nn.functional as F 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator


def _to_tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.float()


@torch.no_grad()
def cosine_sim_mean(a, b, dim=-1, eps=1e-8):
    a = _to_tensor(a)
    b = _to_tensor(b)
    if a.device != b.device:
        b = b.to(a.device)
    a = F.normalize(a, dim=dim, eps=eps)
    b = F.normalize(b, dim=dim, eps=eps)
    sim = (a * b).sum(dim=dim)
    return sim.mean().item()


@torch.no_grad()
def cosine_sim_stats(a, b, dim=-1, eps=1e-8, quantiles=(0.1, 0.5, 0.9)):
    a = _to_tensor(a)
    b = _to_tensor(b)
    if a.device != b.device:
        b = b.to(a.device)
    a = F.normalize(a, dim=dim, eps=eps)
    b = F.normalize(b, dim=dim, eps=eps)
    sim = (a * b).sum(dim=dim).flatten()
    out = {
        "mean": sim.mean().item(),
        "std": sim.std(unbiased=False).item(),
        "min": sim.min().item(),
        "max": sim.max().item(),
    }
    if quantiles is not None and sim.numel() > 0:
        qs = torch.tensor(quantiles, device=sim.device, dtype=sim.dtype)
        qv = torch.quantile(sim, qs).tolist()
        out["quantiles"] = {float(q): float(v) for q, v in zip(quantiles, qv)}
    return out


def plot_two_similarity_curves(
    sim_log,
    key1,
    key2_a,
    key2_b,
    label_a,
    label_b,
    title,
    save_path,
    dpi=300
):
    layers = sorted(sim_log.keys(), key=lambda x: int(x))
    values_a = []
    values_b = []
    for l in layers:
        entry_a = sim_log[l][key1][key2_a]
        entry_b = sim_log[l][key1][key2_b]
        # If patch stats dict → extract mean
        if isinstance(entry_a, dict):
            values_a.append(entry_a["mean"])
        else:
            values_a.append(entry_a)
        if isinstance(entry_b, dict):
            values_b.append(entry_b["mean"])
        else:
            values_b.append(entry_b)
    layers = [int(l) for l in layers]
    plt.figure(figsize=(6, 4))
    plt.plot(layers, values_a, label=label_a)
    plt.plot(layers, values_b, label=label_b)
    plt.xlabel("Layer Index")
    plt.ylabel("Cosine Similarity")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")



# def plot_three_similarity_panels(
#     sim_log,
#     save_path,
#     dpi=300
# ):
#     layers = sorted(sim_log.keys(), key=lambda x: int(x))
#     layers_int = [int(l) for l in layers]

#     def extract_values(key1, key2):
#         values = []
#         for l in layers:
#             entry = sim_log[l][key1][key2]
#             if isinstance(entry, dict):
#                 values.append(entry["mean"])
#             else:
#                 values.append(entry)
#         return values

#     # --- Extract values ---
#     all_hq_lq  = extract_values("all_tokens", "hq_vs_lq_mean")
#     all_hq_res = extract_values("all_tokens", "hq_vs_res_mean")

#     cam_hq_lq  = extract_values("camera_token", "hq_vs_lq_mean")
#     cam_hq_res = extract_values("camera_token", "hq_vs_res_mean")

#     patch_hq_lq  = extract_values("patch_tokens", "hq_vs_lq")
#     patch_hq_res = extract_values("patch_tokens", "hq_vs_res")

#     # --- Create figure with 3 columns (horizontal layout) ---
#     fig, axes = plt.subplots(
#         1, 3,
#         figsize=(18, 5),
#         sharey=True
#     )

#     # Column 1 — All Tokens
#     axes[0].plot(layers_int, all_hq_lq, label="HQ vs LQ")
#     axes[0].plot(layers_int, all_hq_res, label="HQ vs Restored")
#     axes[0].set_title("All Tokens Similarity Across Layers")
#     axes[0].set_xlabel("Layer Index")
#     axes[0].set_ylabel("Cosine Similarity")
#     axes[0].set_ylim(0.0, 1.0)
#     axes[0].grid(True)
#     axes[0].legend()

#     # Column 2 — Camera Token
#     axes[1].plot(layers_int, cam_hq_lq, label="HQ vs LQ")
#     axes[1].plot(layers_int, cam_hq_res, label="HQ vs Restored")
#     axes[1].set_title("Camera Token Similarity Across Layers")
#     axes[1].set_xlabel("Layer Index")
#     axes[1].set_ylim(0.0, 1.0)
#     axes[1].grid(True)
#     axes[1].legend()

#     # Column 3 — Patch Tokens
#     axes[2].plot(layers_int, patch_hq_lq, label="HQ vs LQ")
#     axes[2].plot(layers_int, patch_hq_res, label="HQ vs Restored")
#     axes[2].set_title("Patch Token Similarity Across Layers")
#     axes[2].set_xlabel("Layer Index")
#     axes[2].set_ylim(0.0, 1.0)
#     axes[2].grid(True)
#     axes[2].legend()

#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
#     plt.close()

#     print(f"Saved combined figure: {save_path}")




def plot_three_similarity_panels(
    sim_log,
    save_path,
    dpi=300,
    fontsize=13,
):
    layers = sorted(sim_log.keys(), key=lambda x: int(x))
    layers_int = np.array([int(l) for l in layers])

    def extract_values(key1, key2):
        values = []
        for l in layers:
            entry = sim_log[l][key1][key2]
            if isinstance(entry, dict):
                values.append(entry["mean"])
            else:
                values.append(entry)
        return np.array(values)

    # --- Extract values ---
    all_hq_lq  = extract_values("all_tokens", "hq_vs_lq_mean")
    all_hq_res = extract_values("all_tokens", "hq_vs_res_mean")

    cam_hq_lq  = extract_values("camera_token", "hq_vs_lq_mean")
    cam_hq_res = extract_values("camera_token", "hq_vs_res_mean")

    patch_hq_lq  = extract_values("patch_tokens", "hq_vs_lq")
    patch_hq_res = extract_values("patch_tokens", "hq_vs_res")

    # -------------------------------------------------
    # Premium Plot Styling
    # -------------------------------------------------
    plt.rcParams.update({
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize + 1,
        "legend.fontsize": fontsize - 1,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "axes.linewidth": 1.1,
    })

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 4.6),
        sharey=True
    )

    color_lq  = "#d62728"
    color_res =  "#1f77b4"

    line_width = 2.1   # thinner & prettier
    marker_size = 5  # slightly smaller markers

    def style_axis(ax, title):
        ax.set_title(title, pad=10)
        ax.set_xlabel("Layer Index")
        ax.set_ylim(0.0, 1.0)

        # Extend axis to 40 even if last layer is 39
        ax.set_xlim(layers_int.min(), 40)

        # Force integer ticks every 2 layers (cleaner)
        ax.xaxis.set_major_locator(MultipleLocator(2))

        # Clean axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Subtle grid
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

        ax.tick_params(axis="both", which="major", length=4)

    # ==============================
    # Panel 1 — All Tokens
    # ==============================
    axes[0].plot(
        layers_int,
        all_hq_lq,
        color=color_lq,
        linewidth=line_width,
        marker="o",
        markersize=marker_size,
        label="LQ"
    )

    axes[0].plot(
        layers_int,
        all_hq_res,
        color=color_res,
        linewidth=line_width,
        marker="s",
        markersize=marker_size,
        label="Restored"
    )

    axes[0].set_ylabel("Cosine Similarity")
    style_axis(axes[0], "All Tokens")
    axes[0].legend(frameon=False, loc="lower right")

    # ==============================
    # Panel 2 — Camera Token
    # ==============================
    axes[1].plot(
        layers_int,
        cam_hq_lq,
        color=color_lq,
        linewidth=line_width,
        marker="o",
        markersize=marker_size,
        label="LQ"
    )

    axes[1].plot(
        layers_int,
        cam_hq_res,
        color=color_res,
        linewidth=line_width,
        marker="s",
        markersize=marker_size,
        label="Restored"
    )

    style_axis(axes[1], "Camera Token")
    axes[1].legend(frameon=False, loc="lower right")

    # ==============================
    # Panel 3 — Patch Tokens
    # ==============================
    axes[2].plot(
        layers_int,
        patch_hq_lq,
        color=color_lq,
        linewidth=line_width,
        marker="o",
        markersize=marker_size,
        label="LQ"
    )

    axes[2].plot(
        layers_int,
        patch_hq_res,
        color=color_res,
        linewidth=line_width,
        marker="s",
        markersize=marker_size,
        label="Restored"
    )

    style_axis(axes[2], "Patch Tokens")
    axes[2].legend(frameon=False, loc="lower right")

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved polished premium figure: {save_path}")
    
    
    
    
def featsim_all(hq_encoder_out, lq_encoder_out, raw_output):

    sim_log={}

    assert hq_encoder_out.aux.keys() == lq_encoder_out.aux.keys() == raw_output.aux.keys()
    for feat_layer_key in hq_encoder_out.aux.keys():
        layer_idx = feat_layer_key.split('_')[-1]

        hq_feat = hq_encoder_out.aux[feat_layer_key]  # (b, V, 1+N, D)
        lq_feat = lq_encoder_out.aux[feat_layer_key]  # (b, V, 1+N, D)
        res_feat = raw_output.aux[feat_layer_key]    # (b, V, 1+N, D)

        
        # tokens
        cam_tkn_hq = hq_feat[:, :, 0:1]    # (b, V, 1, D)
        cam_tkn_lq = lq_feat[:, :, 0:1]
        cam_tkn_res = res_feat[:, :, 0:1]

        patch_tkn_hq = hq_feat[:, :, 1:]   # (b, V, N, D)
        patch_tkn_lq = lq_feat[:, :, 1:]
        patch_tkn_res = res_feat[:, :, 1:]

        # --- overall (all tokens)
        all_hq_lq = cosine_sim_mean(hq_feat, lq_feat)
        all_hq_res = cosine_sim_mean(hq_feat, res_feat)

        # --- camera token similarity (mean over V; the token axis is size 1 anyway)
        cam_hq_lq = cosine_sim_mean(cam_tkn_hq, cam_tkn_lq)
        cam_hq_res = cosine_sim_mean(cam_tkn_hq, cam_tkn_res)

        # --- patch token similarity stats (over V*N tokens)
        patch_hq_lq_stats = cosine_sim_stats(patch_tkn_hq, patch_tkn_lq)
        patch_hq_res_stats = cosine_sim_stats(patch_tkn_hq, patch_tkn_res)
        

        sim_log[layer_idx] = {
            "all_tokens": {
                "hq_vs_lq_mean": all_hq_lq,
                "hq_vs_res_mean": all_hq_res,
            },
            "camera_token": {
                "hq_vs_lq_mean": cam_hq_lq,
                "hq_vs_res_mean": cam_hq_res,
            },
            "patch_tokens": {
                "hq_vs_lq": patch_hq_lq_stats,
                "hq_vs_res": patch_hq_res_stats,
            }
        }
    
    return sim_log

