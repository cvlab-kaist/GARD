import os
import torch
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt


def radial_profile(mag):
    H, W = mag.shape
    y, x = np.indices((H, W))
    center = (H//2, W//2)
    r = np.sqrt((x-center[1])**2 + (y-center[0])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), mag.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / (nr + 1e-8)
    return radialprofile


def compute_fft_per_view(feat, grid_h, grid_w, avg_embedding=True):
    """
    Returns list of FFT magnitude spectra per view.
    Assumes batch size = 1.
    """
    B, V, N, D = feat.shape
    assert B == 1, "This version assumes batch size = 1"

    feat = feat[:, :, 1:, :]  # remove camera token
    feat = feat.reshape(B, V, grid_h, grid_w, D)

    if avg_embedding:
        feat = feat.mean(dim=-1)  # (1, V, H, W)

    spectra = []

    for v in range(V):
        f = feat[0, v]  # (H, W)
        freq = fft.fft2(f, dim=(-2, -1))
        freq = fft.fftshift(freq, dim=(-2, -1))
        mag = torch.log1p(torch.abs(freq))
        spectra.append(mag.detach().cpu().numpy())

    return spectra


def plot_freq_analysis_per_view(
    hq_out,
    lq_out,
    res_out,
    save_root,
    scene,
    patch_grid=(27, 36),
    avg_embedding=True
):
    os.makedirs(save_root, exist_ok=True)

    layers = list(hq_out.aux.keys())
    V = hq_out.aux[layers[0]].shape[1]
    L = len(layers)

    # -------------------------------------------------
    # Compute all spectra first (needed for single fig)
    # -------------------------------------------------
    all_data = {}

    for layer in layers:
        hq_specs = compute_fft_per_view(hq_out.aux[layer], *patch_grid, avg_embedding)
        lq_specs = compute_fft_per_view(lq_out.aux[layer], *patch_grid, avg_embedding)
        res_specs = compute_fft_per_view(res_out.aux[layer], *patch_grid, avg_embedding)

        # shared normalization per layer
        all_specs = hq_specs + lq_specs + res_specs
        vmin = min(s.min() for s in all_specs)
        vmax = max(s.max() for s in all_specs)

        all_data[layer] = {
            "hq": hq_specs,
            "lq": lq_specs,
            "res": res_specs,
            "vmin": vmin,
            "vmax": vmax
        }

    # =================================================
    # 1️⃣ HEATMAPS (ALL LAYERS IN ONE IMAGE)
    # =================================================
    fig_hm, axs_hm = plt.subplots(
        L, V * 3,
        figsize=(4 * V * 3, 4 * L)
    )

    if L == 1:
        axs_hm = np.expand_dims(axs_hm, 0)

    for row_idx, layer in enumerate(layers):
        data = all_data[layer]

        for v in range(V):
            axs_hm[row_idx, v*3 + 0].imshow(
                data["hq"][v],
                cmap="inferno",
                vmin=data["vmin"],
                vmax=data["vmax"]
            )
            axs_hm[row_idx, v*3 + 0].set_title(f"{layer} | HQ V{v}")

            axs_hm[row_idx, v*3 + 1].imshow(
                data["lq"][v],
                cmap="inferno",
                vmin=data["vmin"],
                vmax=data["vmax"]
            )
            axs_hm[row_idx, v*3 + 1].set_title(f"{layer} | LQ V{v}")

            axs_hm[row_idx, v*3 + 2].imshow(
                data["res"][v],
                cmap="inferno",
                vmin=data["vmin"],
                vmax=data["vmax"]
            )
            axs_hm[row_idx, v*3 + 2].set_title(f"{layer} | RES V{v}")

            for ax in axs_hm[row_idx, v*3:v*3+3]:
                ax.axis("off")

    plt.suptitle(f"{scene} - FFT Heatmaps (All Layers & Views)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, f"{scene}_ALL_layers_heatmaps.png"))
    plt.close()

    # =================================================
    # 2️⃣ RADIAL PROFILES (ALL LAYERS IN ONE IMAGE)
    # =================================================
    fig_rad, axs_rad = plt.subplots(
        L, V,
        figsize=(5 * V, 4 * L)
    )

    if L == 1:
        axs_rad = np.expand_dims(axs_rad, 0)

    for row_idx, layer in enumerate(layers):
        data = all_data[layer]

        for v in range(V):
            hq_rad = radial_profile(data["hq"][v])
            lq_rad = radial_profile(data["lq"][v])
            res_rad = radial_profile(data["res"][v])

            axs_rad[row_idx, v].plot(hq_rad, label="HQ", linewidth=2)
            axs_rad[row_idx, v].plot(lq_rad, label="LQ", linewidth=2)
            axs_rad[row_idx, v].plot(res_rad, label="RES", linewidth=2)

            axs_rad[row_idx, v].set_title(f"{layer} | View {v}")
            axs_rad[row_idx, v].set_xlabel("Frequency radius")
            axs_rad[row_idx, v].set_ylabel("Energy")
            axs_rad[row_idx, v].legend()

    plt.suptitle(f"{scene} - Radial Frequency Profiles (All Layers)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, f"{scene}_ALL_layers_radial_profiles.png"))
    plt.close()

    print("Saved:")
    print(" - ALL_layers_heatmaps.png")
    print(" - ALL_layers_radial_profiles.png")