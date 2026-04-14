#!/usr/bin/env python3
"""
SDXL Square N=1000: noise-floor analysis.

Loads the Langevin checkpoint (raw latents), VAE-decodes, CLIP-encodes,
then computes permutation floor at each lag. Compares to eigenvalues
from clip_da_autoresearch_square_N1000_T1500.npz.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import json
import time
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figures.style_config import setup_style, SSA_COLOR, MI_COLOR, TEXT_WIDTH

setup_style()

from experiments.noise_floor.noise_floor_utils import (
    analytic_floor, permutation_floor, find_tau_star
)

OUTDIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'figures')
SDXL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'sdxl')


def load_clip_model(device):
    """Load CLIP ViT-L/14."""
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    model = model.to(device).eval()
    return model, processor


def load_vae(device):
    """Load SDXL VAE."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        'stabilityai/sdxl-turbo', subfolder='vae',
        torch_dtype=torch.float16
    ).to(device).eval()
    vae.enable_tiling()
    return vae


def decode_and_clip_encode(latents_flat, vae, clip_model, clip_processor, device,
                           batch_size=8, latent_shape=(4, 64, 64)):
    """Decode latents to images, then CLIP-encode to 768D embeddings."""
    from PIL import Image
    N = latents_flat.shape[0]
    latents = torch.tensor(latents_flat, dtype=torch.float16, device=device)
    latents = latents.reshape(N, *latent_shape)

    all_embeddings = []
    for i in range(0, N, batch_size):
        batch = latents[i:i+batch_size]
        # VAE decode
        with torch.no_grad():
            decoded = vae.decode(batch / vae.config.scaling_factor).sample
        # To PIL
        imgs = ((decoded.float().clamp(-1, 1) + 1) / 2 * 255).byte()
        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
        pil_imgs = [Image.fromarray(img) for img in imgs]
        # CLIP encode
        inputs = clip_processor(images=pil_imgs, return_tensors='pt').to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
        all_embeddings.append(emb.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)  # (N, 768)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_global = time.time()
    print(f'Device: {device}')

    # --- Load eigenvalue data ---
    results_dir = os.path.join(SDXL_DIR, 'results_autoresearch')
    da_npz = np.load(os.path.join(results_dir,
                     'clip_da_autoresearch_square_N1000_T1500.npz'),
                     allow_pickle=True)
    lags = da_npz['snapshot_lags']  # (20,)
    dt = 0.001

    all_eigenvalues = np.zeros((len(lags), 10))
    for i, lag in enumerate(lags):
        all_eigenvalues[i] = da_npz[f'eigenvalues_lag{lag}']

    print(f'Loaded eigenvalues: {len(lags)} lags, range [{lags[0]}, {lags[-1]}]')
    print(f'λ_1 range: [{all_eigenvalues[:,0].min():.5f}, {all_eigenvalues[:,0].max():.5f}]')

    # --- Load Langevin checkpoint ---
    ckpt_path = os.path.join(results_dir,
                             'langevin_ckpt_autoresearch_square_N1000_T1500.npz')
    print(f'\nLoading checkpoint: {ckpt_path}')
    ckpt = np.load(ckpt_path, allow_pickle=True)
    x0_flat = ckpt['x0_flat']  # (1000, 16384)
    N = x0_flat.shape[0]
    print(f'N={N}, latent_dim={x0_flat.shape[1]}')

    # --- Load models ---
    print('\nLoading VAE + CLIP...')
    vae = load_vae(device)
    clip_model, clip_processor = load_clip_model(device)

    # --- CLIP-encode x0 ---
    print('\nCLIP-encoding x0...')
    t0 = time.time()
    clip_x0 = decode_and_clip_encode(x0_flat, vae, clip_model, clip_processor, device)
    print(f'  x0: {clip_x0.shape} [{time.time()-t0:.0f}s]')

    mu_clip = clip_x0.mean(axis=0)
    x0_centered = clip_x0 - mu_clip

    d = clip_x0.shape[1]  # 768
    sigma_f_sq = np.sum(x0_centered**2) / (N * d)
    gamma = d / N
    print(f'  d={d}, N={N}, sigma_f_sq={sigma_f_sq:.6f}, gamma={gamma:.3f}')

    # --- CLIP-encode snapshots + permutation floor at each lag ---
    print(f'\nComputing permutation floor at {len(lags)} lags...')
    perm_floors = []
    analytic_floors_list = []

    for i, lag in enumerate(lags):
        t0 = time.time()
        snap_key = f'snapshot_{lag}'
        x_tau_flat = ckpt[snap_key]  # (1000, 16384)

        # CLIP-encode x_tau
        clip_x_tau = decode_and_clip_encode(
            x_tau_flat, vae, clip_model, clip_processor, device)
        x_tau_centered = clip_x_tau - mu_clip

        # Permutation floor
        pf = permutation_floor(x0_centered, x_tau_centered,
                                n_perm=20, quantile=0.95)
        perm_floors.append(pf)

        # Analytic floor
        af = analytic_floor(lag * dt, sigma_f_sq, d, N)
        analytic_floors_list.append(af)

        elapsed = time.time() - t0
        print(f'  lag={lag:4d}: λ₁={all_eigenvalues[i,0]:.5f}, '
              f'perm_floor={pf:.5f}, analytic={af:.6f} [{elapsed:.0f}s]')

    perm_floors = np.array(perm_floors)
    analytic_floors_arr = np.array(analytic_floors_list)

    # --- Find τ* ---
    print('\nFinding τ*...')
    tau_star_idx_m1, tau_star_m1 = find_tau_star(
        lags, all_eigenvalues, perm_floors, m=1)
    tau_star_idx_m2, tau_star_m2 = find_tau_star(
        lags, all_eigenvalues, perm_floors, m=2)

    if tau_star_m1 is not None:
        print(f'  τ*(m=1): lag={tau_star_m1}, λ₁={all_eigenvalues[tau_star_idx_m1,0]:.5f}, '
              f'floor={perm_floors[tau_star_idx_m1]:.5f}')
    else:
        print('  τ*(m=1): NO lag above floor!')

    if tau_star_m2 is not None:
        print(f'  τ*(m=2): lag={tau_star_m2}, '
              f'λ₁={all_eigenvalues[tau_star_idx_m2,0]:.5f}, '
              f'λ₂={all_eigenvalues[tau_star_idx_m2,1]:.5f}, '
              f'floor={perm_floors[tau_star_idx_m2]:.5f}')
    else:
        print('  τ*(m=2): NO lag above floor!')

    # Compare to old gap-max
    print(f'\n  Old gap-max (N=200): τ*=1189, DA1~gender ρ=0.905')
    print(f'  New gap-max (N=1000): τ*=1265, DA1~gender ρ=0.843')

    # --- Save JSON ---
    results = {
        'experiment': 'sdxl_square_noise_floor_N1000',
        'parameters': {
            'd': d, 'N': N, 'dt': dt, 'T': 1500,
            'sigma_f_sq': float(sigma_f_sq),
            'gamma': float(gamma),
        },
        'lags': lags.tolist(),
        'eigenvalues_top5': all_eigenvalues[:, :5].tolist(),
        'perm_floor': perm_floors.tolist(),
        'analytic_floor': analytic_floors_arr.tolist(),
        'tau_star_m1': {
            'lag': int(tau_star_m1) if tau_star_m1 is not None else None,
            'lambda_1': float(all_eigenvalues[tau_star_idx_m1, 0]) if tau_star_m1 is not None else None,
            'floor': float(perm_floors[tau_star_idx_m1]) if tau_star_m1 is not None else None,
        },
        'tau_star_m2': {
            'lag': int(tau_star_m2) if tau_star_m2 is not None else None,
            'lambda_1': float(all_eigenvalues[tau_star_idx_m2, 0]) if tau_star_m2 is not None else None,
            'lambda_2': float(all_eigenvalues[tau_star_idx_m2, 1]) if tau_star_m2 is not None else None,
            'floor': float(perm_floors[tau_star_idx_m2]) if tau_star_m2 is not None else None,
        },
    }

    json_path = os.path.join(OUTDIR, 'noise_floor_sdxl_N1000.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {json_path}')

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.6))

    ax.plot(lags, all_eigenvalues[:, 0], 'o-', color=SSA_COLOR,
            markersize=4, lw=1.4, label=r'$\lambda_1(\tau)$')
    ax.plot(lags, all_eigenvalues[:, 1], 's-', color='#2ca02c',
            markersize=3.5, lw=1.2, label=r'$\lambda_2(\tau)$')
    ax.plot(lags, all_eigenvalues[:, 2], '^-', color='#ff7f0e',
            markersize=3, lw=1.0, alpha=0.7, label=r'$\lambda_3(\tau)$')
    ax.plot(lags, perm_floors, 'x-', color=MI_COLOR,
            markersize=5, lw=1.2, label='Permutation floor (95th pctile)')
    ax.plot(lags, analytic_floors_arr, '--', color='#999999',
            lw=1.0, label='Analytic floor')

    if tau_star_m1 is not None:
        ax.axvline(tau_star_m1, color='black', ls=':', lw=0.8, alpha=0.7)
        ax.annotate(rf'$\tau^*$ = {tau_star_m1}',
                    xy=(tau_star_m1, all_eigenvalues[tau_star_idx_m1, 0]),
                    xytext=(tau_star_m1 + 100, all_eigenvalues[tau_star_idx_m1, 0] * 1.1),
                    fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.6))

    ax.set_xlabel(r'Lag $\tau$ (Langevin steps)')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('SDXL Square: noise floor (d=768, N=1000)', fontsize=9)
    ax.legend(fontsize=6.5, loc='upper right')
    ax.grid(True, alpha=0.2, linewidth=0.4)

    fig_path = os.path.join(FIGDIR, 'fig_sdxl_noise_floor_N1000.png')
    fig.savefig(fig_path)
    plt.close(fig)
    print(f'Saved {fig_path}')

    total = time.time() - t_global
    print(f'\nTotal: {total:.0f}s = {total/60:.1f}min')


if __name__ == '__main__':
    main()
