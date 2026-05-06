#!/usr/bin/env python3
"""
Extract z_tau CLIP embeddings from the Langevin checkpoint for SDXL elderly.

Loads snapshot_2000 (200, 65536) fp16 latents from the checkpoint,
VAE decodes + CLIP embeds them, and saves z_tau (200, 768) float32.

~1 min on GPU. Produces clip_da_elderly_gs7p5_fc_z_tau.npz.
"""
import sys
import os
import time
import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'experiments', 'sdxl', 'scripts'))

from sdxl_components import RealCLIPEmbedder


def main():
    ckpt_path = os.path.join(project_root,
        'experiments/sdxl/results_da_pcavda/langevin_ckpt_elderly_gs7p5_fc_dt0p0025.npz')
    output_dir = os.path.join(project_root,
        'experiments/sdxl/results_da_pcavda')
    z0_clip_path = os.path.join(project_root,
        'experiments/sdxl/results_multiprompt_sweep/gen_mp_p5_elderly_gs7p5_quick_clip.npz')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load latent snapshot at τ*=2000
    print(f"Loading checkpoint: {ckpt_path}")
    t0 = time.time()
    ckpt = np.load(ckpt_path)
    x_tau = np.array(ckpt['snapshot_2000'], dtype=np.float32)  # fp32 to match VAE
    x0 = ckpt['x0_flat']  # (200, 65536) fp32
    print(f"  Loaded in {time.time()-t0:.1f}s, x_tau={x_tau.shape}, x0={x0.shape}")

    # Also load x0 for verification
    z0_clip = np.load(z0_clip_path)['z0']  # (200, 768)

    # Build VAE + CLIP embedder
    latent_shape = (4, 128, 128)
    print("Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae",
        torch_dtype=torch.float32  # must be fp32 — RealCLIPEmbedder.embed() casts to .float()
    ).to(device)
    vae.eval()

    print("Building CLIP embedder...")
    embedder = RealCLIPEmbedder(vae=vae, device=device)

    # Embed z_tau
    print("Embedding z_tau (lag=2000)...")
    t0 = time.time()
    z_tau = embedder.embed(x_tau, latent_shape=latent_shape, batch_size=4)
    print(f"  Done in {time.time()-t0:.1f}s, z_tau={z_tau.shape}")

    # Also re-embed z0 for cross-check
    print("Re-embedding z0 for cross-check...")
    z0_reembed = embedder.embed(x0, latent_shape=latent_shape, batch_size=4)
    z0_err = np.max(np.abs(z0_reembed - z0_clip))
    print(f"  z0 re-embed max error vs saved: {z0_err:.6f}")

    # Verify: compute C(τ*) from z0 and z_tau, compare to stored
    da_path = os.path.join(output_dir, 'clip_da_elderly_gs7p5_fc_dt0p0025.npz')
    da = np.load(da_path)
    mu_clip = da['mu_clip']
    obs_evals = da['eigenvalues_full_at_tau_star']

    z0_c = z0_clip.astype(np.float64) - mu_clip.astype(np.float64)
    z_tau_c = z_tau.astype(np.float64) - mu_clip.astype(np.float64)
    C_check = z_tau_c.T @ z0_c / 200
    C_check_sym = 0.5 * (C_check + C_check.T)
    check_evals = np.linalg.eigvalsh(C_check_sym)[::-1]
    eval_err = np.max(np.abs(check_evals[:5] - obs_evals[:5]))
    print(f"  C(τ*) eigenvalue cross-check error (top-5): {eval_err:.6f}")

    # Save
    out_path = os.path.join(output_dir, 'clip_da_elderly_gs7p5_fc_z_tau.npz')
    np.savez(out_path, z_tau=z_tau, z0_reembed=z0_reembed)
    print(f"\nSaved {out_path}")
    print(f"  z_tau: {z_tau.shape} {z_tau.dtype}")


if __name__ == '__main__':
    main()
