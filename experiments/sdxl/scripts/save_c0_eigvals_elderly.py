#!/usr/bin/env python3
"""Compute C(0) top eigenvalues + trace for elderly free-conv run.
Saves to results_da_pcavda/c0_eigvals_elderly_gs7p5_fc_dt0p0025.npz."""

import os, sys
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

VAE_SCALE = 0.13025

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_shape = (4, 128, 128)

    from diffusers import StableDiffusionXLPipeline
    print("Loading SDXL VAE...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    vae = pipe.vae.to(device).float().eval()
    del pipe.unet, pipe
    torch.cuda.empty_cache()

    x0_path = "results_multiprompt_sweep/gen_mp_p5_elderly_gs7p5_quick_x0.npz"
    x0 = np.load(x0_path)["x0"]
    print(f"Loaded x0: {x0.shape}")

    from sdxl_components import RealCLIPEmbedder
    emb = RealCLIPEmbedder(vae, device=device, clip_batch_size=16)
    z0 = emb.embed(x0, latent_shape=latent_shape)
    print(f"z0_clip: {z0.shape}")

    z0c = z0 - z0.mean(axis=0)
    C0 = (z0c.T @ z0c) / z0c.shape[0]
    eigvals, _ = np.linalg.eigh(C0.astype(np.float64))
    eigvals = np.sort(eigvals)[::-1]
    tr_C0 = float(np.trace(C0))
    lam1 = float(eigvals[0])

    print(f"tr(C(0))     = {tr_C0:.6e}")
    print(f"lambda_1(0)  = {lam1:.6e}")
    print(f"top 5 eigvals: {eigvals[:5]}")

    out = "results_da_pcavda/c0_eigvals_elderly_gs7p5_fc_dt0p0025.npz"
    np.savez_compressed(out,
        eigenvalues_C0=eigvals,
        tr_C0=np.array(tr_C0),
        lambda1_C0=np.array(lam1),
    )
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
