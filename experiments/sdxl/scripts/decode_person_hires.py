#!/usr/bin/env python3
"""Decode first 20 images from each guidance-sweep x0 file at full resolution.
Saves individual PNGs: person_gs{X}_img{0-19}.png
Uses mmap + single-image decode to avoid OOM.
"""
import os, sys
import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results_guidance_sweep')
OUT_DIR = os.path.join(RESULTS_DIR, 'person_hires')
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                     subfolder="vae", torch_dtype=torch.float16)
vae = vae.to(device).eval()
vae.enable_tiling()
vae.enable_slicing()

N_DECODE = 20

for gs_tag in ['1p0', '3p0', '7p5', '15p0']:
    x0_path = os.path.join(RESULTS_DIR, f'gen_sweep_gs{gs_tag}_x0.npz')
    if not os.path.exists(x0_path):
        x0_path = os.path.join(RESULTS_DIR, f'gen_sweep_gs{gs_tag}_quick_x0.npz')
    if not os.path.exists(x0_path):
        print(f'SKIP {x0_path}')
        continue

    # Load with mmap to avoid reading entire file into RAM
    x0_all = np.load(x0_path, mmap_mode='r')['x0']
    n = min(N_DECODE, x0_all.shape[0])
    print(f'gs={gs_tag}: decoding {n}/{x0_all.shape[0]} latents...')

    for i in range(n):
        out = os.path.join(OUT_DIR, f'person_gs{gs_tag}_img{i}.png')
        if os.path.exists(out) and os.path.getsize(out) > 50000:
            print(f'  skip {out} (exists)')
            continue

        # Copy single latent to avoid keeping mmap reference on GPU
        x0_i = np.array(x0_all[i:i+1])
        latent = torch.from_numpy(x0_i).half().reshape(1, 4, 128, 128).to(device)
        latent = latent / vae.config.scaling_factor
        with torch.no_grad():
            img = vae.decode(latent).sample
        del latent
        torch.cuda.empty_cache()

        img = (img / 2 + 0.5).clamp(0, 1)
        img = (img[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(out)
        print(f'  saved {out}', flush=True)

print('DONE')
