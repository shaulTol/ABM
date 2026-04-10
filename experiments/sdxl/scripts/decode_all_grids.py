"""Decode all latents to a big grid image for each experiment."""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from sdxl_components import decode_latents_to_pil, make_grid_png


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_npz", type=str, required=True)
    p.add_argument("--model_id", type=str, default="stabilityai/sdxl-turbo")
    p.add_argument("--n_cols", type=int, default=20)
    p.add_argument("--output_path", type=str, default=None)
    p.add_argument("--latent_h", type=int, default=64, help="latent H (64 for 512x512, 128 for 1024x1024)")
    p.add_argument("--latent_w", type=int, default=64, help="latent W")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_images", type=int, default=None, help="limit N for speed")
    p.add_argument("--thumb_size", type=int, default=None, help="downsize each image to N px before gridding")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lat = dict(np.load(args.latents_npz))
    x0 = lat["x0"]
    if args.max_images is not None:
        x0 = x0[:args.max_images]
    N = x0.shape[0]
    print(f"Loaded {N} latents from {args.latents_npz}")

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae",
                                         torch_dtype=torch.float32)
    vae = vae.to(device).eval()
    vae.enable_tiling()
    vae.enable_slicing()

    print(f"Decoding {N} latents (shape=4x{args.latent_h}x{args.latent_w}, bs={args.batch_size}, tiled)...")
    images = decode_latents_to_pil(vae, x0, latent_shape=(4, args.latent_h, args.latent_w),
                                    batch_size=args.batch_size, device=device)
    if args.thumb_size is not None:
        images = [im.resize((args.thumb_size, args.thumb_size), Image.LANCZOS) for im in images]

    out = args.output_path
    if out is None:
        base = os.path.basename(args.latents_npz).replace("clip_da_latents_", "").replace(".npz", "")
        out = os.path.join(os.path.dirname(args.latents_npz), f"all_images_{base}.png")

    make_grid_png(images, n_cols=args.n_cols, path=out)
    print(f"Saved: {out} ({N} images, {args.n_cols} cols)")


if __name__ == "__main__":
    main()
