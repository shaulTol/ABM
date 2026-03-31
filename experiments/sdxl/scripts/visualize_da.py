"""
visualize_da.py — Decode DA direction extremes from CLIP space to pixel space.

For each DA direction (DA1, DA2, DA3):
  - Project all N CLIP embeddings onto the direction
  - Select top-16 and bottom-16 latents by projection value
  - Decode latents to pixels via VAE decoder
  - Save image grid: top row = high projection, bottom row = low projection

Usage:
    python scripts/visualize_da.py \
        --results_npz results_guidance_sweep/sdxl_da_sdxl_base_w7.5.npz \
        --latents_npz results_guidance_sweep/sdxl_latents_sdxl_base_w7.5.npz \
        --model_id stabilityai/stable-diffusion-xl-base-1.0 \
        --output_dir figures
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def decode_latents_to_images(vae, latents_flat, latent_shape=(4, 64, 64),
                             batch_size=8, device="cuda"):
    """Decode flat latent vectors to PIL images via VAE decoder.

    Args:
        vae: SDXL VAE decoder
        latents_flat: (N, d) latents in pipeline scale
        latent_shape: (4, H//8, W//8)
        batch_size: decode in batches

    Returns:
        list of PIL Images
    """
    images = []
    VAE_SCALE = 0.13025

    for start in range(0, len(latents_flat), batch_size):
        end = min(start + batch_size, len(latents_flat))
        batch = latents_flat[start:end].to(device).float()
        z = batch.view(-1, *latent_shape)

        # Undo pipeline scaling and decode
        z = z / VAE_SCALE
        with torch.no_grad():
            decoded = vae.decode(z, return_dict=False)[0]

        # (B, 3, H, W) in [-1, 1] -> [0, 255]
        decoded = (decoded + 1.0) / 2.0
        decoded = decoded.clamp(0, 1)
        decoded = (decoded * 255).byte().cpu()

        for img_tensor in decoded:
            img = img_tensor.permute(1, 2, 0).numpy()
            images.append(Image.fromarray(img))

    return images


def make_da_grid(images_high, images_low, n_cols=8, label="DA1"):
    """Create image grid: top rows = high projection, bottom rows = low projection."""
    if not images_high or not images_low:
        return None

    w, h = images_high[0].size
    n_high = len(images_high)
    n_low = len(images_low)
    rows_high = (n_high + n_cols - 1) // n_cols
    rows_low = (n_low + n_cols - 1) // n_cols
    total_rows = rows_high + rows_low + 1  # +1 for separator

    grid = Image.new("RGB", (n_cols * w, total_rows * h), color=(255, 255, 255))

    # Top: high projection
    for i, img in enumerate(images_high):
        row, col = divmod(i, n_cols)
        grid.paste(img, (col * w, row * h))

    # Bottom: low projection (after gap)
    offset_y = (rows_high + 1) * h
    for i, img in enumerate(images_low):
        row, col = divmod(i, n_cols)
        grid.paste(img, (col * w, offset_y + row * h))

    return grid


def parse_args():
    p = argparse.ArgumentParser(description="Visualize DA direction extremes")
    p.add_argument("--results_npz", type=str, required=True,
                   help="Path to sdxl_da_*.npz with DA directions")
    p.add_argument("--latents_npz", type=str, required=True,
                   help="Path to sdxl_latents_*.npz with x0 latents")
    p.add_argument("--model_id", type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--n_extremes", type=int, default=16,
                   help="Number of top/bottom images per DA direction")
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(EXP_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load DA results and latents
    print("Loading results...")
    da_data = np.load(args.results_npz)
    lat_data = np.load(args.latents_npz)

    x0 = torch.from_numpy(lat_data["x0"])  # (N, d)
    N = x0.shape[0]

    # Load Latent-CLIP to project x0 into CLIP space
    print("Loading Latent-CLIP...")
    sys.path.insert(0, os.path.join(SCRIPT_DIR))
    from compute_sdxl_ssa import LatentCLIPEmbedder
    clip_embedder = LatentCLIPEmbedder(device)

    print("Embedding x0 into CLIP space...")
    z0_clip = clip_embedder.embed(x0)  # (N, 512)

    # Load VAE for decoding
    print("Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae",
                                         torch_dtype=torch.float32)
    vae = vae.to(device)
    vae.eval()

    # Extract name from results file for output naming
    basename = os.path.basename(args.results_npz).replace("sdxl_da_", "").replace(".npz", "")

    # Process each DA direction
    for k, key in enumerate(["da1_eigvec", "da2_eigvec"]):
        if key not in da_data:
            continue

        da_vec = torch.from_numpy(da_data[key]).float()  # (512,)
        da_name = f"DA{k+1}"
        print(f"\n=== {da_name} ===")

        # Project CLIP embeddings onto DA direction
        projections = (z0_clip.float() @ da_vec).numpy()

        # Find extremes
        high_idx = np.argsort(projections)[-args.n_extremes:][::-1]
        low_idx = np.argsort(projections)[:args.n_extremes]

        print(f"  High projections: {projections[high_idx[:3]]}")
        print(f"  Low projections: {projections[low_idx[:3]]}")

        # Decode extremes
        print(f"  Decoding {2*args.n_extremes} latents...")
        high_images = decode_latents_to_images(vae, x0[high_idx], device=device)
        low_images = decode_latents_to_images(vae, x0[low_idx], device=device)

        # Make grid
        grid = make_da_grid(high_images, low_images, label=da_name)
        if grid is not None:
            out_path = os.path.join(output_dir, f"sdxl_{da_name.lower()}_extremes_{basename}.png")
            grid.save(out_path)
            print(f"  Saved: {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
