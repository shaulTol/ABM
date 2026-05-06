"""Decode random samples from each extended-prompt cell and create a multi-page PDF.

Each page = one prompt (6 pages total).
Each page shows 4 columns (gs=1.0, 3.0, 7.5, 15.0) × 5 rows of 4 images = 20 images per gs.
Images are decoded from x0 latents via SDXL VAE.

Usage:
    python decode_sample_grid.py [--n_per_cell 20] [--seed 42]
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SDXL_SHARED = os.path.join(SCRIPT_DIR, '..', 'shared')
sys.path.insert(0, SDXL_SHARED)
from sdxl_components import decode_latents_to_pil

PROMPTS = [
    (0, "bat",    "a photo of a bat"),
    (1, "crane",  "a photo of a crane"),
    (2, "mouse",  "a photo of a mouse"),
    (3, "seal",   "a photo of a seal"),
    (4, "jaguar", "a photo of a jaguar"),
    (5, "palm",   "a photo of a palm"),
]
GS_VALUES = ["1p0", "3p0", "7p5", "15p0"]
GS_LABELS = ["w=1.0", "w=3.0", "w=7.5", "w=15.0"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_per_cell", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="extended_prompts_samples.pdf")
    p.add_argument("--thumb_size", type=int, default=256,
                   help="Resize decoded images to this size (px)")
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE
    from diffusers import AutoencoderKL
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae",
                                         torch_dtype=torch.float32)
    vae = vae.to(device).eval()
    vae.enable_tiling()
    vae.enable_slicing()
    print(f"VAE loaded on {device}")

    # Load SSA values from JSON files
    import json
    ssa_values = {}
    for pi, label, _ in PROMPTS:
        for gs in GS_VALUES:
            json_path = os.path.join(SCRIPT_DIR,
                f"clip_da_ext_p{pi}_{label}_gs{gs}_dt0p005.json")
            if os.path.exists(json_path):
                with open(json_path) as f:
                    d = json.load(f)
                ssa_values[(pi, gs)] = d.get("ssa_fixed20", None)

    # Build PDF
    n_cols_img = 4  # images per row within a gs column
    n_rows_img = (args.n_per_cell + n_cols_img - 1) // n_cols_img  # rows of images

    with PdfPages(args.output) as pdf:
        for pi, label, prompt in PROMPTS:
            print(f"\n=== Page {pi+1}/6: {prompt} ===")

            # Decode all 4 gs variants
            all_grids = []
            for gi, gs in enumerate(GS_VALUES):
                npz_path = os.path.join(SCRIPT_DIR,
                    f"gen_ext_p{pi}_{label}_gs{gs}_quick_x0.npz")
                if not os.path.exists(npz_path):
                    print(f"  MISSING: {npz_path}")
                    all_grids.append(None)
                    continue

                x0 = np.load(npz_path)["x0"]
                N = x0.shape[0]
                idx = rng.choice(N, size=min(args.n_per_cell, N), replace=False)
                idx.sort()
                x0_sub = x0[idx]

                # Latent shape for 1024x1024: (4, 128, 128)
                images = decode_latents_to_pil(vae, x0_sub,
                    latent_shape=(4, 128, 128),
                    batch_size=args.batch_size, device=device)

                # Resize
                images = [im.resize((args.thumb_size, args.thumb_size),
                                     Image.LANCZOS) for im in images]

                # Make grid image
                w, h = args.thumb_size, args.thumb_size
                grid_w = n_cols_img * w
                grid_h = n_rows_img * h
                grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
                for i, img in enumerate(images):
                    r, c = divmod(i, n_cols_img)
                    grid.paste(img, (c * w, r * h))
                all_grids.append(grid)
                print(f"  {GS_LABELS[gi]}: decoded {len(images)} images")

            # Create matplotlib figure: 4 columns
            fig, axes = plt.subplots(1, 4, figsize=(20, 5 * n_rows_img + 1.5))
            fig.suptitle(f'"{prompt}"', fontsize=18, fontweight='bold', y=0.98)

            for gi, (gs, gs_label) in enumerate(zip(GS_VALUES, GS_LABELS)):
                ax = axes[gi]
                ssa = ssa_values.get((pi, gs), None)
                ssa_str = f"  (Ŝ={ssa:.2f})" if ssa is not None else ""
                ax.set_title(f"{gs_label}{ssa_str}", fontsize=14)
                ax.axis('off')
                if all_grids[gi] is not None:
                    ax.imshow(np.array(all_grids[gi]))

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  Page {pi+1} saved.")

    print(f"\nDone: {args.output}")


if __name__ == "__main__":
    main()
