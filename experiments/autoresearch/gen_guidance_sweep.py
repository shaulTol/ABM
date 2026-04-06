#!/usr/bin/env python3
"""Generate 'a person' at multiple guidance scales, save grids as PDF."""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForText2Image
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def make_grid(images, cols=10):
    n = len(images)
    rows = (n + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for i, img in enumerate(images):
        grid.paste(img, ((i % cols) * w, (i // cols) * h))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", type=int, default=50)
    parser.add_argument("--prompt", type=str, default="a person")
    parser.add_argument("--output_dir", type=str, default="results_guidance_sweep")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    guidance_scales = [0.1, 0.5, 1.0, 2.5, 7.5]

    # Load Turbo pipeline
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)

    for gs in guidance_scales:
        print(f"\n{'='*60}")
        print(f"Generating {args.n_images} images with guidance_scale={gs}")
        print(f"{'='*60}")

        images = []
        for i in range(args.n_images):
            gen = torch.Generator(device=device).manual_seed(i)
            img = pipe(
                args.prompt,
                num_inference_steps=4,
                guidance_scale=gs,
                generator=gen,
                width=512,
                height=512,
            ).images[0]
            images.append(img)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{args.n_images}")

        grid = make_grid(images, cols=10)
        grid.save(os.path.join(args.output_dir, f"grid_gs{gs}.png"))
        print(f"  Saved grid_gs{gs}.png")

    # Combine all grids into one PDF
    pdf_path = os.path.join(args.output_dir, "guidance_sweep.pdf")
    with PdfPages(pdf_path) as pdf:
        for gs in guidance_scales:
            grid = Image.open(os.path.join(args.output_dir, f"grid_gs{gs}.png"))
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(np.array(grid))
            ax.set_title(f'"{args.prompt}" — guidance_scale={gs}', fontsize=16)
            ax.axis("off")
            plt.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"\nSaved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
