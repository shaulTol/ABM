#!/usr/bin/env python3
"""Generate 'a person' at a single guidance scale with SDXL Base, save grid."""

import os
import argparse
import torch
import numpy as np
from PIL import Image


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
    parser.add_argument("--guidance_scale", type=float, required=True)
    parser.add_argument("--output_dir", type=str, default="results_guidance_sweep")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from diffusers import AutoPipelineForText2Image

    # Load SDXL Base pipeline
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)

    gs = args.guidance_scale
    print(f"Generating {args.n_images} images with guidance_scale={gs}")
    print(f"Prompt: '{args.prompt}'")

    images = []
    for i in range(args.n_images):
        gen = torch.Generator(device=device).manual_seed(i)
        img = pipe(
            args.prompt,
            num_inference_steps=50,
            guidance_scale=gs,
            generator=gen,
            width=1024,
            height=1024,
            original_size=(1024, 1024),
            target_size=(1024, 1024),
        ).images[0]
        images.append(img)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_images}")

    grid = make_grid(images, cols=10)
    grid.save(os.path.join(args.output_dir, f"grid_gs{gs}.png"))
    print(f"Saved grid_gs{gs}.png")


if __name__ == "__main__":
    main()
