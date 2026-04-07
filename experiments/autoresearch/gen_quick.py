#!/usr/bin/env python3
"""Quick generation script for testing prompts."""
import os, sys, argparse, torch
from PIL import Image
from diffusers import AutoPipelineForText2Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--n_images", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="results_quick")
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16",
    ).to("cuda")

    images = []
    for i in range(args.n_images):
        img = pipe(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            original_size=(1024, 1024), target_size=(1024, 1024),
            generator=torch.Generator("cuda").manual_seed(i),
        ).images[0]
        img.save(f"{args.output_dir}/img_{i:03d}.png")
        images.append(img)
        print(f"  {i+1}/{args.n_images}")

    # Make grid
    cols, rows = 5, 4
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for idx, img in enumerate(images[:cols*rows]):
        grid.paste(img, ((idx % cols) * w, (idx // cols) * h))
    grid.save(f"{args.output_dir}/grid.png")
    print(f"Saved grid to {args.output_dir}/grid.png")

if __name__ == "__main__":
    main()
