#!/usr/bin/env python3
"""
Generate images from two prompts to compute a grounded CLIP direction.
E.g., 50 "left profile" + 50 "right profile" → mean difference = left/right direction.
Then project existing experiment's images onto this direction.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from search_parallel_ellipses import load_clip_model, clip_encode_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_a", type=str, required=True,
                        help="Prompt for group A")
    parser.add_argument("--prompt_b", type=str, required=True,
                        help="Prompt for group B")
    parser.add_argument("--n_images", type=int, default=50,
                        help="Number of images per prompt")
    parser.add_argument("--output_dir", type=str, default="results_direction")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/group_a", exist_ok=True)
    os.makedirs(f"{args.output_dir}/group_b", exist_ok=True)

    # Load Turbo pipeline
    from diffusers import AutoPipelineForText2Image
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    # Generate group A
    print(f"Generating {args.n_images} images for prompt A: {args.prompt_a}")
    images_a = []
    for i in range(args.n_images):
        img = pipe(
            prompt=args.prompt_a,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(i),
        ).images[0]
        img.save(f"{args.output_dir}/group_a/img_{i:03d}.png")
        images_a.append(img)
        if (i + 1) % 10 == 0:
            print(f"  Group A: {i+1}/{args.n_images}")

    # Generate group B
    print(f"Generating {args.n_images} images for prompt B: {args.prompt_b}")
    images_b = []
    for i in range(args.n_images):
        img = pipe(
            prompt=args.prompt_b,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(i),
        ).images[0]
        img.save(f"{args.output_dir}/group_b/img_{i:03d}.png")
        images_b.append(img)
        if (i + 1) % 10 == 0:
            print(f"  Group B: {i+1}/{args.n_images}")

    del pipe
    torch.cuda.empty_cache()

    # CLIP encode
    print("Computing CLIP embeddings...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess, tokenizer = load_clip_model(device)

    emb_a = clip_encode_images(images_a, clip_model, preprocess, device)  # (N, D)
    emb_b = clip_encode_images(images_b, clip_model, preprocess, device)  # (N, D)

    mean_a = emb_a.mean(axis=0)
    mean_b = emb_b.mean(axis=0)
    direction = mean_b - mean_a
    direction = direction / np.linalg.norm(direction)

    # Save
    np.savez(f"{args.output_dir}/direction_data.npz",
             emb_a=emb_a, emb_b=emb_b,
             mean_a=mean_a, mean_b=mean_b,
             direction=direction)

    # Report
    proj_a = emb_a @ direction
    proj_b = emb_b @ direction
    print(f"\nDirection computed:")
    print(f"  Group A mean projection: {proj_a.mean():.4f} ± {proj_a.std():.4f}")
    print(f"  Group B mean projection: {proj_b.mean():.4f} ± {proj_b.std():.4f}")
    print(f"  Separation (Cohen's d): {(proj_b.mean() - proj_a.mean()) / np.sqrt((proj_a.std()**2 + proj_b.std()**2)/2):.2f}")

    config = {
        "prompt_a": args.prompt_a,
        "prompt_b": args.prompt_b,
        "n_images": args.n_images,
    }
    with open(f"{args.output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
