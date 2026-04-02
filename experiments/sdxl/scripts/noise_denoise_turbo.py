"""Take an image, noise it halfway, denoise 20 times with SDXL Turbo, save results."""
import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForImage2Image
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--prompt", type=str, default="a superrealistic professional photograph of an androgynous person")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--strength", type=float, default=0.5, help="Noise strength (0=no noise, 1=full noise)")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="output_noise_denoise")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    init_image = Image.open(args.image).convert("RGB").resize((512, 512))
    init_image.save(os.path.join(args.output_dir, "init_image.png"))

    all_images = []
    for i in range(args.n):
        generator = torch.Generator(device="cuda").manual_seed(args.seed + i)
        result = pipe(
            prompt=args.prompt,
            image=init_image,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            generator=generator,
        )
        img = result.images[0]
        img.save(os.path.join(args.output_dir, f"sample_{i:03d}.png"))
        all_images.append(img)
        print(f"Generated {i+1}/{args.n}", flush=True)

    # Make grid
    n_cols = 5
    n_rows = (args.n + n_cols - 1) // n_cols
    w, h = all_images[0].size
    grid = Image.new("RGB", (n_cols * w, n_rows * h))
    for i, img in enumerate(all_images):
        grid.paste(img, ((i % n_cols) * w, (i // n_cols) * h))
    grid.save(os.path.join(args.output_dir, "grid.png"))
    print(f"Saved grid to {args.output_dir}/grid.png")


if __name__ == "__main__":
    main()
