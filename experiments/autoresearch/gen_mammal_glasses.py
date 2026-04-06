#!/usr/bin/env python3
"""
Generate a distribution of mammals from a SINGLE prompt + scribble.

Prompt: broad mammal prompt (high variance → diverse animals)
Scribble: animal shape with two circles suggesting glasses on the face.
The model sometimes interprets the circles as glasses, sometimes ignores them.

Goal: PC1 = mammal type, DA1 = glasses vs no glasses.
Runs on both Turbo and Base for comparison.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from search_parallel_ellipses import (
    load_clip_model, clip_encode_images, clip_text_direction, TEXT_PAIRS,
)


def make_animal_glasses_scribble(size=512, lw=4, blur=5):
    """Just glasses shape, small, in upper-center. Room for animal body below."""
    img = Image.new("RGB", (size, size), "black")
    d = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 3
    r = 25
    gap = 8

    # Left lens
    d.ellipse([cx - gap - 2*r, cy - r, cx - gap, cy + r], outline="white", width=lw)
    # Right lens
    d.ellipse([cx + gap, cy - r, cx + gap + 2*r, cy + r], outline="white", width=lw)
    # Bridge
    d.arc([cx - gap, cy - 10, cx + gap, cy + 10], 180, 0, fill="white", width=lw)
    # Arms (temples)
    d.line([cx - gap - 2*r, cy, cx - gap - 2*r - 30, cy - 10], fill="white", width=lw)
    d.line([cx + gap + 2*r, cy, cx + gap + 2*r + 30, cy - 10], fill="white", width=lw)

    return img.filter(ImageFilter.GaussianBlur(blur))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="results_mammal_glasses")
    parser.add_argument("--controlnet_scale", type=float, default=0.5)
    parser.add_argument("--prompt", type=str,
                        default="a wild animal, nature photography, diverse species")
    parser.add_argument("--blur", type=float, default=5.0,
                        help="Gaussian blur radius for scribble")
    parser.add_argument("--no_scribble", action="store_true",
                        help="Skip ControlNet, use plain text2img")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="Override guidance scale for Base model")
    parser.add_argument("--base_only", action="store_true",
                        help="Only run Base model, skip Turbo")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Prompt: '{args.prompt}'")
    print(f"ControlNet scale: {args.controlnet_scale}")
    print(f"N images: {args.n_images}")

    # Scribble
    scribble = make_animal_glasses_scribble(blur=args.blur)
    scribble.save(os.path.join(args.output_dir, "scribble.png"))

    # Save config
    config = {
        "prompt": args.prompt,
        "controlnet_scale": args.controlnet_scale,
        "n_images": args.n_images,
        "blur": args.blur,
        "method": "single_prompt_single_scribble",
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Load pipelines ──
    from diffusers import (
        ControlNetModel,
        StableDiffusionXLControlNetPipeline,
        AutoPipelineForText2Image,
    )

    if not args.no_scribble:
        controlnet = ControlNetModel.from_pretrained(
            "xinsir/controlnet-scribble-sdxl-1.0",
            torch_dtype=torch.float16, use_safetensors=True,
        )
    else:
        controlnet = None

    models = {
        "turbo": {
            "model_id": "stabilityai/sdxl-turbo",
            "num_steps": 4,
            "guidance_scale": 0.0,
            "variant": "fp16",
            "resolution": 512,
        },
        "base": {
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "num_steps": 50,
            "guidance_scale": 7.5,
            "variant": "fp16",
            "resolution": 1024,
        },
    }

    # Apply overrides
    if args.guidance_scale is not None:
        models["base"]["guidance_scale"] = args.guidance_scale
    if args.base_only:
        models = {k: v for k, v in models.items() if k == "base"}

    all_images = {}

    for model_name, model_cfg in models.items():
        print(f"\n{'='*60}")
        print(f"Generating {args.n_images} images with {model_name.upper()}")
        print(f"  steps={model_cfg['num_steps']}, guidance={model_cfg['guidance_scale']}")
        print(f"{'='*60}")

        if not args.no_scribble:
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                model_cfg["model_id"],
                controlnet=controlnet,
                torch_dtype=torch.float16,
                variant=model_cfg["variant"],
                use_safetensors=True,
            ).to(device)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_cfg["model_id"],
                torch_dtype=torch.float16,
                variant=model_cfg["variant"],
                use_safetensors=True,
            ).to(device)
        pipe.set_progress_bar_config(disable=True)

        model_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        res = model_cfg["resolution"]
        if not args.no_scribble:
            scribble_resized = scribble.resize((res, res), Image.LANCZOS)

        images = []
        for i in range(args.n_images):
            generator = torch.Generator(device=device).manual_seed(i)
            gen_kwargs = dict(
                prompt=args.prompt,
                guidance_scale=model_cfg["guidance_scale"],
                num_inference_steps=model_cfg["num_steps"],
                height=res, width=res,
                generator=generator,
            )
            if not args.no_scribble:
                gen_kwargs["image"] = scribble_resized
                gen_kwargs["controlnet_conditioning_scale"] = args.controlnet_scale
            if res == 1024:
                gen_kwargs["original_size"] = (1024, 1024)
                gen_kwargs["target_size"] = (1024, 1024)
            result = pipe(**gen_kwargs)
            img = result.images[0]
            images.append(img)
            img.save(os.path.join(model_dir, f"img_{i:03d}.png"))

            if (i + 1) % 20 == 0:
                print(f"  [{model_name}] {i+1}/{args.n_images}", flush=True)

        all_images[model_name] = images

        # Grid: 10x20
        print(f"  Creating grid...")
        cols = 20
        rows = min(10, (len(images) + cols - 1) // cols)
        thumb = 128
        grid = Image.new("RGB", (thumb * cols, thumb * rows))
        for idx, img in enumerate(images[:rows * cols]):
            r, c = idx // cols, idx % cols
            grid.paste(img.resize((thumb, thumb)), (c * thumb, r * thumb))
        grid.save(os.path.join(model_dir, "grid.png"))

        del pipe
        torch.cuda.empty_cache()

    # ── CLIP analysis ──
    print(f"\n{'='*60}")
    print("CLIP analysis")
    print(f"{'='*60}")

    clip_model, tokenizer, preprocess = load_clip_model(device)

    glasses_dir = clip_text_direction(
        clip_model, tokenizer,
        "an animal", "an animal wearing sunglasses", device
    )

    # Also get some text directions for interpretation
    text_dirs = {}
    for text_a, text_b, name in TEXT_PAIRS:
        text_dirs[name] = clip_text_direction(clip_model, tokenizer, text_a, text_b, device)
    text_dirs["glasses_animal"] = glasses_dir

    for model_name, images in all_images.items():
        print(f"\n  {model_name.upper()}:")
        embs = clip_encode_images(clip_model, preprocess, images, device)

        # PCA
        centered = embs - embs.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / len(centered)
        eigvals, eigvecs = np.linalg.eigh(cov)

        pc1 = eigvecs[:, -1]
        pc2 = eigvecs[:, -2]
        pc1_proj = embs @ pc1
        pc2_proj = embs @ pc2
        glasses_proj = embs @ glasses_dir

        var1 = eigvals[-1] / eigvals.sum()
        var2 = eigvals[-2] / eigvals.sum()

        # Correlations of PCs with text directions
        pc1_corrs = {}
        pc2_corrs = {}
        for name, td in text_dirs.items():
            proj = embs @ td
            pc1_corrs[name] = float(np.corrcoef(pc1_proj, proj)[0, 1])
            pc2_corrs[name] = float(np.corrcoef(pc2_proj, proj)[0, 1])

        # Print top correlations
        print(f"    PC1 var: {var1:.3f}, PC2 var: {var2:.3f}")
        sorted_pc1 = sorted(pc1_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"    PC1 top correlations:")
        for name, corr in sorted_pc1[:5]:
            print(f"      {name}: {corr:.3f}")
        sorted_pc2 = sorted(pc2_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"    PC2 top correlations:")
        for name, corr in sorted_pc2[:5]:
            print(f"      {name}: {corr:.3f}")

        model_dir = os.path.join(args.output_dir, model_name)
        np.savez(os.path.join(model_dir, "clip_data.npz"),
                 embeddings=embs, pc1=pc1, pc2=pc2,
                 pc1_proj=pc1_proj, pc2_proj=pc2_proj,
                 glasses_proj=glasses_proj, glasses_dir=glasses_dir,
                 eigvals=eigvals[-10:])

        results = {
            "model": model_name,
            "prompt": args.prompt,
            "controlnet_scale": args.controlnet_scale,
            "n_images": len(images),
            "pc1_var_ratio": float(var1),
            "pc2_var_ratio": float(var2),
            "pc1_text_corrs": pc1_corrs,
            "pc2_text_corrs": pc2_corrs,
        }
        with open(os.path.join(model_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
