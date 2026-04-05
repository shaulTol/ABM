#!/usr/bin/env python3
"""
Generate image pairs for computing grounded CLIP directions.

For each attribute (glasses, brightness, smile), generate 50 pairs:
  - Base image: person generated with text2img (ControlNet or plain)
  - Edited image: img2img of base with attribute-specific prompt

The CLIP direction is then mean(edit_emb - base_emb) over 50 pairs,
giving a visual-grounded direction free of CLIP text bias.
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
    load_clip_model, clip_encode_images, clip_text_direction,
    make_scribble_face_outline, TEXT_PAIRS,
)


# ── Attribute definitions ──────────────────────────────────────────────

ATTRIBUTES = {
    "glasses": {
        "edit_prompt": "a person wearing glasses",
    },
    "brightness": {
        "edit_prompt": "a bright, well-lit photograph of a person",
    },
    "smile": {
        "edit_prompt": "a person with a big smile",
    },
}

BASE_PROMPTS = [
    "a person",
    "a street portrait",
    "a portrait photograph",
    "a photograph of someone",
    "a close-up of a face",
]


# ── Scribble generators ───────────────────────────────────────────────

def make_face_boxed(size=512, lw=3, blur=5):
    """Face outline + bounding rectangle, then blur."""
    img = Image.new("RGB", (size, size), "black")
    d = ImageDraw.Draw(img)
    d.ellipse([140, 100, 372, 410], outline="white", width=lw)
    d.ellipse([185, 205, 225, 228], outline="white", width=lw)
    d.ellipse([287, 205, 327, 228], outline="white", width=lw)
    d.line([256, 240, 248, 295], fill="white", width=lw)
    d.line([248, 295, 264, 295], fill="white", width=lw)
    d.arc([218, 315, 294, 355], 0, 180, fill="white", width=lw)
    d.rectangle([140, 100, 372, 410], outline="white", width=lw)
    return img.filter(ImageFilter.GaussianBlur(blur))


def make_wider_face(size=512, lw=3, blur=5):
    """Wider face oval."""
    img = Image.new("RGB", (size, size), "black")
    d = ImageDraw.Draw(img)
    d.ellipse([110, 60, 402, 430], outline="white", width=lw)
    d.ellipse([170, 180, 215, 208], outline="white", width=lw)
    d.ellipse([297, 180, 342, 208], outline="white", width=lw)
    d.line([256, 225, 245, 290], fill="white", width=lw)
    d.line([245, 290, 267, 290], fill="white", width=lw)
    d.arc([205, 310, 307, 360], 0, 180, fill="white", width=lw)
    return img.filter(ImageFilter.GaussianBlur(blur))


def load_scribbles():
    """Load/generate the 4 scribble conditioning images."""
    scribbles = {}

    # face_blurred: programmatic
    scribbles["face_blurred"] = make_scribble_face_outline().filter(
        ImageFilter.GaussianBlur(5))

    # face_boxed: programmatic
    scribbles["face_boxed"] = make_face_boxed()

    # dps_blurred: from file
    dps_path = os.path.join(os.path.dirname(__file__), "dps_scribble_blurred.png")
    if os.path.exists(dps_path):
        scribbles["dps_blurred"] = Image.open(dps_path).convert("RGB")
    else:
        print(f"Warning: {dps_path} not found, skipping dps_blurred")

    # wider_face: programmatic
    scribbles["wider_face"] = make_wider_face()

    return scribbles


# ── Pair generation configs ───────────────────────────────────────────

def build_pair_configs(n_pairs=50):
    """Build configs for 50 pairs with diverse scribbles and prompts.

    ~25 use ControlNet (scribble + prompt), ~25 use plain text2img (no scribble).
    """
    scribble_names = ["face_blurred", "face_boxed", "dps_blurred", "wider_face"]
    configs = []

    for i in range(n_pairs):
        base_prompt = BASE_PROMPTS[i % len(BASE_PROMPTS)]
        seed = i

        if i < 25:
            # ControlNet pairs
            scribble_name = scribble_names[i % len(scribble_names)]
            configs.append({
                "idx": i,
                "seed": seed,
                "base_prompt": base_prompt,
                "scribble": scribble_name,
                "use_controlnet": True,
            })
        else:
            # Plain text2img pairs (no scribble)
            configs.append({
                "idx": i,
                "seed": seed,
                "base_prompt": base_prompt,
                "scribble": None,
                "use_controlnet": False,
            })

    return configs


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.3,
                        help="img2img strength for edit step")
    parser.add_argument("--output_dir", type=str, default="results_image_pairs")
    parser.add_argument("--controlnet_scale", type=float, default=0.8)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--attributes", nargs="+",
                        default=list(ATTRIBUTES.keys()),
                        help="Which attributes to generate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load scribbles ──
    scribbles = load_scribbles()
    scrib_dir = os.path.join(args.output_dir, "scribbles")
    os.makedirs(scrib_dir, exist_ok=True)
    for name, img in scribbles.items():
        img.save(os.path.join(scrib_dir, f"{name}.png"))
    print(f"Loaded {len(scribbles)} scribbles")

    # ── Load ControlNet text2img pipeline ──
    print("Loading SDXL Turbo + ControlNet pipeline...")
    from diffusers import (
        ControlNetModel,
        StableDiffusionXLControlNetPipeline,
        AutoPipelineForImage2Image,
        AutoPipelineForText2Image,
    )

    controlnet = ControlNetModel.from_pretrained(
        "xinsir/controlnet-scribble-sdxl-1.0",
        torch_dtype=torch.float16, use_safetensors=True,
    )
    cn_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        controlnet=controlnet,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    ).to(device)
    cn_pipe.set_progress_bar_config(disable=True)

    # ── Load plain text2img pipeline (no ControlNet) ──
    print("Loading plain text2img pipeline...")
    t2i_pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    ).to(device)
    t2i_pipe.set_progress_bar_config(disable=True)

    # ── Load img2img pipeline ──
    print("Loading img2img pipeline...")
    i2i_pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16, variant="fp16",
    ).to(device)
    i2i_pipe.set_progress_bar_config(disable=True)

    # ── Build pair configs ──
    pair_configs = build_pair_configs(args.n_pairs)

    # ── Save full config ──
    full_config = {
        "n_pairs": args.n_pairs,
        "strength": args.strength,
        "controlnet_scale": args.controlnet_scale,
        "num_steps": args.num_steps,
        "attributes": args.attributes,
        "pair_configs": pair_configs,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(full_config, f, indent=2)

    # ── Generate pairs for each attribute ──
    for attr_name in args.attributes:
        attr = ATTRIBUTES[attr_name]
        attr_dir = os.path.join(args.output_dir, attr_name)
        os.makedirs(attr_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Attribute: {attr_name} — edit prompt: '{attr['edit_prompt']}'")
        print(f"{'='*60}")

        for cfg in pair_configs:
            idx = cfg["idx"]
            seed = cfg["seed"]
            base_prompt = cfg["base_prompt"]

            # Step 1: Generate base image
            generator = torch.Generator(device=device).manual_seed(seed)
            if cfg["use_controlnet"] and cfg["scribble"] in scribbles:
                scribble_img = scribbles[cfg["scribble"]]
                base_result = cn_pipe(
                    prompt=base_prompt,
                    image=scribble_img,
                    controlnet_conditioning_scale=args.controlnet_scale,
                    guidance_scale=0.0,
                    num_inference_steps=args.num_steps,
                    height=512, width=512,
                    generator=generator,
                )
            else:
                base_result = t2i_pipe(
                    prompt=base_prompt,
                    guidance_scale=0.0,
                    num_inference_steps=args.num_steps,
                    height=512, width=512,
                    generator=generator,
                )
            base_img = base_result.images[0]

            # Step 2: Edit base image with attribute prompt via img2img
            edit_generator = torch.Generator(device=device).manual_seed(seed + 1000)
            edit_result = i2i_pipe(
                prompt=attr["edit_prompt"],
                image=base_img,
                strength=args.strength,
                num_inference_steps=args.num_steps,
                guidance_scale=0.0,
                generator=edit_generator,
            )
            edit_img = edit_result.images[0]

            # Save pair
            base_img.save(os.path.join(attr_dir, f"pair_{idx:03d}_base.png"))
            edit_img.save(os.path.join(attr_dir, f"pair_{idx:03d}_edit.png"))

            if (idx + 1) % 10 == 0:
                print(f"  [{attr_name}] Generated {idx+1}/{args.n_pairs} pairs",
                      flush=True)

        # Make comparison grid (4x5 grid of first 20 pairs, side by side)
        print(f"  Creating comparison grid for {attr_name}...")
        grid_rows, grid_cols = 4, 5
        n_grid = min(grid_rows * grid_cols, args.n_pairs)
        cell_w, cell_h = 512 * 2, 512  # base + edit side by side
        grid = Image.new("RGB", (cell_w * grid_cols, cell_h * grid_rows))
        for gi in range(n_grid):
            bp = os.path.join(attr_dir, f"pair_{gi:03d}_base.png")
            ep = os.path.join(attr_dir, f"pair_{gi:03d}_edit.png")
            if os.path.exists(bp) and os.path.exists(ep):
                b = Image.open(bp)
                e = Image.open(ep)
                row, col = gi // grid_cols, gi % grid_cols
                grid.paste(b, (col * cell_w, row * cell_h))
                grid.paste(e, (col * cell_w + 512, row * cell_h))
        grid.save(os.path.join(attr_dir, "grid_comparison.png"))
        print(f"  Saved grid to {attr_dir}/grid_comparison.png")

    # ── Compute CLIP directions ──
    print("\n" + "="*60)
    print("Computing CLIP directions from image pairs...")
    print("="*60)

    clip_model, tokenizer, preprocess = load_clip_model(device)

    directions = {}
    for attr_name in args.attributes:
        attr_dir = os.path.join(args.output_dir, attr_name)
        base_images = []
        edit_images = []
        for i in range(args.n_pairs):
            bp = os.path.join(attr_dir, f"pair_{i:03d}_base.png")
            ep = os.path.join(attr_dir, f"pair_{i:03d}_edit.png")
            if os.path.exists(bp) and os.path.exists(ep):
                base_images.append(Image.open(bp).convert("RGB"))
                edit_images.append(Image.open(ep).convert("RGB"))

        base_embs = clip_encode_images(clip_model, preprocess, base_images, device)
        edit_embs = clip_encode_images(clip_model, preprocess, edit_images, device)

        diffs = edit_embs - base_embs
        direction = diffs.mean(axis=0)
        direction = direction / np.linalg.norm(direction)
        directions[attr_name] = direction

        print(f"  {attr_name}: computed from {len(base_images)} pairs, "
              f"mean diff norm = {np.linalg.norm(diffs, axis=1).mean():.4f}")

    # ── Compare with text directions ──
    print("\nComparing image-pair directions with text directions...")
    text_dirs = {}
    for text_a, text_b, name in TEXT_PAIRS:
        d = clip_text_direction(clip_model, tokenizer, text_a, text_b, device)
        text_dirs[name] = d

    comparison = {}
    for attr_name, img_dir_vec in directions.items():
        comparison[attr_name] = {}
        for text_name, text_dir_vec in text_dirs.items():
            cos = float(np.dot(img_dir_vec, text_dir_vec))
            comparison[attr_name][text_name] = cos
        # Print top matches
        sorted_matches = sorted(comparison[attr_name].items(),
                                key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  {attr_name} image direction — top text correlations:")
        for tname, cos_val in sorted_matches[:5]:
            print(f"    {tname}: cos={cos_val:.3f}")

    # ── Save results ──
    np.savez(os.path.join(args.output_dir, "clip_directions.npz"),
             **{f"{k}_direction": v for k, v in directions.items()})

    results = {
        "n_pairs": args.n_pairs,
        "strength": args.strength,
        "attributes": args.attributes,
        "comparison_with_text_dirs": comparison,
    }
    with open(os.path.join(args.output_dir, "direction_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")
    print(json.dumps(results, indent=2))
    print("\nDone!")


if __name__ == "__main__":
    main()
