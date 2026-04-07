#!/usr/bin/env python3
"""Quick generation script for testing prompts."""
import os, sys, argparse, torch
from PIL import Image, ImageDraw, ImageFilter


def make_frame_scribble(size=1024, margin=80, lw=6, blur=8):
    """Faded rectangular frame scribble on black background."""
    img = Image.new("RGB", (size, size), "black")
    d = ImageDraw.Draw(img)
    d.rectangle(
        [margin, margin, size - margin, size - margin],
        outline="white", width=lw,
    )
    return img.filter(ImageFilter.GaussianBlur(blur))


def make_frame_scribble_small(size=1024, margin=320, lw=6, blur=8):
    """Small centered frame scribble."""
    return make_frame_scribble(size=size, margin=margin, lw=lw, blur=blur)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--n_images", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="results_quick")
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--scribble", type=str, default=None,
                        choices=["frame", "frame_small"],
                        help="Use ControlNet with a scribble shape")
    parser.add_argument("--controlnet_scale", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.scribble:
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        controlnet = ControlNetModel.from_pretrained(
            "xinsir/controlnet-scribble-sdxl-1.0",
            torch_dtype=torch.float16,
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16, variant="fp16",
        ).to("cuda")

        if args.scribble == "frame":
            scribble_img = make_frame_scribble()
        elif args.scribble == "frame_small":
            scribble_img = make_frame_scribble_small()
        scribble_img.save(f"{args.output_dir}/scribble.png")
        print(f"Using ControlNet scribble: {args.scribble}, scale={args.controlnet_scale}")
    else:
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16, variant="fp16",
        ).to("cuda")
        scribble_img = None

    images = []
    for i in range(args.n_images):
        gen_kwargs = dict(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            original_size=(1024, 1024), target_size=(1024, 1024),
            generator=torch.Generator("cuda").manual_seed(i),
        )
        if scribble_img is not None:
            gen_kwargs["image"] = scribble_img
            gen_kwargs["controlnet_conditioning_scale"] = args.controlnet_scale

        img = pipe(**gen_kwargs).images[0]
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
