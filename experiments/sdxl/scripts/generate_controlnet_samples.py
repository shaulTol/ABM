"""Generate a grid of images with SDXL + ControlNet conditioning."""
import os
import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--control_image", type=str, required=True)
    p.add_argument("--prompt", type=str, default="an image of a")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--controlnet_model_id", type=str, default="xinsir/controlnet-scribble-sdxl-1.0")
    p.add_argument("--controlnet_scale", type=float, default=0.8)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--output_dir", type=str, default="output_controlnet")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")

    print("Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model_id, torch_dtype=torch.float16, use_safetensors=True)

    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    ).to(device)

    control_img = Image.open(args.control_image).convert("RGB").resize((512, 512))

    print(f"Generating {args.n} images...")
    for i in range(args.n):
        generator = torch.Generator(device=device).manual_seed(args.seed + i)
        result = pipe(
            prompt=args.prompt,
            image=control_img,
            controlnet_conditioning_scale=args.controlnet_scale,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            height=512, width=512,
            generator=generator,
        )
        result.images[0].save(os.path.join(args.output_dir, f"sample_{i:03d}.png"))
        print(f"  {i+1}/{args.n}", flush=True)

    # Save grid
    from torchvision.utils import make_grid
    import torchvision.transforms as T
    imgs = [T.ToTensor()(Image.open(os.path.join(args.output_dir, f"sample_{i:03d}.png")))
            for i in range(args.n)]
    ncol = 5
    grid = make_grid(imgs, nrow=ncol, padding=4)
    T.ToPILImage()(grid).save(os.path.join(args.output_dir, "grid.png"))
    print(f"Saved grid to {args.output_dir}/grid.png")


if __name__ == "__main__":
    main()
