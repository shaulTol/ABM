"""Generate quick samples from SDXL Turbo and/or Base."""
import argparse, os, torch
from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--mode", choices=["turbo", "base", "both"], default="both")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode in ("turbo", "both"):
        print("=== SDXL Turbo ===", flush=True)
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16).to(device)
        out = os.path.join(args.output_dir, "turbo")
        os.makedirs(out, exist_ok=True)
        for i in range(args.n):
            gen = torch.Generator(device).manual_seed(args.seed + i)
            img = pipe(args.prompt, num_inference_steps=4, guidance_scale=0.0,
                       generator=gen).images[0]
            img.save(os.path.join(out, f"sample_{i:03d}.png"))
            print(f"  turbo {i+1}/{args.n}", flush=True)
        del pipe
        torch.cuda.empty_cache()

    if args.mode in ("base", "both"):
        print("=== SDXL Base ===", flush=True)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16).to(device)
        out = os.path.join(args.output_dir, "base")
        os.makedirs(out, exist_ok=True)
        for i in range(args.n):
            gen = torch.Generator(device).manual_seed(args.seed + i)
            img = pipe(args.prompt, num_inference_steps=50, guidance_scale=7.5,
                       original_size=(1024, 1024), target_size=(1024, 1024),
                       generator=gen).images[0]
            img.save(os.path.join(out, f"sample_{i:03d}.png"))
            print(f"  base {i+1}/{args.n}", flush=True)
        del pipe
        torch.cuda.empty_cache()

    print("Done!")

if __name__ == "__main__":
    main()
