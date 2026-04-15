"""Decode 200 images sorted by DA1 and PC1 projections, save grids."""
import os
import argparse
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL


def decode_latents(vae, x0_flat, latent_shape, order, batch_size=1, device="cuda"):
    """Decode latents in given order, return PIL images."""
    vae.enable_tiling()
    vae.enable_slicing()
    images = []
    x0_sorted = x0_flat[order]
    for i in range(0, len(x0_sorted), batch_size):
        batch = torch.from_numpy(x0_sorted[i:i+batch_size]).float().to(device)
        batch = batch.reshape(-1, *latent_shape)
        with torch.no_grad():
            pix = vae.decode(batch / vae.config.scaling_factor).sample
        pix = (pix.clamp(-1, 1) + 1) / 2
        for j in range(pix.shape[0]):
            img = pix[j].permute(1, 2, 0).cpu().numpy()
            images.append(Image.fromarray((img * 255).astype(np.uint8)))
    return images


def make_grid(images, n_cols=20, thumb_size=256):
    """Tile images into a grid."""
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    thumbs = [im.resize((thumb_size, thumb_size), Image.LANCZOS) for im in images]
    grid = Image.new("RGB", (n_cols * thumb_size, n_rows * thumb_size))
    for i, im in enumerate(thumbs):
        r, c = divmod(i, n_cols)
        grid.paste(im, (c * thumb_size, r * thumb_size))
    return grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="results_da_pcavda")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()

    configs = [
        {
            "label": "elderly",
            "x0_path": "results_multiprompt_sweep/gen_mp_p5_elderly_gs7p5_quick_x0.npz",
            "clip_path": "results_multiprompt_sweep/gen_mp_p5_elderly_gs7p5_quick_clip.npz",
            "da_path": "results_da_pcavda/clip_da_elderly_gs7p5.npz",
        },
        {
            "label": "person",
            "x0_path": "results_guidance_sweep/gen_sweep_gs7p5_quick_x0.npz",
            "clip_path": "results_guidance_sweep/gen_sweep_gs7p5_quick_clip.npz",
            "da_path": "results_da_pcavda/clip_da_person_gs7p5.npz",
        },
    ]

    latent_shape = (4, 128, 128)

    for cfg in configs:
        label = cfg["label"]
        print(f"\n=== {label} ===")

        # Load data
        x0 = np.load(cfg["x0_path"])["x0"]  # (200, 65536)
        z0 = np.load(cfg["clip_path"])["z0"]  # (200, 768)
        da_data = np.load(cfg["da_path"])
        da1 = da_data["da1_eigvec"]  # (768,)

        # PC1 in CLIP space
        z_centered = z0 - z0.mean(0)
        cov = z_centered.T @ z_centered / len(z_centered)
        evals, evecs = np.linalg.eigh(cov)
        pc1 = evecs[:, -1]

        # Project
        da1_proj = z_centered @ da1
        pc1_proj = z_centered @ pc1

        # Sort orders
        da1_order = np.argsort(da1_proj)
        pc1_order = np.argsort(pc1_proj)

        cos_da1_pc1 = abs(np.dot(da1, pc1))
        print(f"  |cos(DA1, PC1)| = {cos_da1_pc1:.4f}")
        print(f"  DA1 proj range: [{da1_proj.min():.3f}, {da1_proj.max():.3f}]")
        print(f"  PC1 proj range: [{pc1_proj.min():.3f}, {pc1_proj.max():.3f}]")

        # Decode and make grids
        for sort_name, order in [("da1", da1_order), ("pc1", pc1_order)]:
            print(f"  Decoding {sort_name}-sorted grid...")
            images = decode_latents(vae, x0, latent_shape, order,
                                    batch_size=1, device=device)
            grid = make_grid(images, n_cols=20, thumb_size=256)
            out = os.path.join(args.output_dir, f"grid_{label}_{sort_name}_sorted.png")
            grid.save(out)
            print(f"  Saved {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
