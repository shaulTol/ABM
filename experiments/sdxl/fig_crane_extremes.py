"""Decode highest/lowest PC1 and DA1 crane images from x0 latents.

Loads CLIP embeddings, projects onto PC1 and DA1, picks top/bottom 5,
decodes x0 latents via SDXL VAE, saves image grids.
"""
import numpy as np, torch, os, sys
from PIL import Image
from diffusers import AutoencoderKL

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

K = 5  # top/bottom K images per direction


def decode_latents(vae, latents, latent_shape=(4, 128, 128), batch_size=1):
    """Decode (N, C*H*W) flat latents to PIL images. Uses float32 + tiling."""
    vae.enable_tiling()
    vae.enable_slicing()
    images = []
    for i in range(0, len(latents), batch_size):
        batch = torch.from_numpy(latents[i:i+batch_size]).float().cuda()
        batch = batch.reshape(-1, *latent_shape)
        with torch.no_grad():
            pix = vae.decode(batch / vae.config.scaling_factor).sample
        pix = (pix.clamp(-1, 1) + 1) / 2
        for j in range(pix.shape[0]):
            img = pix[j].permute(1, 2, 0).cpu().numpy()
            images.append(Image.fromarray((img * 255).astype(np.uint8)))
    return images


def make_grid(images, ncols=5, padding=4):
    """Arrange PIL images into a grid."""
    w, h = images[0].size
    nrows = (len(images) + ncols - 1) // ncols
    grid = Image.new('RGB', (ncols * w + (ncols-1)*padding, nrows * h + (nrows-1)*padding), (255,255,255))
    for i, img in enumerate(images):
        r, c = divmod(i, ncols)
        grid.paste(img, (c * (w + padding), r * (h + padding)))
    return grid


def main():
    # Load CLIP embeddings
    clip_path = os.path.join(SCRIPT_DIR, "z_clip_traj_crane_gs7p5_dt0p0025.npz")
    clip_data = np.load(clip_path)
    z0_clip = clip_data['z0_clip']  # (N, 768)
    N, D = z0_clip.shape
    print(f"z0_clip: {z0_clip.shape}")

    # Load DA eigenvectors
    eig_path = os.path.join(SCRIPT_DIR, "da_eigvecs_crane_gs7p5_dt0p0025.npz")
    eig_data = np.load(eig_path)
    da1 = eig_data['eigvecs'][:, 0]  # (768,)
    pc1 = eig_data['pc1']  # (768,)
    print(f"DA1 shape: {da1.shape}, PC1 shape: {pc1.shape}")

    # Project onto PC1 and DA1
    mu = z0_clip.mean(axis=0)
    z_c = z0_clip - mu
    proj_pc1 = z_c @ pc1  # (N,)
    proj_da1 = z_c @ da1  # (N,)

    # Get top/bottom K indices, skipping CLIP duplicates
    def pick_unique(proj, embeddings, top=True, k=K):
        """Pick k indices with highest/lowest proj, skipping duplicate embeddings."""
        order = np.argsort(proj)[::-1].copy() if top else np.argsort(proj).copy()
        seen_hashes = set()
        result = []
        for idx in order:
            h = hash(embeddings[idx].tobytes())
            if h not in seen_hashes:
                seen_hashes.add(h)
                result.append(idx)
                if len(result) == k:
                    break
        return np.array(result)

    pc1_top = pick_unique(proj_pc1, z0_clip, top=True)
    pc1_bot = pick_unique(proj_pc1, z0_clip, top=False)
    da1_top = pick_unique(proj_da1, z0_clip, top=True)
    # Replace rank 4 (idx 40) with rank 6 (idx 8) per user request
    da1_top = np.array([da1_top[0], da1_top[1], da1_top[2], 8, da1_top[4]])
    da1_bot = pick_unique(proj_da1, z0_clip, top=False)

    print(f"PC1 top indices: {pc1_top}, proj values: {proj_pc1[pc1_top]}")
    print(f"PC1 bot indices: {pc1_bot}, proj values: {proj_pc1[pc1_bot]}")
    print(f"DA1 top indices: {da1_top}, proj values: {proj_da1[da1_top]}")
    print(f"DA1 bot indices: {da1_bot}, proj values: {proj_da1[da1_bot]}")

    # Load x0 latents from Langevin checkpoint (matches CLIP embedding indices)
    # Note: x0_flat has 125/200 unique due to save bug, but indices match z0_clip
    ckpt_path = os.path.join(SCRIPT_DIR, "langevin_ckpt_crane_gs7p5_dt0p0025.npz")
    ckpt = np.load(ckpt_path)
    x0_flat = ckpt['x0_flat'].astype(np.float32)  # (N, 65536)
    print(f"x0_flat shape: {x0_flat.shape}")

    # Load VAE (float32 — fp16 produces all-black images for 1024x1024)
    print("Loading SDXL VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae",
        torch_dtype=torch.float32
    ).cuda()

    # Decode and save grids
    all_idx = np.concatenate([pc1_top, pc1_bot, da1_top, da1_bot])
    unique_idx = np.unique(all_idx)
    print(f"Decoding {len(unique_idx)} unique images...")
    decoded = {}
    imgs = decode_latents(vae, x0_flat[unique_idx])
    for i, idx in enumerate(unique_idx):
        decoded[idx] = imgs[i]

    # Make grids: top row = highest, bottom row = lowest
    # PC1 grid
    pc1_imgs = [decoded[i] for i in pc1_top] + [decoded[i] for i in pc1_bot]
    pc1_grid = make_grid(pc1_imgs, ncols=K)
    pc1_grid.save(os.path.join(SCRIPT_DIR, "crane_pc1_extremes.png"))
    print(f"Saved crane_pc1_extremes.png (top row=high PC1, bottom row=low PC1)")

    # DA1 grid
    da1_imgs = [decoded[i] for i in da1_top] + [decoded[i] for i in da1_bot]
    da1_grid = make_grid(da1_imgs, ncols=K)
    da1_grid.save(os.path.join(SCRIPT_DIR, "crane_da1_extremes.png"))
    print(f"Saved crane_da1_extremes.png (top row=high DA1, bottom row=low DA1)")

    # Combined 4-row grid for easy comparison
    all_imgs = [decoded[i] for i in pc1_top] + [decoded[i] for i in pc1_bot] + \
               [decoded[i] for i in da1_top] + [decoded[i] for i in da1_bot]
    combined = make_grid(all_imgs, ncols=K)
    combined.save(os.path.join(SCRIPT_DIR, "crane_pc1_da1_extremes.png"))
    print(f"Saved crane_pc1_da1_extremes.png (rows: PC1 high, PC1 low, DA1 high, DA1 low)")


if __name__ == "__main__":
    main()
