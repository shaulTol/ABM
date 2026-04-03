"""Diagnostics for Latent-CLIP vs CLIP validation."""
import argparse
import numpy as np
import torch
import open_clip
import latent_clip
from huggingface_hub import hf_hub_download
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
from scipy.stats import spearmanr
import os


TEXT_PROBES = [
    "a photograph of a man",
    "a photograph of a woman",
    "a photograph of an old person",
    "a photograph of a young person",
    "a photograph of a person with dark skin",
    "a photograph of a person with light skin",
    "a photograph of a smiling person",
    "a photograph of a serious person",
    "a photograph of a person facing left",
    "a photograph of a person facing right",
    "a close up portrait",
    "a wide shot photograph",
    "a photograph with bright lighting",
    "a photograph with dark lighting",
    "a colorful photograph",
    "a black and white photograph",
    "a photograph of a person with glasses",
    "a photograph of a person with a beard",
    "a photograph of a person with long hair",
    "a photograph of a person with short hair",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_npz", type=str, required=True)
    p.add_argument("--save_debug_dir", type=str, default="debug_decode")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === 1. Print all text probes ===
    print("=" * 60)
    print("1. TEXT PROBES USED:")
    print("=" * 60)
    for i, probe in enumerate(TEXT_PROBES):
        print(f"  [{i:2d}] \"{probe}\"")

    # === 4. What are these latents? ===
    print("\n" + "=" * 60)
    print("4. LATENT SOURCE INFO:")
    print("=" * 60)
    data = np.load(args.latents_npz, allow_pickle=True)
    print(f"  File: {args.latents_npz}")
    print(f"  Keys: {list(data.keys())}")
    x0 = data["x0"]
    print(f"  x0 shape: {x0.shape}, dtype: {x0.dtype}")
    print(f"  x0 min: {x0.min():.4f}, max: {x0.max():.4f}, mean: {x0.mean():.4f}, std: {x0.std():.4f}")
    # Check if there's metadata
    for key in data.keys():
        if key != "x0":
            val = data[key]
            if isinstance(val, np.ndarray) and val.ndim == 0:
                print(f"  {key}: {val.item()}")
            elif isinstance(val, np.ndarray) and val.size < 20:
                print(f"  {key}: {val}")
            else:
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

    N = x0.shape[0]
    x0_4d = torch.tensor(x0.reshape(N, 4, 64, 64), dtype=torch.float32)

    # === Decode ALL latents via VAE ===
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-turbo", subfolder="vae").to(device)
    vae.eval()

    all_decoded_np = []
    with torch.no_grad():
        for i in range(0, N, 4):
            batch = x0_4d[i:i+4].to(device) / 0.13025
            decoded = vae.decode(batch).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            for img_t in decoded:
                img_np = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                all_decoded_np.append(img_np)

    # === 2. Decoded image stats ===
    print("\n" + "=" * 60)
    print("2. DECODED IMAGE STATS:")
    print("=" * 60)
    all_arr = np.stack(all_decoded_np)
    print(f"  Shape: {all_arr.shape}, dtype: {all_arr.dtype}")
    print(f"  Min: {all_arr.min()}, Max: {all_arr.max()}, Mean: {all_arr.mean():.2f}, Std: {all_arr.std():.2f}")
    print(f"  Per-channel means: R={all_arr[:,:,:,0].mean():.2f}, G={all_arr[:,:,:,1].mean():.2f}, B={all_arr[:,:,:,2].mean():.2f}")

    # === 5. Save 3 debug images ===
    print("\n" + "=" * 60)
    print("5. SAVING DEBUG IMAGES:")
    print("=" * 60)
    os.makedirs(args.save_debug_dir, exist_ok=True)
    rng = np.random.RandomState(42)
    debug_idx = rng.choice(N, 3, replace=False)
    for idx in debug_idx:
        path = os.path.join(args.save_debug_dir, f"sample_{idx:03d}.png")
        Image.fromarray(all_decoded_np[idx]).save(path)
        print(f"  Saved {path} (idx={idx})")

    # === Compute embeddings ===
    # Latent-CLIP
    ckpt = hf_hub_download("wendlerc/latent-clip-b-8-512-34b-80k", "checkpoints/epoch_34.pt")
    model_lc, _, _ = latent_clip.create_model_and_transforms(
        "Latent-ViT-B-8-512", pretrained=ckpt, device=device)
    model_lc.eval()
    tokenizer_lc = open_clip.get_tokenizer("ViT-B-32")

    with torch.no_grad():
        lc_img_feats = []
        for i in range(0, N, 32):
            batch = x0_4d[i:i+32].to(device)
            emb = model_lc.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            lc_img_feats.append(emb.cpu())
    lc_img_feats = torch.cat(lc_img_feats, dim=0)

    # Standard CLIP
    model_clip, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device)
    tokenizer_clip = open_clip.get_tokenizer("ViT-B-32")
    model_clip.eval()

    clip_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    with torch.no_grad():
        clip_img_feats = []
        for i in range(N):
            pil_img = Image.fromarray(all_decoded_np[i])
            inp = clip_transform(pil_img).unsqueeze(0).to(device)
            emb = model_clip.encode_image(inp)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            clip_img_feats.append(emb.float().cpu())
    clip_img_feats = torch.cat(clip_img_feats, dim=0)

    # === 3. Per-probe similarity stats ===
    print("\n" + "=" * 60)
    print("3. PER-PROBE COSINE SIMILARITY STATS:")
    print("=" * 60)
    header = f"{'Probe':<45} {'Model':>5}  {'min':>7} {'max':>7} {'mean':>7} {'std':>7}  {'Spearman':>8}"
    print(header)
    print("-" * len(header))

    with torch.no_grad():
        for probe_text in TEXT_PROBES:
            # LC
            tokens_lc = tokenizer_lc([probe_text]).to(device)
            lc_text = model_lc.encode_text(tokens_lc)
            lc_text = lc_text / lc_text.norm(dim=-1, keepdim=True)
            lc_sims = (lc_img_feats @ lc_text.cpu().T).squeeze().numpy()

            # CLIP
            tokens_clip = tokenizer_clip([probe_text]).to(device)
            clip_text = model_clip.encode_text(tokens_clip)
            clip_text = clip_text / clip_text.norm(dim=-1, keepdim=True)
            clip_sims = (clip_img_feats @ clip_text.float().cpu().T).squeeze().numpy()

            rho, _ = spearmanr(lc_sims, clip_sims)

            print(f"{probe_text:<45} {'LC':>5}  {lc_sims.min():>7.4f} {lc_sims.max():>7.4f} {lc_sims.mean():>7.4f} {lc_sims.std():>7.4f}  {rho:>8.4f}")
            print(f"{'':45} {'CLIP':>5}  {clip_sims.min():>7.4f} {clip_sims.max():>7.4f} {clip_sims.mean():>7.4f} {clip_sims.std():>7.4f}")
            print()


if __name__ == "__main__":
    main()
