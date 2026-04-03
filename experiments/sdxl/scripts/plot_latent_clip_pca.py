"""PCA in Latent-CLIP space with VAE-decoded thumbnails."""
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from PIL import Image
from diffusers import AutoencoderKL
import latent_clip
from huggingface_hub import hf_hub_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_npz", type=str, required=True,
                   help="Path to sdxl_latents_*.npz (must contain 'x0')")
    p.add_argument("--title", type=str, default="Latent-CLIP PCA")
    p.add_argument("--output", type=str, default="latent_clip_pca.png")
    p.add_argument("--n_thumbs", type=int, default=100,
                   help="Number of thumbnails to show (evenly spaced)")
    p.add_argument("--thumb_size", type=int, default=40)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load latents
    x0 = np.load(args.latents_npz)["x0"]  # (N, 16384)
    N = x0.shape[0]
    x0_4d = torch.tensor(x0.reshape(N, 4, 64, 64), dtype=torch.float32)
    print(f"Loaded {N} latents from {args.latents_npz}")

    # Latent-CLIP embedding
    checkpoint_path = hf_hub_download(
        "wendlerc/latent-clip-b-8-512-34b-80k", "checkpoints/epoch_34.pt"
    )
    model_lc, _, _ = latent_clip.create_model_and_transforms(
        "Latent-ViT-B-8-512", pretrained=checkpoint_path, device=device,
    )
    model_lc.eval()
    with torch.no_grad():
        clip_feats = []
        for i in range(0, N, 32):
            batch = x0_4d[i:i+32].to(device)
            emb = model_lc.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            clip_feats.append(emb.cpu().numpy())
    clip_feats = np.concatenate(clip_feats, axis=0)
    print(f"Latent-CLIP embeddings: {clip_feats.shape}")

    # PCA
    pca = PCA(n_components=2)
    scores = pca.fit_transform(clip_feats)

    # Decode subset for thumbnails
    step = max(1, N // args.n_thumbs)
    idx = np.arange(0, N, step)[:args.n_thumbs]

    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-turbo", subfolder="vae").to(device)
    vae.eval()

    images_pil = []
    with torch.no_grad():
        for i in range(0, len(idx), 4):
            batch_idx = idx[i:i+4]
            batch = x0_4d[batch_idx].to(device) / 0.13025
            decoded = vae.decode(batch).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            for img_t in decoded:
                img_np = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                images_pil.append(Image.fromarray(img_np))

    # Plot
    fig, ax = plt.subplots(figsize=(16, 12))
    for i, j in enumerate(idx):
        x, y = scores[j, 0], scores[j, 1]
        thumb = images_pil[i].resize((args.thumb_size, args.thumb_size))
        im = OffsetImage(np.array(thumb), zoom=0.7)
        ab = AnnotationBbox(im, (x, y), frameon=True, pad=0.1,
                            bboxprops=dict(edgecolor="gray", linewidth=0.5))
        ax.add_artist(ab)

    ax.set_xlim(scores[idx, 0].min() - 0.01, scores[idx, 0].max() + 0.01)
    ax.set_ylim(scores[idx, 1].min() - 0.01, scores[idx, 1].max() + 0.01)
    ax.set_xlabel(f"Latent-CLIP PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=13)
    ax.set_ylabel(f"Latent-CLIP PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=13)
    ax.set_title(args.title, fontsize=13)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    print(f"PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}")


if __name__ == "__main__":
    main()
