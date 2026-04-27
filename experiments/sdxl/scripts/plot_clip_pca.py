"""Plot generated images in CLIP space (PC1 vs PC2) with thumbnails."""
import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--output", type=str, default="clip_pca.png")
    p.add_argument("--thumb_size", type=int, default=64)
    args = p.parse_args()

    # Collect image paths
    paths = sorted([
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if f.startswith("sample_") and f.endswith(".png")
    ])
    print(f"Found {len(paths)} images")

    # Load CLIP
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai")
    except ImportError:
        import clip
        model, preprocess = clip.load("ViT-B/32", device="cpu")

    model.eval()

    # Embed images
    images_pil = [Image.open(p).convert("RGB") for p in paths]
    inputs = torch.stack([preprocess(img) for img in images_pil])

    with torch.no_grad():
        features = model.encode_image(inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.float().numpy()

    print(f"CLIP embeddings: {features.shape}")

    # PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(features)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")

    # Plot with thumbnails
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.9, s=20, color='tab:blue', zorder=1)

    for i, (x, y) in enumerate(coords):
        thumb = images_pil[i].resize((args.thumb_size, args.thumb_size))
        im = OffsetImage(np.array(thumb), zoom=0.7)
        ab = AnnotationBbox(im, (x, y), frameon=True, pad=0.1,
                            bboxprops=dict(edgecolor='gray', linewidth=0.5))
        ax.add_artist(ab)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("ControlNet circle scribble — SDXL Turbo samples in CLIP space")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
