"""Plot two image sets in shared CLIP PCA space with thumbnails."""
import os
import argparse
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def load_images(image_dir, prefix="sample_", ext=".png"):
    """Load images from directory, trying common naming patterns."""
    paths = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.endswith(ext) and not f.startswith("grid") and not f.startswith("final")
    ])
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir_a", type=str, required=True, help="First image directory")
    p.add_argument("--dir_b", type=str, required=True, help="Second image directory")
    p.add_argument("--label_a", type=str, default="Set A")
    p.add_argument("--label_b", type=str, default="Set B")
    p.add_argument("--output", type=str, default="clip_pca_compare.png")
    p.add_argument("--thumb_size", type=int, default=36)
    p.add_argument("--max_images", type=int, default=100)
    args = p.parse_args()

    paths_a = load_images(args.dir_a)[:args.max_images]
    paths_b = load_images(args.dir_b)[:args.max_images]
    print(f"Set A ({args.label_a}): {len(paths_a)} images")
    print(f"Set B ({args.label_b}): {len(paths_b)} images")

    # Load CLIP
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    model.eval()

    # Embed both sets
    all_pil = []
    labels = []
    for p_img in paths_a:
        all_pil.append(Image.open(p_img).convert("RGB"))
        labels.append("A")
    for p_img in paths_b:
        all_pil.append(Image.open(p_img).convert("RGB"))
        labels.append("B")

    inputs = torch.stack([preprocess(img) for img in all_pil])
    with torch.no_grad():
        features = model.encode_image(inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.float().numpy()

    labels = np.array(labels)
    print(f"Total embeddings: {features.shape}")

    # Joint PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(features)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 11))

    colors = {"A": "tab:blue", "B": "tab:red"}
    for label in ["A", "B"]:
        mask = labels == label
        name = args.label_a if label == "A" else args.label_b
        ax.scatter(coords[mask, 0], coords[mask, 1], alpha=0.6, s=15,
                   color=colors[label], label=name, zorder=1)

    for i, (x, y) in enumerate(coords):
        thumb = all_pil[i].resize((args.thumb_size, args.thumb_size))
        im = OffsetImage(np.array(thumb), zoom=0.6)
        ec = colors[labels[i]]
        ab = AnnotationBbox(im, (x, y), frameon=True, pad=0.1,
                            bboxprops=dict(edgecolor=ec, linewidth=1.2))
        ax.add_artist(ab)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("CLIP space comparison")
    ax.legend(fontsize=12, loc="upper left")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
