"""Plot generated images along CLIP male/female axis + orthogonal PC1."""
import os
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
    p.add_argument("--output", type=str, default="clip_gender.png")
    p.add_argument("--thumb_size", type=int, default=40)
    args = p.parse_args()

    # Collect image paths
    paths = sorted([
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if f.startswith("sample_") and f.endswith(".png")
    ])
    print(f"Found {len(paths)} images")

    # Load CLIP
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    # Embed images
    images_pil = [Image.open(p).convert("RGB") for p in paths]
    inputs = torch.stack([preprocess(img) for img in images_pil])

    with torch.no_grad():
        img_features = model.encode_image(inputs)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        img_features = img_features.float().numpy()

    # Compute male/female text direction
    with torch.no_grad():
        text_male = tokenizer(["a photograph of a man"])
        text_female = tokenizer(["a photograph of a woman"])
        emb_male = model.encode_text(text_male)
        emb_female = model.encode_text(text_female)
        emb_male = emb_male / emb_male.norm(dim=-1, keepdim=True)
        emb_female = emb_female / emb_female.norm(dim=-1, keepdim=True)

    gender_dir = (emb_male - emb_female).float().numpy().flatten()
    gender_dir = gender_dir / np.linalg.norm(gender_dir)

    # Project images onto gender axis
    gender_scores = img_features @ gender_dir  # positive = more male

    # Orthogonal residual, then PCA for second axis
    residual = img_features - np.outer(gender_scores, gender_dir)
    pca = PCA(n_components=1)
    ortho_scores = pca.fit_transform(residual).flatten()

    print(f"Gender score range: [{gender_scores.min():.3f}, {gender_scores.max():.3f}]")
    print(f"Orthogonal PC1 explained var: {pca.explained_variance_ratio_[0]:.1%}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(gender_scores, ortho_scores, alpha=0.9, s=20, color='tab:blue', zorder=1)

    for i, (x, y) in enumerate(zip(gender_scores, ortho_scores)):
        thumb = images_pil[i].resize((args.thumb_size, args.thumb_size))
        im = OffsetImage(np.array(thumb), zoom=0.7)
        ab = AnnotationBbox(im, (x, y), frameon=True, pad=0.1,
                            bboxprops=dict(edgecolor='gray', linewidth=0.5))
        ax.add_artist(ab)

    ax.set_xlabel("← Female                    Male →", fontsize=13)
    ax.set_ylabel(f"Orthogonal PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_title("ControlNet circle + 'human being' — CLIP gender axis")
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
