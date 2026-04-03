"""Compare PCA directions in CLIP vs Latent-CLIP space."""
import argparse
import numpy as np
import torch
import open_clip
import latent_clip
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.decomposition import PCA


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_npz", type=str, required=True)
    p.add_argument("--image_dir", type=str, required=True,
                   help="Dir with sample_*.png generated from same distribution")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Latent-CLIP PCA ---
    x0 = np.load(args.latents_npz)["x0"]
    N = x0.shape[0]
    x0_4d = torch.tensor(x0.reshape(N, 4, 64, 64), dtype=torch.float32)

    ckpt = hf_hub_download("wendlerc/latent-clip-b-8-512-34b-80k", "checkpoints/epoch_34.pt")
    model_lc, _, _ = latent_clip.create_model_and_transforms(
        "Latent-ViT-B-8-512", pretrained=ckpt, device=device)
    model_lc.eval()

    with torch.no_grad():
        lc_feats = []
        for i in range(0, N, 32):
            batch = x0_4d[i:i+32].to(device)
            emb = model_lc.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            lc_feats.append(emb.cpu().numpy())
    lc_feats = np.concatenate(lc_feats, axis=0)

    pca_lc = PCA(n_components=5)
    pca_lc.fit(lc_feats)
    print(f"Latent-CLIP PCA variance: {pca_lc.explained_variance_ratio_[:5]}")

    # --- Standard CLIP PCA ---
    import os
    model_clip, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model_clip.eval()

    paths = sorted([os.path.join(args.image_dir, f)
                    for f in os.listdir(args.image_dir)
                    if f.startswith("sample_") and f.endswith(".png")])
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = torch.stack([preprocess(img) for img in images])

    with torch.no_grad():
        clip_feats = []
        for i in range(0, len(inputs), 32):
            batch = inputs[i:i+32].to(device)
            emb = model_clip.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            clip_feats.append(emb.float().cpu().numpy())
    clip_feats = np.concatenate(clip_feats, axis=0)

    pca_clip = PCA(n_components=5)
    pca_clip.fit(clip_feats)
    print(f"CLIP PCA variance: {pca_clip.explained_variance_ratio_[:5]}")

    # --- Gender direction in CLIP ---
    with torch.no_grad():
        emb_m = model_clip.encode_text(tokenizer(["a photograph of a man"]).to(device))
        emb_f = model_clip.encode_text(tokenizer(["a photograph of a woman"]).to(device))
        emb_m = emb_m / emb_m.norm(dim=-1, keepdim=True)
        emb_f = emb_f / emb_f.norm(dim=-1, keepdim=True)
    gender_dir = (emb_m - emb_f).float().cpu().numpy().flatten()
    gender_dir = gender_dir / np.linalg.norm(gender_dir)

    # --- Cross-space cosine similarities ---
    print("\n=== PC cosine similarities (CLIP vs Latent-CLIP) ===")
    for i in range(3):
        for j in range(3):
            cos = abs(np.dot(pca_clip.components_[i], pca_lc.components_[j]))
            print(f"  |cos(CLIP_PC{i+1}, LC_PC{j+1})| = {cos:.4f}")

    print("\n=== Gender direction correlations ===")
    for i in range(3):
        cos_clip = abs(np.dot(pca_clip.components_[i], gender_dir))
        cos_lc = abs(np.dot(pca_lc.components_[i], gender_dir))
        print(f"  |cos(CLIP_PC{i+1}, gender)| = {cos_clip:.4f}")
        print(f"  |cos(LC_PC{i+1}, gender)| = {cos_lc:.4f}")


if __name__ == "__main__":
    main()
