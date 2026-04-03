"""Compare PCA directions in CLIP vs Latent-CLIP space using same latents."""
import argparse
import numpy as np
import torch
import open_clip
import latent_clip
from huggingface_hub import hf_hub_download
from diffusers import AutoencoderKL
from sklearn.decomposition import PCA
from PIL import Image
from torchvision import transforms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_npz", type=str, required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load latents
    x0 = np.load(args.latents_npz)["x0"]
    N = x0.shape[0]
    x0_4d = torch.tensor(x0.reshape(N, 4, 64, 64), dtype=torch.float32)
    print(f"Loaded {N} latents")

    # --- Latent-CLIP embeddings ---
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
    print(f"Latent-CLIP: {lc_feats.shape}")

    # --- Decode latents to images, then standard CLIP ---
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-turbo", subfolder="vae").to(device)
    vae.eval()

    model_clip, _, preprocess_clip = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model_clip.eval()

    # CLIP preprocess for decoded images
    clip_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    with torch.no_grad():
        clip_feats = []
        for i in range(0, N, 4):
            batch = x0_4d[i:i+4].to(device) / 0.13025
            decoded = vae.decode(batch).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            # Convert to PIL, then CLIP preprocess
            clip_inputs = []
            for img_t in decoded:
                img_np = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                clip_inputs.append(clip_transform(pil_img))
            clip_batch = torch.stack(clip_inputs).to(device)
            emb = model_clip.encode_image(clip_batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            clip_feats.append(emb.float().cpu().numpy())
    clip_feats = np.concatenate(clip_feats, axis=0)
    print(f"CLIP: {clip_feats.shape}")

    # --- PCA ---
    pca_lc = PCA(n_components=5)
    pca_lc.fit(lc_feats)
    pca_clip = PCA(n_components=5)
    pca_clip.fit(clip_feats)

    print(f"\nLatent-CLIP PCA variance: {[f'{v:.1%}' for v in pca_lc.explained_variance_ratio_[:5]]}")
    print(f"CLIP PCA variance:        {[f'{v:.1%}' for v in pca_clip.explained_variance_ratio_[:5]]}")

    # --- Gender direction ---
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
            if cos > 0.1:
                print(f"  |cos(CLIP_PC{i+1}, LC_PC{j+1})| = {cos:.4f}")

    print("\n=== Gender direction correlations ===")
    for i in range(3):
        cos_clip = abs(np.dot(pca_clip.components_[i], gender_dir))
        cos_lc = abs(np.dot(pca_lc.components_[i], gender_dir))
        print(f"  |cos(CLIP_PC{i+1}, gender)| = {cos_clip:.4f}    |cos(LC_PC{i+1}, gender)| = {cos_lc:.4f}")

    # --- Per-sample correlation of projections ---
    print("\n=== Per-sample projection correlations ===")
    for i in range(3):
        proj_clip = clip_feats @ pca_clip.components_[i]
        proj_lc = lc_feats @ pca_lc.components_[i]
        corr = abs(np.corrcoef(proj_clip, proj_lc)[0, 1])
        print(f"  |corr(CLIP_PC{i+1} scores, LC_PC{i+1} scores)| = {corr:.4f}")

    # Gender score correlation
    gender_clip = clip_feats @ gender_dir
    gender_lc = lc_feats @ gender_dir
    corr_gender = np.corrcoef(gender_clip, gender_lc)[0, 1]
    print(f"  corr(CLIP gender scores, LC gender scores) = {corr_gender:.4f}")


if __name__ == "__main__":
    main()
