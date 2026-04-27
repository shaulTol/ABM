"""Interpret Latent-CLIP PCA directions by correlating with CLIP text axes."""
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


SEMANTIC_AXES = [
    ("male", "female"),
    ("old person", "young person"),
    ("dark skin", "light skin"),
    ("smiling", "serious"),
    ("facing left", "facing right"),
    ("close up portrait", "wide shot"),
    ("bright lighting", "dark lighting"),
    ("colorful", "monochrome"),
    ("blurry", "sharp"),
    ("outdoors", "indoors"),
    ("beard", "no beard"),
    ("glasses", "no glasses"),
    ("hat", "no hat"),
    ("long hair", "short hair"),
    ("makeup", "no makeup"),
    ("happy", "sad"),
    ("angry", "calm"),
    ("formal clothing", "casual clothing"),
    ("looking at camera", "looking away"),
    ("bald", "full hair"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_npz", type=str, required=True)
    p.add_argument("--n_pcs", type=int, default=5)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load latents
    x0 = np.load(args.latents_npz)["x0"]
    N = x0.shape[0]
    x0_4d = torch.tensor(x0.reshape(N, 4, 64, 64), dtype=torch.float32)
    print(f"Loaded {N} latents")

    # --- Latent-CLIP embeddings + PCA ---
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

    pca_lc = PCA(n_components=args.n_pcs)
    lc_scores = pca_lc.fit_transform(lc_feats)
    print(f"Latent-CLIP PCA variance: {[f'{v:.1%}' for v in pca_lc.explained_variance_ratio_]}")

    # --- Decode latents → pixels → standard CLIP embeddings ---
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-turbo", subfolder="vae").to(device)
    vae.eval()

    model_clip, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model_clip.eval()

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
    print(f"CLIP image embeddings: {clip_feats.shape}")

    # --- Compute CLIP text directions ---
    print(f"\n{'Semantic axis':<35} ", end="")
    for k in range(args.n_pcs):
        print(f"{'LC_PC'+str(k+1):>10}", end="")
    print()
    print("-" * (35 + 10 * args.n_pcs))

    with torch.no_grad():
        for pos_word, neg_word in SEMANTIC_AXES:
            prompt_pos = f"a photograph of a {pos_word}"
            prompt_neg = f"a photograph of a {neg_word}"
            emb_pos = model_clip.encode_text(tokenizer([prompt_pos]).to(device))
            emb_neg = model_clip.encode_text(tokenizer([prompt_neg]).to(device))
            emb_pos = emb_pos / emb_pos.norm(dim=-1, keepdim=True)
            emb_neg = emb_neg / emb_neg.norm(dim=-1, keepdim=True)
            text_dir = (emb_pos - emb_neg).float().cpu().numpy().flatten()
            text_dir = text_dir / np.linalg.norm(text_dir)

            # Project each image's CLIP embedding onto this text direction
            clip_proj = clip_feats @ text_dir  # (N,)

            # Correlate with each LC PC score
            label = f"{pos_word} vs {neg_word}"
            print(f"{label:<35} ", end="")
            for k in range(args.n_pcs):
                corr = np.corrcoef(clip_proj, lc_scores[:, k])[0, 1]
                print(f"{corr:>10.3f}", end="")
            print()

    # Also check CLIP PCA correlations for reference
    pca_clip = PCA(n_components=args.n_pcs)
    clip_scores = pca_clip.fit_transform(clip_feats)
    print(f"\nCLIP PCA variance: {[f'{v:.1%}' for v in pca_clip.explained_variance_ratio_]}")

    print(f"\n{'Semantic axis':<35} ", end="")
    for k in range(args.n_pcs):
        print(f"{'CLIP_PC'+str(k+1):>10}", end="")
    print()
    print("-" * (35 + 10 * args.n_pcs))

    with torch.no_grad():
        for pos_word, neg_word in SEMANTIC_AXES:
            prompt_pos = f"a photograph of a {pos_word}"
            prompt_neg = f"a photograph of a {neg_word}"
            emb_pos = model_clip.encode_text(tokenizer([prompt_pos]).to(device))
            emb_neg = model_clip.encode_text(tokenizer([prompt_neg]).to(device))
            emb_pos = emb_pos / emb_pos.norm(dim=-1, keepdim=True)
            emb_neg = emb_neg / emb_neg.norm(dim=-1, keepdim=True)
            text_dir = (emb_pos - emb_neg).float().cpu().numpy().flatten()
            text_dir = text_dir / np.linalg.norm(text_dir)

            clip_proj = clip_feats @ text_dir
            label = f"{pos_word} vs {neg_word}"
            print(f"{label:<35} ", end="")
            for k in range(args.n_pcs):
                corr = np.corrcoef(clip_proj, clip_scores[:, k])[0, 1]
                print(f"{corr:>10.3f}", end="")
            print()


if __name__ == "__main__":
    main()
