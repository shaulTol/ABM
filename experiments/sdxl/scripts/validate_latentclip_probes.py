"""Validate Latent-CLIP as text probe: compare text-image similarities in
Latent-CLIP vs standard CLIP using the same samples and same text probes.
If Spearman rho > 0.9, Latent-CLIP is a faithful proxy for standard CLIP."""
import argparse
import numpy as np
import torch
import open_clip
import latent_clip
from huggingface_hub import hf_hub_download
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
from scipy.stats import spearmanr, pearsonr


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

# Also test difference axes (pos - neg similarity)
SEMANTIC_AXES = [
    ("a photograph of a man", "a photograph of a woman"),
    ("a photograph of an old person", "a photograph of a young person"),
    ("a photograph of a person with dark skin", "a photograph of a person with light skin"),
    ("a photograph of a smiling person", "a photograph of a serious person"),
    ("a close up portrait", "a wide shot photograph"),
]


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

    # --- Latent-CLIP model ---
    ckpt = hf_hub_download("wendlerc/latent-clip-b-8-512-34b-80k", "checkpoints/epoch_34.pt")
    model_lc, _, _ = latent_clip.create_model_and_transforms(
        "Latent-ViT-B-8-512", pretrained=ckpt, device=device)
    model_lc.eval()
    tokenizer_lc = open_clip.get_tokenizer("ViT-B-32")  # same tokenizer

    # LC image embeddings
    with torch.no_grad():
        lc_img_feats = []
        for i in range(0, N, 32):
            batch = x0_4d[i:i+32].to(device)
            emb = model_lc.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            lc_img_feats.append(emb.cpu())
    lc_img_feats = torch.cat(lc_img_feats, dim=0)  # (N, 512)

    # --- Standard CLIP model ---
    model_clip, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device)
    tokenizer_clip = open_clip.get_tokenizer("ViT-B-32")
    model_clip.eval()

    # Decode latents → pixels → standard CLIP embeddings
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-turbo", subfolder="vae").to(device)
    vae.eval()

    clip_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    with torch.no_grad():
        clip_img_feats = []
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
            clip_img_feats.append(emb.float().cpu())
    clip_img_feats = torch.cat(clip_img_feats, dim=0)  # (N, 512)
    print(f"Computed image embeddings: LC {lc_img_feats.shape}, CLIP {clip_img_feats.shape}")

    # --- Single text probe similarities ---
    print(f"\n{'Text probe':<50} {'Spearman':>10} {'Pearson':>10}")
    print("-" * 72)

    with torch.no_grad():
        for probe_text in TEXT_PROBES:
            # Latent-CLIP text embedding
            tokens_lc = tokenizer_lc([probe_text]).to(device)
            lc_text = model_lc.encode_text(tokens_lc)
            lc_text = lc_text / lc_text.norm(dim=-1, keepdim=True)
            lc_sims = (lc_img_feats @ lc_text.cpu().T).squeeze().numpy()

            # Standard CLIP text embedding
            tokens_clip = tokenizer_clip([probe_text]).to(device)
            clip_text = model_clip.encode_text(tokens_clip)
            clip_text = clip_text / clip_text.norm(dim=-1, keepdim=True)
            clip_sims = (clip_img_feats @ clip_text.float().cpu().T).squeeze().numpy()

            rho, p_spear = spearmanr(lc_sims, clip_sims)
            r, p_pear = pearsonr(lc_sims, clip_sims)
            print(f"{probe_text:<50} {rho:>10.4f} {r:>10.4f}")

    # --- Semantic axis (difference) correlations ---
    print(f"\n{'Semantic axis':<50} {'Spearman':>10} {'Pearson':>10}")
    print("-" * 72)

    with torch.no_grad():
        for pos_text, neg_text in SEMANTIC_AXES:
            # Latent-CLIP
            lc_pos = model_lc.encode_text(tokenizer_lc([pos_text]).to(device))
            lc_neg = model_lc.encode_text(tokenizer_lc([neg_text]).to(device))
            lc_pos = lc_pos / lc_pos.norm(dim=-1, keepdim=True)
            lc_neg = lc_neg / lc_neg.norm(dim=-1, keepdim=True)
            lc_dir = (lc_pos - lc_neg).float().cpu()
            lc_dir = lc_dir / lc_dir.norm()
            lc_scores = (lc_img_feats @ lc_dir.T).squeeze().numpy()

            # Standard CLIP
            clip_pos = model_clip.encode_text(tokenizer_clip([pos_text]).to(device))
            clip_neg = model_clip.encode_text(tokenizer_clip([neg_text]).to(device))
            clip_pos = clip_pos / clip_pos.norm(dim=-1, keepdim=True)
            clip_neg = clip_neg / clip_neg.norm(dim=-1, keepdim=True)
            clip_dir = (clip_pos - clip_neg).float().cpu()
            clip_dir = clip_dir / clip_dir.norm()
            clip_scores = (clip_img_feats @ clip_dir.T).squeeze().numpy()

            rho, _ = spearmanr(lc_scores, clip_scores)
            r, _ = pearsonr(lc_scores, clip_scores)
            label = f"{pos_text.split('of ')[-1]} vs {neg_text.split('of ')[-1]}"
            print(f"{label:<50} {rho:>10.4f} {r:>10.4f}")


if __name__ == "__main__":
    main()
