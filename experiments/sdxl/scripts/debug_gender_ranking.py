"""Check if LC and CLIP agree on who's male/female."""
import numpy as np
import torch
import open_clip
import latent_clip
from huggingface_hub import hf_hub_download
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load latents
x0 = np.load("results_controlnet/sdxl_latents_sdxl_ssa_cn_dps_face.npz")["x0"]
N = x0.shape[0]
x0_4d = torch.tensor(x0.reshape(N, 4, 64, 64), dtype=torch.float32)

# Latent-CLIP
ckpt = hf_hub_download("wendlerc/latent-clip-b-8-512-34b-80k", "checkpoints/epoch_34.pt")
model_lc, _, _ = latent_clip.create_model_and_transforms(
    "Latent-ViT-B-8-512", pretrained=ckpt, device=device)
model_lc.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

with torch.no_grad():
    lc_feats = []
    for i in range(0, N, 32):
        batch = x0_4d[i:i+32].to(device)
        emb = model_lc.encode_image(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        lc_feats.append(emb.cpu())
    lc_feats = torch.cat(lc_feats, dim=0)

    lc_man = model_lc.encode_text(tokenizer(["a photograph of a man"]).to(device))
    lc_woman = model_lc.encode_text(tokenizer(["a photograph of a woman"]).to(device))
    lc_man = lc_man / lc_man.norm(dim=-1, keepdim=True)
    lc_woman = lc_woman / lc_woman.norm(dim=-1, keepdim=True)

lc_score = ((lc_feats @ lc_man.cpu().T) - (lc_feats @ lc_woman.cpu().T)).squeeze().numpy()

# Standard CLIP
model_clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
model_clip.eval()

clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])

vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-turbo", subfolder="vae").to(device)
vae.eval()

with torch.no_grad():
    clip_feats = []
    for i in range(0, N, 4):
        batch = x0_4d[i:i+4].to(device) / 0.13025
        decoded = vae.decode(batch).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        for img_t in decoded:
            img_np = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            inp = clip_transform(pil_img).unsqueeze(0).to(device)
            emb = model_clip.encode_image(inp)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            clip_feats.append(emb.float().cpu())
    clip_feats = torch.cat(clip_feats, dim=0)

    clip_man = model_clip.encode_text(tokenizer(["a photograph of a man"]).to(device))
    clip_woman = model_clip.encode_text(tokenizer(["a photograph of a woman"]).to(device))
    clip_man = clip_man / clip_man.norm(dim=-1, keepdim=True)
    clip_woman = clip_woman / clip_woman.norm(dim=-1, keepdim=True)

clip_score = ((clip_feats @ clip_man.float().cpu().T) - (clip_feats @ clip_woman.float().cpu().T)).squeeze().numpy()

# Print top/bottom 5
print("=== GENDER AXIS: sim('man') - sim('woman') ===")
print(f"{'':>5}  LC score range: [{lc_score.min():.6f}, {lc_score.max():.6f}], std={lc_score.std():.6f}")
print(f"{'':>5}  CLIP score range: [{clip_score.min():.6f}, {clip_score.max():.6f}], std={clip_score.std():.6f}")

print("\n--- Top 5 MOST MALE (highest score) ---")
print(f"{'Rank':>4}  {'Idx':>4}  {'LC score':>10}  {'CLIP score':>11}  {'LC rank':>8}  {'CLIP rank':>9}")
lc_order = np.argsort(-lc_score)
clip_order = np.argsort(-clip_score)
lc_ranks = np.argsort(np.argsort(-lc_score))
clip_ranks = np.argsort(np.argsort(-clip_score))

print("\nBy CLIP ranking:")
for r in range(5):
    idx = clip_order[r]
    print(f"  {r+1:>2}   {idx:>4}  {lc_score[idx]:>10.6f}  {clip_score[idx]:>11.6f}  {lc_ranks[idx]+1:>8}  {clip_ranks[idx]+1:>9}")

print("\nBy LC ranking:")
for r in range(5):
    idx = lc_order[r]
    print(f"  {r+1:>2}   {idx:>4}  {lc_score[idx]:>10.6f}  {clip_score[idx]:>11.6f}  {lc_ranks[idx]+1:>8}  {clip_ranks[idx]+1:>9}")

print("\n--- Bottom 5 MOST FEMALE (lowest score) ---")
print("\nBy CLIP ranking:")
for r in range(5):
    idx = clip_order[-(r+1)]
    print(f"  {N-r:>2}   {idx:>4}  {lc_score[idx]:>10.6f}  {clip_score[idx]:>11.6f}  {lc_ranks[idx]+1:>8}  {clip_ranks[idx]+1:>9}")

print("\nBy LC ranking:")
for r in range(5):
    idx = lc_order[-(r+1)]
    print(f"  {N-r:>2}   {idx:>4}  {lc_score[idx]:>10.6f}  {clip_score[idx]:>11.6f}  {lc_ranks[idx]+1:>8}  {clip_ranks[idx]+1:>9}")

# Also save the debug images for the extremes
import os
os.makedirs("debug_decode/extremes", exist_ok=True)
with torch.no_grad():
    for label, indices in [("clip_male", clip_order[:5]), ("clip_female", clip_order[-5:]),
                           ("lc_male", lc_order[:5]), ("lc_female", lc_order[-5:])]:
        for rank, idx in enumerate(indices):
            batch = x0_4d[idx:idx+1].to(device) / 0.13025
            decoded = vae.decode(batch).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            img_np = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img_np).save(f"debug_decode/extremes/{label}_rank{rank+1}_idx{idx}.png")
print("\nSaved extreme images to debug_decode/extremes/")
