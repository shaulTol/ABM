#!/usr/bin/env python3
"""Elderly DA1 vs PC1 grid for the free-conv stopping run (dt=0.0025, T=2000)."""

import os, sys, json, argparse
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "shared"))

VAE_SCALE = 0.13025


def decode_latents(vae, latents_flat, latent_shape, device, batch_size=8):
    images = []
    N = latents_flat.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = torch.from_numpy(latents_flat[start:end]).float().to(device)
        z = batch.view(-1, *latent_shape) / VAE_SCALE
        with torch.no_grad():
            decoded = vae.decode(z, return_dict=False)[0]
        pixels = ((decoded + 1.0) / 2.0).clamp(0, 1)
        pixel_bytes = (pixels * 255).byte().cpu()
        for img_t in pixel_bytes:
            images.append(Image.fromarray(img_t.permute(1, 2, 0).numpy()))
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_da_pcavda")
    parser.add_argument("--name", default="elderly_gs7p5_fc_dt0p0025",
                        help="basename of clip_da_<name>.{json,npz}")
    parser.add_argument("--x0_path",
                        default="../guidance_sweep/results_multiprompt_sweep/gen_mp_p5_elderly_gs7p5_quick_x0.npz")
    parser.add_argument("--output",
                        default="results_da_pcavda/fig_elderly_da_vs_pca_freeconv.png")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--use_taustar", action="store_true",
                        help="Use eigvecs from C(tau*_freeconv) (longest lag) "
                             "instead of the peak-gap-lag DA1/DA2.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_shape = (4, 128, 128)  # 1024x1024

    print("Loading SDXL VAE...")
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    vae = pipe.vae.to(device).float().eval()
    del pipe.unet, pipe
    torch.cuda.empty_cache()

    js_path = os.path.join(args.results_dir, f"clip_da_{args.name}.json")
    npz_path = os.path.join(args.results_dir, f"clip_da_{args.name}.npz")
    js = json.load(open(js_path))
    da_npz = np.load(npz_path)
    if args.use_taustar:
        V = da_npz["eigvecs_full_topk_at_tau_star"]  # (768, 20)
        da1 = V[:, 0].astype(np.float64)
        da2 = V[:, 1].astype(np.float64)
        tau_star = int(da_npz["tau_star_freeconv"])
        print(f"Using DA1/DA2 from C(tau* freeconv = {tau_star}, longest lag)")
    else:
        da1 = da_npz["da1_eigvec"]
        da2 = da_npz["da2_eigvec"]
        print(f"Using DA1/DA2 from peak-gap lag (default)")

    x0_flat = np.load(args.x0_path)["x0"]
    print(f"Decoding {x0_flat.shape[0]} images...")
    images = decode_latents(vae, x0_flat, latent_shape, device)

    from sdxl_components import RealCLIPEmbedder
    clip_embedder = RealCLIPEmbedder(vae, device=device, clip_batch_size=16)
    print("CLIP embedding...")
    z0_clip = clip_embedder.embed(x0_flat, latent_shape=latent_shape)
    print(f"  z0_clip shape: {z0_clip.shape}")

    z0c = z0_clip - z0_clip.mean(axis=0)
    cov = (z0c.T @ z0c) / z0_clip.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1].copy()
    pc1 = eigvecs[:, idx[0]]
    pc1_var = eigvals[idx[0]] / eigvals.sum()
    cos_da1_pc1 = abs(float(np.dot(da1, pc1)))

    da1_scores = z0c @ da1
    da2_scores = z0c @ da2
    pc1_scores = z0c @ pc1

    k = args.k
    da1_order = np.argsort(da1_scores)
    pc1_order = np.argsort(pc1_scores)

    da1_low = da1_order[:k]
    da1_high = da1_order[-k:][::-1]
    pc1_low = pc1_order[:k]
    pc1_high = pc1_order[-k:][::-1]

    print(f"\nDA1 high indices: {da1_high.tolist()}")
    print(f"DA1 low  indices: {da1_low.tolist()}")
    print(f"PC1 high indices: {pc1_high.tolist()}")
    print(f"PC1 low  indices: {pc1_low.tolist()}")

    # Probe correlations from JSON
    da1_corr = js.get("da1_correlations", {})
    da1_top3 = sorted(da1_corr.items(), key=lambda x: abs(x[1]["rho"]), reverse=True)[:3]

    from scipy.stats import spearmanr
    from transformers import CLIPModel, CLIPTokenizer
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_model.eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    probe_pairs = {
        "gender": ("a photograph of a man", "a photograph of a woman"),
        "age": ("a photograph of a young person", "a photograph of an old person"),
        "realism": ("a realistic photograph", "an artistic painting"),
    }
    pc1_corrs = {}
    da1_probe_corrs = {}
    da2_probe_corrs = {}
    for name, (pos, neg) in probe_pairs.items():
        with torch.no_grad():
            tok_p = tokenizer(pos, return_tensors="pt", padding=True).to(device)
            tok_n = tokenizer(neg, return_tensors="pt", padding=True).to(device)
            emb_p = clip_model.get_text_features(**tok_p)
            emb_n = clip_model.get_text_features(**tok_n)
            emb_p = emb_p / emb_p.norm(dim=-1, keepdim=True)
            emb_n = emb_n / emb_n.norm(dim=-1, keepdim=True)
            axis = (emb_p - emb_n).squeeze().cpu().numpy()
            axis = axis / np.linalg.norm(axis)
        proj = z0c @ axis
        rho, _ = spearmanr(pc1_scores, proj)
        pc1_corrs[name] = rho
        rho_da1, _ = spearmanr(da1_scores, proj)
        rho_da2, _ = spearmanr(da2_scores, proj)
        da1_probe_corrs[name] = float(rho_da1)
        da2_probe_corrs[name] = float(rho_da2)
    pc1_top3 = sorted(pc1_corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    da1_probe_top = sorted(da1_probe_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    da2_probe_top = sorted(da2_probe_corrs.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\nDA1 top probes (from JSON, peak-gap-lag): {da1_top3}")
    print(f"DA1 (this run's vec) probe rho: {da1_probe_top}")
    print(f"DA2 (this run's vec) probe rho: {da2_probe_top}")
    print(f"PC1 top probes: {pc1_top3}")

    # --- Figure ---
    thumb = 180
    fig_w = k * thumb / 100 + 1.5
    fig_h = 5.5
    fig = plt.figure(figsize=(fig_w + 1, fig_h))
    gs = GridSpec(4, k + 1, figure=fig, hspace=0.15, wspace=0.05,
                  left=0.01, right=0.99, top=0.98, bottom=0.02,
                  width_ratios=[0.6] + [1] * k)

    def place_row(row, indices, label_text):
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.text(0.9, 0.5, label_text, ha="right", va="center",
                      fontsize=9, fontweight="bold", transform=ax_label.transAxes)
        ax_label.axis("off")
        for col, img_idx in enumerate(indices):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(images[img_idx])
            ax.axis("off")

    place_row(0, da1_high, "DA1\nhigh")
    place_row(1, da1_low, "DA1\nlow")
    place_row(2, pc1_high, "PC1\nhigh")
    place_row(3, pc1_low, "PC1\nlow")

    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    hires = args.output.replace(".png", "_hires.png")
    fig.savefig(hires, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {args.output}")
    print(f"Saved: {hires}")
    plt.close(fig)

    # Save indices for later reuse
    out_npz = args.output.replace(".png", "_indices.npz")
    np.savez_compressed(out_npz,
        da1_high=da1_high, da1_low=da1_low,
        pc1_high=pc1_high, pc1_low=pc1_low,
        da1_scores=da1_scores, da2_scores=da2_scores, pc1_scores=pc1_scores,
        cos_da1_pc1=np.array(cos_da1_pc1),
        pc1_var=np.array(pc1_var),
    )
    print(f"Saved: {out_npz}")


if __name__ == "__main__":
    main()
