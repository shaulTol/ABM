#!/usr/bin/env python3
"""
Compare DA1 directions at different stopping criteria for SDXL Square N=1000.

Three τ* values:
  - Old gap-max: τ*=1265 (from original run_ssa_da_clip.py)
  - Conservative noise floor: τ*=1422 (upper analytic bound)
  - Aggressive noise floor: τ*=1500 (lower analytic bound)

Loads eigenvectors from the DA npz, decodes+CLIP-encodes x0 latents to get
individual embeddings, then computes gender/age/etc correlations for each DA1.

Requires GPU (VAE decode + CLIP encode).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import json
import numpy as np
import torch
from scipy.stats import spearmanr

OUTDIR = os.path.dirname(os.path.abspath(__file__))
SDXL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'sdxl')


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results_dir = os.path.join(SDXL_DIR, 'results_autoresearch')

    # --- Load eigenvector data ---
    da_npz = np.load(os.path.join(results_dir,
                     'clip_da_autoresearch_square_N1000_T1500.npz'),
                     allow_pickle=True)
    lags = da_npz['snapshot_lags']
    mu_clip = da_npz['mu_clip']  # (768,)

    print(f"Available lags: {lags}")
    print(f"mu_clip shape: {mu_clip.shape}")

    # The three τ* values to compare
    tau_old = 1265    # old gap-max
    tau_cons = 1422   # conservative noise floor
    tau_aggr = 1500   # aggressive noise floor (last lag in grid)

    # Check which lags are available
    tau_values = {'gap_max': tau_old, 'conservative': tau_cons, 'aggressive': tau_aggr}
    da1_directions = {}

    for label, tau in tau_values.items():
        key = f'eigvecs_lag{tau}'
        if key in da_npz:
            eigvecs = da_npz[key]  # (768, 3)
            da1 = eigvecs[:, 0]    # first eigenvector = DA1
            da1_directions[label] = da1
            print(f"  {label}: τ={tau}, DA1 shape={da1.shape}")
        else:
            print(f"  {label}: τ={tau} NOT in npz (available: {[k for k in da_npz.files if 'eigvecs' in k]})")

    if len(da1_directions) < 2:
        print("ERROR: Need at least 2 DA1 directions to compare")
        return

    # Also load the stored da1_eigvec (at gap-max τ*)
    if 'da1_eigvec' in da_npz:
        da1_stored = da_npz['da1_eigvec']
        cos_stored = abs(np.dot(da1_stored / np.linalg.norm(da1_stored),
                                da1_directions.get('gap_max', np.zeros(768)) /
                                max(np.linalg.norm(da1_directions.get('gap_max', np.zeros(768))), 1e-12)))
        print(f"  Sanity: |cos(da1_stored, da1@{tau_old})| = {cos_stored:.6f}")

    # --- Direction comparisons (no GPU needed) ---
    print("\n=== DA1 Direction Comparisons ===")
    labels = list(da1_directions.keys())
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            d1 = da1_directions[labels[i]]
            d2 = da1_directions[labels[j]]
            cos = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
            angle = np.degrees(np.arccos(np.clip(abs(cos), 0, 1)))
            print(f"  {labels[i]:15s} vs {labels[j]:15s}: |cos|={abs(cos):.6f}, angle={angle:.2f}°")

    # --- Load latents and CLIP-encode for correlation analysis ---
    latents_path = os.path.join(results_dir,
                                'clip_da_latents_autoresearch_square_N1000_T1500.npz')
    if not os.path.exists(latents_path):
        print(f"\nLatents file not found: {latents_path}")
        print("Cannot compute gender correlations without CLIP embeddings.")
        # Save direction-only results
        save_results(da1_directions, tau_values, {}, {})
        return

    print(f"\nLoading latents from {latents_path}...")
    latents_npz = np.load(latents_path)
    x0 = latents_npz['x0']  # (N, 4, 64, 64)
    N = x0.shape[0]
    print(f"  x0 shape: {x0.shape}")

    # VAE decode + CLIP encode
    print("\nLoading VAE + CLIP models...")
    from diffusers import AutoencoderKL
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image

    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-turbo",
                                         subfolder="vae",
                                         torch_dtype=torch.float32).to(device)
    vae.eval()

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    VAE_SCALE = 0.13025
    vae_bs = 4
    clip_bs = 16

    # Reshape if flattened (N, 16384) → (N, 4, 64, 64)
    if x0.ndim == 2:
        latent_dim = x0.shape[1]
        c = 4
        hw = int(np.sqrt(latent_dim // c))
        x0 = x0.reshape(N, c, hw, hw)
        print(f"  Reshaped to {x0.shape}")

    print(f"VAE-decoding + CLIP-encoding {N} images...")
    all_embeddings = []
    x0_torch = torch.from_numpy(x0).to(torch.float32)

    for start in range(0, N, vae_bs):
        end = min(start + vae_bs, N)
        batch = x0_torch[start:end].to(device) / VAE_SCALE

        with torch.no_grad():
            decoded = vae.decode(batch, return_dict=False)[0]
        pixels = ((decoded + 1.0) / 2.0).clamp(0, 1).cpu().float()

        pil_images = []
        for k in range(pixels.shape[0]):
            img = (pixels[k].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img))

        for cs in range(0, len(pil_images), clip_bs):
            ce = min(cs + clip_bs, len(pil_images))
            inputs = clip_processor(images=pil_images[cs:ce],
                                    return_tensors="pt").to(device)
            with torch.no_grad():
                emb = clip_model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(emb.cpu().numpy())

        if (start // vae_bs) % 50 == 0:
            print(f"  {end}/{N}")

    clip_embeddings = np.concatenate(all_embeddings, axis=0)  # (N, 768)
    print(f"  CLIP embeddings: {clip_embeddings.shape}")

    # --- Text probes ---
    print("\nDefining text probes...")
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    probe_pairs = [
        ("gender", "a photograph of a woman", "a photograph of a man"),
        ("age", "a photograph of a young person", "a photograph of an old person"),
        ("brightness", "a bright image", "a dark image"),
        ("framing", "a close-up portrait", "a wide-angle photograph"),
        ("expression", "a smiling person", "a serious person"),
    ]

    probe_axes = {}
    for name, text_pos, text_neg in probe_pairs:
        inputs_pos = tokenizer(text_pos, return_tensors="pt", padding=True,
                               truncation=True).to(device)
        inputs_neg = tokenizer(text_neg, return_tensors="pt", padding=True,
                               truncation=True).to(device)
        with torch.no_grad():
            emb_pos = clip_model.get_text_features(**inputs_pos)
            emb_neg = clip_model.get_text_features(**inputs_neg)
            emb_pos = emb_pos / emb_pos.norm(dim=-1, keepdim=True)
            emb_neg = emb_neg / emb_neg.norm(dim=-1, keepdim=True)
        axis = (emb_pos - emb_neg).cpu().numpy().squeeze(0)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        probe_axes[name] = axis

    # --- Correlations for each DA1 ---
    print("\n=== DA1 Correlations ===")
    all_cosines = {}
    all_correlations = {}

    for label, da1 in da1_directions.items():
        tau = tau_values[label]
        print(f"\n--- {label} (τ={tau}) ---")

        # Cosine with probes
        da1_norm = da1 / (np.linalg.norm(da1) + 1e-12)
        cosines = {}
        for name, axis in probe_axes.items():
            cos = float(np.dot(da1_norm, axis))
            cosines[name] = cos

        # Per-sample Spearman
        Xc = clip_embeddings - mu_clip
        da1_scores = Xc @ da1
        correlations = {}
        for name, axis in probe_axes.items():
            probe_scores = clip_embeddings @ axis
            rho, pval = spearmanr(da1_scores, probe_scores)
            correlations[name] = {"rho": float(rho), "pvalue": float(pval)}

        print("  Cosines with text probes:")
        for name in probe_axes:
            print(f"    {name:<15} cos = {cosines[name]:>7.4f}")
        print("  Spearman correlations:")
        for name in probe_axes:
            r = correlations[name]
            print(f"    {name:<15} rho = {r['rho']:>7.4f} (p={r['pvalue']:.2e})")

        all_cosines[label] = cosines
        all_correlations[label] = correlations

    # --- PC1 analysis ---
    print("\n=== PC1 Correlations ===")
    Xc = clip_embeddings - mu_clip
    cov = Xc.T @ Xc / (N - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1].copy()
    pc1 = eigvecs[:, idx[0]]
    pc2 = eigvecs[:, idx[1]]
    var_explained_1 = eigvals[idx[0]] / eigvals.sum() * 100
    var_explained_2 = eigvals[idx[1]] / eigvals.sum() * 100
    print(f"  PC1 var explained: {var_explained_1:.1f}%")
    print(f"  PC2 var explained: {var_explained_2:.1f}%")

    pc1_scores = Xc @ pc1
    pc2_scores = Xc @ pc2

    pc1_cosines = {}
    pc1_correlations = {}
    pc2_cosines = {}
    pc2_correlations = {}
    for name, axis in probe_axes.items():
        pc1_cosines[name] = float(np.dot(pc1 / np.linalg.norm(pc1), axis))
        pc2_cosines[name] = float(np.dot(pc2 / np.linalg.norm(pc2), axis))
        probe_scores = clip_embeddings @ axis
        rho1, p1 = spearmanr(pc1_scores, probe_scores)
        rho2, p2 = spearmanr(pc2_scores, probe_scores)
        pc1_correlations[name] = {"rho": float(rho1), "pvalue": float(p1)}
        pc2_correlations[name] = {"rho": float(rho2), "pvalue": float(p2)}

    print("  PC1 cosines:")
    for name in probe_axes:
        print(f"    {name:<15} cos = {pc1_cosines[name]:>7.4f}")
    print("  PC1 Spearman:")
    for name in probe_axes:
        r = pc1_correlations[name]
        print(f"    {name:<15} rho = {r['rho']:>7.4f} (p={r['pvalue']:.2e})")
    print("  PC2 cosines:")
    for name in probe_axes:
        print(f"    {name:<15} cos = {pc2_cosines[name]:>7.4f}")
    print("  PC2 Spearman:")
    for name in probe_axes:
        r = pc2_correlations[name]
        print(f"    {name:<15} rho = {r['rho']:>7.4f} (p={r['pvalue']:.2e})")

    # DA1 vs PC1 alignment
    for label, da1 in da1_directions.items():
        cos_da1_pc1 = abs(float(np.dot(da1 / np.linalg.norm(da1), pc1 / np.linalg.norm(pc1))))
        print(f"  |cos(DA1_{label}, PC1)| = {cos_da1_pc1:.4f}")

    # --- Summary table ---
    print("\n=== Summary: Gender Correlation ===")
    print(f"{'Direction':>15s}  {'τ*':>6s}  {'cos(gender)':>12s}  {'ρ(gender)':>10s}")
    print("-" * 55)
    for label in da1_directions:
        tau = tau_values[label]
        cos_g = all_cosines[label]['gender']
        rho_g = all_correlations[label]['gender']['rho']
        print(f"{'DA1_'+label:>15s}  {tau:>6d}  {cos_g:>12.4f}  {rho_g:>10.4f}")
    print(f"{'PC1':>15s}  {'—':>6s}  {pc1_cosines['gender']:>12.4f}  {pc1_correlations['gender']['rho']:>10.4f}")
    print(f"{'PC2':>15s}  {'—':>6s}  {pc2_cosines['gender']:>12.4f}  {pc2_correlations['gender']['rho']:>10.4f}")

    save_results(da1_directions, tau_values, all_cosines, all_correlations,
                 pc1_cosines, pc1_correlations, pc2_cosines, pc2_correlations)


def save_results(da1_directions, tau_values, all_cosines, all_correlations,
                 pc1_cosines=None, pc1_correlations=None,
                 pc2_cosines=None, pc2_correlations=None):
    results = {
        'experiment': 'sdxl_square_N1000_da1_stopping_comparison',
        'tau_values': {k: int(v) for k, v in tau_values.items()},
        'direction_cosines': {},
        'probe_cosines': {k: v for k, v in all_cosines.items()},
        'probe_correlations': {k: v for k, v in all_correlations.items()},
    }
    if pc1_cosines is not None:
        results['pc1_cosines'] = pc1_cosines
        results['pc1_correlations'] = pc1_correlations
        results['pc2_cosines'] = pc2_cosines
        results['pc2_correlations'] = pc2_correlations

    # Pairwise direction cosines
    labels = list(da1_directions.keys())
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            d1 = da1_directions[labels[i]]
            d2 = da1_directions[labels[j]]
            cos = float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))
            results['direction_cosines'][f"{labels[i]}_vs_{labels[j]}"] = {
                'cosine': cos,
                'abs_cosine': abs(cos),
                'angle_deg': float(np.degrees(np.arccos(np.clip(abs(cos), 0, 1)))),
            }

    json_path = os.path.join(OUTDIR, 'da1_stopping_comparison_N1000.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {json_path}")


if __name__ == '__main__':
    main()
