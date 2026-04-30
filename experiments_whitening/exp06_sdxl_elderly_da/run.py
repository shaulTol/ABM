#!/usr/bin/env python3
"""
Exp 6: SDXL Elderly DA — whitened variant.

1. Load N=2000 elderly gs=7.5 x0 latents → VAE decode → CLIP embed → z0 (2000, 768)
2. Compute whitening W = Sigma_hat^{-1/2} from z0 (LW shrinkage)
3. Load elderly Langevin checkpoint (N=200, ~113 snapshots)
4. CLIP embed x0 + all snapshots → whiten with W → compute DA + SSA
5. Compare whitened DA1 with original DA1

No Langevin re-run, no UNet. Only VAE + CLIP.

Usage (cluster, salmon):
    python experiments_whitening/exp06_sdxl_elderly_da/run.py \
        --elderly_x0_2000 /path/to/gen_mp_p5_elderly_gs7p5_x0.npz \
        --elderly_ckpt /path/to/langevin_ckpt_elderly_gs7p5_fc_dt0p0025.npz \
        --output_dir /path/to/output
"""
import sys, os, json, argparse, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'sdxl', 'scripts'))

import torch
from experiments_whitening.shared.whitening import WhiteningTransform

MODEL_ID = 'stabilityai/stable-diffusion-xl-base-1.0'
LATENT_SHAPE = (4, 128, 128)
DT = 0.0025  # elderly canonical dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--elderly_x0_2000', required=True,
                        help='N=2000 x0 latents for whitening matrix')
    parser.add_argument('--elderly_ckpt', required=True,
                        help='Langevin checkpoint with x0_flat + snapshots')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 1. Load VAE + CLIP ──
    from diffusers import AutoencoderKL
    from sdxl_components import RealCLIPEmbedder, compute_ssa_da

    print("=== Loading VAE ===")
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float16,
    ).to(device).float().eval()

    print("=== Loading CLIP ===")
    clip_embedder = RealCLIPEmbedder(
        vae=vae, device=device, vae_batch_size=4, clip_batch_size=8,
    )

    # ── 2. CLIP embed N=2000 x0 → whitening W ──
    print("\n=== CLIP embedding N=2000 x0 for whitening ===")
    t0 = time.time()
    x0_2000 = np.load(args.elderly_x0_2000)['x0']
    print(f"  x0 shape: {x0_2000.shape}")
    z0_2000 = clip_embedder.embed(x0_2000, latent_shape=LATENT_SHAPE)
    print(f"  z0_2000: {z0_2000.shape}, took {time.time()-t0:.1f}s")

    wt = WhiteningTransform(z0_2000, use_ledoit_wolf=True)
    diag = wt.diagnostics()
    print(f"  LW alpha={diag['alpha_lw']:.4f}, "
          f"cond={diag['cond_before']:.1f} -> {diag['cond_after']:.1f}")

    # ── 3. Load checkpoint, CLIP embed N=200 x0 + snapshots ──
    print(f"\n=== Loading Langevin checkpoint: {args.elderly_ckpt} ===")
    ckpt = np.load(args.elderly_ckpt, allow_pickle=True)
    x0_200 = ckpt['x0_flat']
    snapshot_lags = sorted(
        int(k.split('_')[1]) for k in ckpt.files
        if k.startswith('snapshot_') and k != 'snapshot_lags'
        and k != 'snapshot_dtype_on_disk'
    )
    print(f"  x0: {x0_200.shape}, {len(snapshot_lags)} snapshots")

    print("\n=== CLIP embedding x0 (N=200) ===")
    z0_np = clip_embedder.embed(x0_200, latent_shape=LATENT_SHAPE)

    print(f"\n=== CLIP embedding {len(snapshot_lags)} snapshots ===")
    z_snaps = {}
    z_snaps_w = {}
    t0 = time.time()
    for i, lag in enumerate(snapshot_lags):
        snap = ckpt[f'snapshot_{lag}'].astype(np.float32)
        z = clip_embedder.embed(snap, latent_shape=LATENT_SHAPE)
        z_snaps[lag] = z
        z_snaps_w[lag] = wt.whiten(z)
        if (i + 1) % 20 == 0 or lag == snapshot_lags[-1]:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(snapshot_lags) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(snapshot_lags)}] lag={lag}, "
                  f"{rate:.2f} snaps/s, ETA {eta/60:.1f}min")

    z0_w = wt.whiten(z0_np)

    # ── 4. Compute DA — whitened ──
    print("\n=== Computing whitened DA ===")
    da_w = compute_ssa_da(z0_w, z_snaps_w, snapshot_lags,
                          n_eigvals=10, n_eigvecs=5,
                          ssa_min_lag=0, ssa_only=False, dt=DT)
    print(f"  SSA_whitened = {da_w['ssa']:.4f}")

    # ── 5. Compute DA — original ──
    print("\n=== Computing original DA ===")
    da_o = compute_ssa_da(z0_np, z_snaps, snapshot_lags,
                          n_eigvals=10, n_eigvecs=5,
                          ssa_min_lag=0, ssa_only=False, dt=DT)
    print(f"  SSA_original = {da_o['ssa']:.4f}")

    # ── 6. Compare DA1 directions ──
    cos_sim = None
    if da_w.get('da1_eigvec') is not None and da_o.get('da1_eigvec') is not None:
        da1_tilde = da_w['da1_eigvec']
        da1_orig = da_o['da1_eigvec']
        # Back-project whitened DA1 to original CLIP space
        da1_unwhitened = wt.W_inv @ da1_tilde
        da1_unwhitened /= np.linalg.norm(da1_unwhitened)
        cos_sim = float(abs(np.dot(da1_unwhitened, da1_orig)))
        print(f"  |cos(DA1_whitened->orig, DA1_original)| = {cos_sim:.3f}")

    gap_w = da_w.get('gap12_profile')
    gap_o = da_o.get('gap12_profile')
    if gap_w is not None:
        idx = np.argmax(gap_w)
        print(f"  Whitened best gap12 = {gap_w[idx]:.2f}x at lag {snapshot_lags[idx]}")
    if gap_o is not None:
        idx = np.argmax(gap_o)
        print(f"  Original best gap12 = {gap_o[idx]:.2f}x at lag {snapshot_lags[idx]}")

    # ── 7. Save ──
    results = {
        'ssa_whitened': float(da_w['ssa']),
        'ssa_original': float(da_o['ssa']),
        'da1_cos_similarity': cos_sim,
        'whitening_diagnostics': diag,
        'n_snapshots': len(snapshot_lags),
        'N_whitening': int(z0_2000.shape[0]),
        'N_trajectories': int(z0_np.shape[0]),
        'D_clip': int(z0_np.shape[1]),
        'dt': DT,
    }
    out_json = os.path.join(args.output_dir, 'elderly_da_whitened.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")

    save_data = {
        'W': wt.W, 'W_inv': wt.W_inv, 'mu': wt.mu,
        'z0_clip': z0_np,
        'rho_sq_whitened': da_w.get('rho_sq', np.array([])),
        'rho_sq_original': da_o.get('rho_sq', np.array([])),
    }
    if da_w.get('da1_eigvec') is not None:
        save_data['da1_whitened'] = da_w['da1_eigvec']
    if da_o.get('da1_eigvec') is not None:
        save_data['da1_original'] = da_o['da1_eigvec']
    np.savez(os.path.join(args.output_dir, 'elderly_da_whitened.npz'), **save_data)
    wt.save(os.path.join(args.output_dir, 'W_elderly_gs7p5.npz'))
    print("Done!")


if __name__ == '__main__':
    main()
