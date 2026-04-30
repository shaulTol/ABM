#!/usr/bin/env python3
"""
Exp 5: SDXL Guidance Sweep — whitened SSA from existing checkpoints.

For each (prompt, guidance_scale) cell:
  1. Load N=2000 x0 latents → VAE decode → CLIP embed → z0 (2000, 768)
  2. Compute per-cell whitening W = Sigma_hat^{-1/2} from z0 (LW shrinkage)
  3. Load Langevin checkpoint (N=200 trajectories, ~112 snapshots)
  4. CLIP embed x0 + all snapshots → whiten with W → compute SSA

No Langevin re-run, no UNet. Only VAE + CLIP.

Usage (cluster, salmon):
    python experiments_whitening/exp05_sdxl_guidance/run_whitened_langevin.py \
        --cell_idx 0 \
        --x0_dir /path/to/results_multiprompt_sweep \
        --guidance_sweep_dir /path/to/results_guidance_sweep \
        --output_dir /path/to/output
"""
import sys, os, json, argparse, time, glob
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'sdxl', 'scripts'))

import torch
from experiments_whitening.shared.whitening import WhiteningTransform

# Cell grid: 3 prompts × 4 guidance scales
PROMPTS = ['person', 'cat', 'elderly']
GUIDANCE_SCALES = [1.0, 3.0, 7.5, 15.0]
CELLS = [(p, gs) for p in PROMPTS for gs in GUIDANCE_SCALES]

MODEL_ID = 'stabilityai/stable-diffusion-xl-base-1.0'
LATENT_SHAPE = (4, 128, 128)
DT = 0.005  # matches canonical SDXL sweep


# --- File lookup helpers ---

# Map prompt → x0 file pattern in multiprompt_sweep
PROMPT_X0_PREFIX = {
    'person': None,  # person is in guidance_sweep, not multiprompt
    'cat': 'gen_mp_p0_cat',
    'elderly': 'gen_mp_p5_elderly',
}

# Map prompt → checkpoint prefix in multiprompt_sweep
PROMPT_CKPT_PREFIX = {
    'person': None,  # person checkpoints in guidance_sweep
    'cat': 'langevin_ckpt_mp_p0_cat',
    'elderly': 'langevin_ckpt_mp_p5_elderly',
}


def gs_str(gs):
    return f"{gs:.1f}".replace('.', 'p')


def find_x0_2000(x0_dir, guidance_sweep_dir, prompt, gs):
    """Find N=2000 x0 latent file (NOT the quick N=200 version)."""
    gss = gs_str(gs)
    if prompt == 'person':
        # Person is in guidance_sweep dir
        path = os.path.join(guidance_sweep_dir, f'gen_sweep_gs{gss}_x0.npz')
    else:
        prefix = PROMPT_X0_PREFIX[prompt]
        path = os.path.join(x0_dir, f'{prefix}_gs{gss}_x0.npz')
    if os.path.exists(path):
        return path
    return None


def find_langevin_ckpt(x0_dir, guidance_sweep_dir, prompt, gs):
    """Find Langevin checkpoint (N=200 trajectories + snapshots)."""
    gss = gs_str(gs)
    if prompt == 'person':
        path = os.path.join(guidance_sweep_dir, f'langevin_ckpt_sweep_gs{gss}.npz')
    else:
        prefix = PROMPT_CKPT_PREFIX[prompt]
        path = os.path.join(x0_dir, f'{prefix}_gs{gss}.npz')
    if os.path.exists(path):
        return path
    return None


def load_vae_and_clip(device):
    """Load only VAE + CLIP (no UNet)."""
    from diffusers import AutoencoderKL
    from sdxl_components import RealCLIPEmbedder

    print("=== Loading VAE ===")
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID, subfolder="vae",
        torch_dtype=torch.float16,
    ).to(device).float().eval()

    print("=== Loading CLIP ===")
    clip_embedder = RealCLIPEmbedder(
        vae=vae, device=device, vae_batch_size=4, clip_batch_size=8,
    )
    return clip_embedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_idx', type=int, required=True)
    parser.add_argument('--x0_dir', required=True,
                        help='multiprompt_sweep dir with x0 + checkpoints')
    parser.add_argument('--guidance_sweep_dir', required=True,
                        help='guidance_sweep dir (person prompt)')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    prompt, gs = CELLS[args.cell_idx]
    print(f"Cell {args.cell_idx}: prompt={prompt}, gs={gs}")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 1. Locate files ──
    x0_path = find_x0_2000(args.x0_dir, args.guidance_sweep_dir, prompt, gs)
    ckpt_path = find_langevin_ckpt(args.x0_dir, args.guidance_sweep_dir, prompt, gs)
    if x0_path is None:
        print(f"ERROR: N=2000 x0 not found for {prompt} gs={gs}")
        sys.exit(1)
    if ckpt_path is None:
        print(f"ERROR: Langevin checkpoint not found for {prompt} gs={gs}")
        sys.exit(1)
    print(f"  x0 (N=2000): {x0_path}")
    print(f"  checkpoint:   {ckpt_path}")

    # ── 2. Load VAE + CLIP ──
    clip_embedder = load_vae_and_clip(device)

    # ── 3. CLIP embed N=2000 x0 → compute whitening W ──
    print("\n=== CLIP embedding N=2000 x0 for whitening ===")
    t0 = time.time()
    x0_2000 = np.load(x0_path)['x0']
    print(f"  x0 shape: {x0_2000.shape}")
    z0_2000 = clip_embedder.embed(x0_2000, latent_shape=LATENT_SHAPE)
    print(f"  z0_2000: {z0_2000.shape}, took {time.time()-t0:.1f}s")

    wt = WhiteningTransform(z0_2000, use_ledoit_wolf=True)
    diag = wt.diagnostics()
    print(f"  LW alpha={diag['alpha_lw']:.4f}, "
          f"cond={diag['cond_before']:.1f} -> {diag['cond_after']:.1f}")

    # ── 4. Load checkpoint, CLIP embed N=200 x0 + snapshots ──
    print("\n=== Loading Langevin checkpoint ===")
    ckpt = np.load(ckpt_path, allow_pickle=True)
    x0_200 = ckpt['x0_flat']  # (200, 65536)
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

    # ── 5. Compute SSA — whitened + original ──
    from sdxl_components import compute_ssa_da

    print("\n=== Computing whitened SSA ===")
    res_w = compute_ssa_da(z0_w, z_snaps_w, snapshot_lags,
                           n_eigvals=10, n_eigvecs=3,
                           ssa_min_lag=0, ssa_only=True, dt=DT)
    print(f"  SSA_whitened = {res_w['ssa']:.4f}")

    print("=== Computing original SSA ===")
    res_o = compute_ssa_da(z0_np, z_snaps, snapshot_lags,
                           n_eigvals=10, n_eigvecs=3,
                           ssa_min_lag=0, ssa_only=True, dt=DT)
    print(f"  SSA_original = {res_o['ssa']:.4f}")

    # ── 6. Save ──
    gss = gs_str(gs)
    results = {
        'prompt': prompt,
        'guidance_scale': gs,
        'ssa_whitened': float(res_w['ssa']),
        'ssa_original': float(res_o['ssa']),
        'ratio': float(res_w['ssa'] / res_o['ssa']) if res_o['ssa'] > 0 else None,
        'whitening_diagnostics': diag,
        'dt': DT,
        'N_whitening': int(z0_2000.shape[0]),
        'N_trajectories': int(z0_np.shape[0]),
        'n_snapshots': len(snapshot_lags),
    }
    out_json = os.path.join(args.output_dir, f'whitened_{prompt}_gs{gss}.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")

    np.savez(
        os.path.join(args.output_dir, f'whitened_{prompt}_gs{gss}.npz'),
        rho_sq_whitened=res_w.get('rho_sq', np.array([])),
        rho_sq_original=res_o.get('rho_sq', np.array([])),
        W=wt.W, W_inv=wt.W_inv, mu=wt.mu,
    )
    wt.save(os.path.join(args.output_dir, f'W_{prompt}_gs{gss}.npz'))
    print("Done!")


if __name__ == '__main__':
    main()
