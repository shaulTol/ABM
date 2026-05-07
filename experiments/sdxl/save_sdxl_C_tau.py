#!/usr/bin/env python3
"""
Save full autocovariance matrices C_sym(tau) for SDXL elderly experiment.

Loads Langevin checkpoint (or re-runs from x0), embeds via VAE+CLIP,
computes full 768x768 C_tau_sym at each of ~111 snapshot lags.

Usage:
    # N=200 (resume from checkpoint):
    python save_sdxl_C_tau.py --n_samples 200 \
        --resume_checkpoint results/langevin_ckpt_mp_p5_elderly_gs7p5.npz

    # N=1000 (re-run Langevin from x0):
    python save_sdxl_C_tau.py --n_samples 1000 \
        --resume_x0 results/gen_mp_p5_elderly_gs7p5_x0.npz

Output: results/elderly_gs7p5_n{N}_C_tau_full.npz
"""
import sys, os
import argparse
import time
import gc

import numpy as np
import torch

# Allow imports from project root and own directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from sdxl_components import RealCLIPEmbedder
from run_ssa_da_clip import compute_snapshot_lags
from compute_sdxl_ssa_v2 import SDXLScore, run_langevin_latent_space

RESDIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESDIR, exist_ok=True)

# Fixed parameters matching run_da_elderly_person.sh
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "a portrait of an elderly person"
GUIDANCE_SCALE = 7.5
GEN_STEPS = 50
HEIGHT = WIDTH = 1024
LATENT_SHAPE = (4, 128, 128)
D_LATENT = 4 * 128 * 128  # 65536
T = 2000
DT = 0.0005  # default; override with --dt
T_SCORE = 1
SNAPSHOT_SCHEDULE = "log_and_fixed"
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, required=True, help="200 or 1000")
    p.add_argument("--resume_checkpoint", type=str, default=None,
                   help="Langevin checkpoint .npz (has x0 + all snapshots)")
    p.add_argument("--resume_x0", type=str, default=None,
                   help="x0 file .npz (re-runs Langevin)")
    p.add_argument("--clip_batch_size", type=int, default=4)
    p.add_argument("--vae_batch_size", type=int, default=4)
    p.add_argument("--score_batch_size", type=int, default=16)
    p.add_argument("--langevin_ckpt_interval", type=int, default=500,
                   help="Save intermediate Langevin checkpoint every N steps (N=1000 safety)")
    p.add_argument("--dt", type=float, default=None,
                   help="Langevin dt (default: use DT constant)")
    return p.parse_args()


def load_from_checkpoint(ckpt_path, n_samples):
    """Load x0 and snapshots from Langevin checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = np.load(ckpt_path, allow_pickle=True)
    x0 = ckpt["x0_flat"][:n_samples]  # (N, D_latent)
    snapshot_lags = ckpt["snapshot_lags"].tolist()
    var_f = float(ckpt["var_f"])

    snapshots = {}
    for lag in snapshot_lags:
        key = f"snapshot_{lag}"
        if key in ckpt:
            snap = ckpt[key][:n_samples].astype(np.float32)
            snapshots[lag] = snap
    print(f"  x0: {x0.shape}, {len(snapshots)} snapshots, var_f={var_f:.6f}")
    return x0, snapshots, snapshot_lags, var_f


def run_langevin_from_x0(x0_path, n_samples, args, device):
    """Load x0, build SDXL pipeline, run Langevin with periodic checkpointing."""
    print(f"Loading x0: {x0_path}")
    x0_data = np.load(x0_path)
    x0_key = "x0" if "x0" in x0_data else "x0_flat"
    x0_all = x0_data[x0_key]
    print(f"  x0 shape: {x0_all.shape}, using first {n_samples}")
    x0_all = x0_all[:n_samples]

    samples_flat = torch.from_numpy(x0_all).float()
    var_f = samples_flat.var(dim=0).mean().item()
    print(f"  var_f = {var_f:.6f}")

    snapshot_lags = compute_snapshot_lags(T, SNAPSHOT_SCHEDULE)
    print(f"  Snapshot schedule: {len(snapshot_lags)} lags")

    # Build SDXL score function
    print("\n=== Loading SDXL pipeline ===")
    from diffusers import StableDiffusionXLPipeline, DDPMScheduler
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe = pipe.to(device)

    # Encode prompt
    prompt_out = pipe.encode_prompt(PROMPT, device=device, num_images_per_prompt=1,
                                     do_classifier_free_guidance=(GUIDANCE_SCALE > 1.0))
    if GUIDANCE_SCALE > 1.0:
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = prompt_out
    else:
        prompt_embeds, _, pooled_prompt_embeds, _ = prompt_out
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None

    # Time IDs for micro-conditioning (shape (1,6) — SDXLScore expands internally)
    time_ids = torch.tensor([[HEIGHT, WIDTH, 0, 0, HEIGHT, WIDTH]],
                            dtype=torch.float16, device=device)

    scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    unet = pipe.unet.eval()

    score_fn = SDXLScore(
        unet=unet, scheduler=scheduler, t=T_SCORE,
        prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
        time_ids=time_ids, device=device, latent_shape=LATENT_SHAPE,
        batch_size=args.score_batch_size, guidance_scale=GUIDANCE_SCALE,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    )

    # Run Langevin with periodic checkpointing for safety
    print(f"\n=== Langevin ({n_samples} traj x {T} steps, dt={DT}) ===")
    ckpt_interval = args.langevin_ckpt_interval
    temp_ckpt_path = os.path.join(RESDIR, f"_langevin_temp_n{n_samples}.npz")

    # Check for existing temp checkpoint to resume from
    start_step = 0
    snapshots_so_far = {}
    if os.path.exists(temp_ckpt_path):
        print(f"  Found temp checkpoint, attempting resume...")
        try:
            temp = np.load(temp_ckpt_path, allow_pickle=True)
            start_step = int(temp["last_step"])
            x_state = torch.from_numpy(temp["x_state"]).float().to(device)
            x0_flat_saved = torch.from_numpy(temp["x0_flat"]).float()
            for key in temp.files:
                if key.startswith("snapshot_"):
                    lag = int(key.split("_")[1])
                    snapshots_so_far[lag] = temp[key]
            print(f"  Resumed from step {start_step}, {len(snapshots_so_far)} snapshots cached")
        except Exception as e:
            print(f"  Failed to load temp checkpoint: {e}, starting fresh")
            start_step = 0
            snapshots_so_far = {}

    if start_step == 0:
        # Fresh start
        d = samples_flat.shape[1]
        sigma_f = np.sqrt(var_f)
        sqrt_dt = np.sqrt(DT)

        torch.manual_seed(SEED)
        idx = torch.randint(0, samples_flat.shape[0], (n_samples,))
        x = samples_flat[idx].to(device)
        x0_flat_saved = x.clone().cpu()
    else:
        x = x_state
        d = x.shape[1]
        sigma_f = np.sqrt(var_f)
        sqrt_dt = np.sqrt(DT)

    snap_set = set(snapshot_lags)
    t_start = time.time()

    for t in range(start_step + 1, T + 1):
        grad = score_fn(x)
        drift = 0.5 * var_f * DT * grad
        noise = sigma_f * sqrt_dt * torch.randn_like(x)
        x = x + drift + noise

        if t in snap_set:
            snapshots_so_far[t] = x.clone().cpu().numpy()

        if t % 100 == 0 or t == T:
            elapsed = time.time() - t_start
            actual_steps = t - start_step
            rate = actual_steps / elapsed if elapsed > 0 else 0
            eta = (T - t) / rate / 3600 if rate > 0 else 0
            print(f"    step {t}/{T}, ||x||={x.norm(dim=-1).mean():.2f}, "
                  f"{rate:.2f} steps/s, ETA {eta:.1f}h", flush=True)

        # Periodic checkpoint
        if ckpt_interval > 0 and t % ckpt_interval == 0 and t < T:
            ckpt_data = {
                "x_state": x.cpu().numpy().astype(np.float16),
                "x0_flat": x0_flat_saved.numpy(),
                "last_step": np.array(t),
            }
            for lag, snap in snapshots_so_far.items():
                ckpt_data[f"snapshot_{lag}"] = snap.astype(np.float16) if isinstance(snap, np.ndarray) else snap.numpy().astype(np.float16)
            np.savez_compressed(temp_ckpt_path, **ckpt_data)
            print(f"    Saved temp checkpoint at step {t}", flush=True)

    # Convert remaining torch tensors to numpy
    snapshots_np = {}
    for lag, snap in snapshots_so_far.items():
        if isinstance(snap, torch.Tensor):
            snapshots_np[lag] = snap.numpy().astype(np.float32)
        else:
            snapshots_np[lag] = snap.astype(np.float32)

    # Clean up temp checkpoint
    if os.path.exists(temp_ckpt_path):
        os.remove(temp_ckpt_path)
        print(f"  Removed temp checkpoint")

    # Free GPU
    del score_fn, unet
    vae = pipe.vae.to(device).float().eval()
    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    return x0_flat_saved.numpy(), snapshots_np, snapshot_lags, var_f, vae


def embed_all_clips(vae, x0, snapshots, snapshot_lags, args, device):
    """Embed x0 and all snapshots to CLIP space."""
    print("\n=== CLIP embedding ===")
    embedder = RealCLIPEmbedder(vae, device=str(device),
                                 vae_batch_size=args.vae_batch_size,
                                 clip_batch_size=args.clip_batch_size)

    N = x0.shape[0]
    D_clip = embedder.embed_dim
    print(f"  N={N}, D_clip={D_clip}, {len(snapshot_lags)} snapshots")

    # Embed x0
    print("  Embedding x0...")
    z0 = embedder.embed(x0, latent_shape=LATENT_SHAPE)
    print(f"    z0: {z0.shape}, ||z0||_mean={np.linalg.norm(z0, axis=1).mean():.4f}")

    # Embed each snapshot
    z_snapshots = {}
    for i, lag in enumerate(snapshot_lags):
        if lag not in snapshots:
            continue
        snap = snapshots[lag]
        if snap.dtype == np.float16:
            snap = snap.astype(np.float32)
        z = embedder.embed(snap, latent_shape=LATENT_SHAPE)
        z_snapshots[lag] = z
        if (i + 1) % 20 == 0 or i == 0 or (i + 1) == len(snapshot_lags):
            print(f"    [{i+1}/{len(snapshot_lags)}] lag={lag}", flush=True)

    # Free embedder GPU memory
    del embedder
    torch.cuda.empty_cache()
    gc.collect()

    return z0, z_snapshots


def compute_C_tau_matrices(z0, z_snapshots, snapshot_lags):
    """Compute full C_tau_sym matrices from CLIP embeddings."""
    N, D = z0.shape
    mu = z0.mean(axis=0)
    z0_c = z0 - mu

    # C(0)
    C0 = (z0_c.T @ z0_c) / N

    used_lags = sorted(lag for lag in snapshot_lags if lag in z_snapshots)
    L = len(used_lags)
    print(f"\n=== Computing C_tau matrices: {L} lags, {D}x{D} ===")

    C_tau_all = np.zeros((L, D, D), dtype=np.float64)
    eigenvalues_all = np.zeros((L, D))
    rho_sq_all = np.zeros(L)
    tr_C0 = np.trace(C0)

    for i, lag in enumerate(used_lags):
        z_tau_c = z_snapshots[lag] - mu
        C_tau = (z_tau_c.T @ z0_c) / N
        C_tau_sym = 0.5 * (C_tau + C_tau.T)
        C_tau_all[i] = C_tau_sym

        # Eigenvalues for consistency check
        eigvals = np.linalg.eigvalsh(C_tau_sym)[::-1].copy()
        eigenvalues_all[i] = eigvals

        # rho^2 for SSA
        tr_C_tau = np.trace(C_tau)
        rho_sq_all[i] = (tr_C_tau / tr_C0) ** 2

        if (i + 1) % 20 == 0 or i == 0 or (i + 1) == L:
            sym_err = np.max(np.abs(C_tau_sym - C_tau_sym.T))
            print(f"  [{i+1:3d}/{L}] lag={lag:>5d}  "
                  f"lambda_1={eigvals[0]:.6f}  rho^2={rho_sq_all[i]:.6f}  "
                  f"sym_err={sym_err:.2e}", flush=True)

    return C0, C_tau_all, eigenvalues_all, rho_sq_all, np.array(used_lags), mu


def main():
    global DT
    args = parse_args()
    if args.dt is not None:
        DT = args.dt
    N = args.n_samples
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    print(f"=== Save SDXL C_tau: elderly gs=7.5, N={N}, dt={DT} ===")
    print(f"  Device: {device}")

    vae = None

    if args.resume_checkpoint:
        x0, snapshots, snapshot_lags, var_f = load_from_checkpoint(
            args.resume_checkpoint, N)

        # Load VAE separately
        print("\n=== Loading SDXL VAE ===")
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        vae = pipe.vae.to(device).float().eval()
        del pipe.unet, pipe
        torch.cuda.empty_cache()

    elif args.resume_x0:
        x0, snapshots, snapshot_lags, var_f, vae = run_langevin_from_x0(
            args.resume_x0, N, args, device)

    else:
        print("ERROR: must provide --resume_checkpoint or --resume_x0")
        sys.exit(1)

    # CLIP embedding
    z0, z_snapshots = embed_all_clips(vae, x0, snapshots, snapshot_lags, args, device)

    # Free latent-space data
    del x0, snapshots, vae
    gc.collect()

    # Compute C_tau matrices
    C0, C_tau_all, eigenvalues, rho_sq, used_lags, mu = compute_C_tau_matrices(
        z0, z_snapshots, snapshot_lags)

    # Save
    D_clip = z0.shape[1]
    phys_times = used_lags.astype(float) * DT
    sigma_f_sq = float(np.trace(C0) / D_clip)

    dt_tag = f"_dt{DT}".replace(".", "p")
    out_path = os.path.join(RESDIR, f"elderly_gs7p5_n{N}{dt_tag}_C_tau_full.npz")
    print(f"\n=== Saving to {out_path} ===")

    np.savez_compressed(out_path,
        lags=used_lags,
        phys_times=phys_times,
        C_tau=C_tau_all,
        C0=C0,
        eigenvalues=eigenvalues,
        rho_sq=rho_sq,
        mu=mu,
        N=np.array(N),
        d=np.array(D_clip),
        sigma_f_sq=np.array(sigma_f_sq),
        dt=np.array(DT),
        T=np.array(T),
        prompt=np.array(PROMPT),
        guidance_scale=np.array(GUIDANCE_SCALE),
        model_id=np.array(MODEL_ID),
        snapshot_schedule=np.array(SNAPSHOT_SCHEDULE),
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved: {size_mb:.1f} MB")

    # Verification
    L = len(used_lags)
    print(f"\n  Verification:")
    print(f"    C_tau shape: {C_tau_all.shape} (expected ({L}, {D_clip}, {D_clip}))")
    print(f"    C0 shape: {C0.shape}")
    max_sym_err = max(np.max(np.abs(C_tau_all[i] - C_tau_all[i].T)) for i in range(L))
    print(f"    Max symmetry error: {max_sym_err:.2e}")
    print(f"    tr(C0) = {np.trace(C0):.6f}, sigma_f^2 * d = {sigma_f_sq * D_clip:.6f}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} min ({elapsed/3600:.1f}h)")
    print("DONE")


if __name__ == "__main__":
    main()
