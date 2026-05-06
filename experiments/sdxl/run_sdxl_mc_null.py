#!/usr/bin/env python3
"""
MC Null Test for SDXL Elderly DA Directions (cluster array job).

Compares observed DA eigenvalue at τ* against a pipeline-matched Gaussian null.
The null model:
1. Samples latents from N(0, Σ_hat_latent) via SVD of the observed x0 latents
2. Runs Langevin with the SAME SDXLScore instance (same prompt, gs, t=1)
3. VAE decodes + CLIP embeds at t=0 and t=τ*
4. Computes C(τ*), eigendecomposes, stores top eigenvalues

IMPORTANT: Each replicate takes ~19 hours of Langevin on L40S (200 traj × 2000 steps).
We use B=3 total (1 per job). Each job has --time=2-00:00:00 (48h).

The SDXLScore is held FIXED across all replicates — only the initial latents differ.
"""
import sys
import os
import time
import json
import argparse
import numpy as np
import torch

# Add own directory and project root (2 levels up) to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from compute_sdxl_ssa_v2 import SDXLScore, run_langevin_latent_space
from sdxl_components import RealCLIPEmbedder


def build_sdxl_score(device, guidance_scale=7.5, t_score=1, latent_shape=(4, 128, 128),
                     score_batch_size=16):
    """Build SDXLScore for 'a portrait of an elderly person' at gs=7.5."""
    from diffusers import StableDiffusionXLPipeline, DDPMScheduler
    import torch

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    print(f"Loading SDXL: {model_id}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to(device)

    # Encode prompt
    prompt = "a portrait of an elderly person"
    negative_prompt = ""
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, \
        negative_pooled_prompt_embeds = pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            device=device, num_images_per_prompt=1, do_classifier_free_guidance=True)

    # Micro-conditioning time_ids (MUST be 1024×1024)
    time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]],
                            dtype=torch.float16, device=device)

    # Scheduler for alpha_bar
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    unet = pipe.unet
    # Free pipeline memory (keep UNet)
    del pipe.vae, pipe.text_encoder, pipe.text_encoder_2
    del pipe.tokenizer, pipe.tokenizer_2
    import gc; gc.collect(); torch.cuda.empty_cache()

    score_fn = SDXLScore(
        unet=unet, scheduler=scheduler, t=t_score,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        time_ids=time_ids,
        device=device, latent_shape=latent_shape,
        batch_size=score_batch_size,
        guidance_scale=guidance_scale,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    )
    return score_fn


def sample_null_latents(x0_np, n_samples, seed):
    """Sample from N(0, Σ_hat) via SVD of centered x0.

    x0: (N_orig, d) original latents. We compute SVD of centered x0,
    then sample: x_null = V^T @ diag(s/sqrt(N)) @ z for z ~ N(0, I_N).
    """
    N_orig, d = x0_np.shape
    mu = x0_np.mean(axis=0)
    x_c = x0_np - mu  # (N_orig, d)

    # Economy SVD: x_c = U @ diag(s) @ Vt, where U is (N,N), s is (N,), Vt is (N,d)
    U, s, Vt = np.linalg.svd(x_c, full_matrices=False)  # s has N_orig values
    # Σ_hat = Vt.T @ diag(s²/(N-1)) @ Vt
    # Sample: x_null = mu + Vt.T @ diag(s/sqrt(N-1)) @ z
    scale = s / np.sqrt(N_orig - 1)  # (N_orig,)

    rng = np.random.RandomState(seed)
    z = rng.randn(n_samples, N_orig)  # (n_samples, N_orig)
    x_null = mu + (z * scale[None, :]) @ Vt  # (n_samples, d)
    return x_null.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=None)
    parser.add_argument('--reps_per_task', type=int, default=1,
                        help='Replicates per task (default 1; each takes ~19h)')
    parser.add_argument('--N', type=int, default=200)
    parser.add_argument('--T', type=int, default=2000)
    parser.add_argument('--dt', type=float, default=0.0025)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--n_eigvals', type=int, default=10)
    parser.add_argument('--latents_path', type=str,
                        default='../sdxl/results_da_pcavda/clip_da_latents_elderly_gs7p5.npz')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--score_batch_size', type=int, default=16)
    parser.add_argument('--clip_batch_size', type=int, default=4)
    args = parser.parse_args()

    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Task {task_id}: device={device}, reps={args.reps_per_task}, "
          f"N={args.N}, T={args.T}, dt={args.dt}")

    # --- Load observed latents for SVD ---
    print("Loading observed latents...")
    lat = np.load(args.latents_path)
    x0_obs = lat['x0']  # (200, 65536)
    var_f = x0_obs.var(axis=0).mean()  # scalar diffusion coefficient
    print(f"  x0 shape: {x0_obs.shape}, var_f: {var_f:.4f}")

    latent_shape = (4, 128, 128)
    d_latent = int(np.prod(latent_shape))
    assert x0_obs.shape[1] == d_latent, f"Expected {d_latent}, got {x0_obs.shape[1]}"

    # --- Build SDXLScore (FIXED across all replicates) ---
    print("Building SDXLScore...")
    t_build = time.time()
    score_fn = build_sdxl_score(
        device, guidance_scale=args.guidance_scale,
        latent_shape=latent_shape,
        score_batch_size=args.score_batch_size,
    )
    print(f"  SDXLScore built in {time.time()-t_build:.0f}s")

    # --- Build CLIP embedder ---
    print("Building CLIP embedder...")
    embedder = RealCLIPEmbedder(device=device)

    # --- Run null replicates ---
    null_evals_all = []
    seed_offset = task_id * args.reps_per_task

    for rep in range(args.reps_per_task):
        seed = 9000 + seed_offset + rep
        print(f"\n{'='*60}")
        print(f"Replicate {rep+1}/{args.reps_per_task} (seed={seed})")
        print(f"{'='*60}")

        # 1. Sample null latents from N(0, Σ_hat) via SVD
        t0 = time.time()
        x_null_np = sample_null_latents(x0_obs, args.N, seed)
        x_null = torch.from_numpy(x_null_np).to(device)
        var_f_null = x_null_np.var(axis=0).mean()
        print(f"  Sampled null latents: shape={x_null_np.shape}, "
              f"var_f_null={var_f_null:.4f} (obs: {var_f:.4f})")

        # 2. Run Langevin with FIXED SDXLScore
        snapshot_lags = [args.T]  # only need final state
        print(f"  Running Langevin (T={args.T}, dt={args.dt})...")
        t_lang = time.time()
        snapshots, x0_traj = run_langevin_latent_space(
            score_fn=score_fn,
            samples_flat=x_null,
            var_f=var_f_null,
            n_traj=args.N,
            T=args.T,
            dt=args.dt,
            snapshot_lags=snapshot_lags,
            device=device,
            seed=seed + 50000,
        )
        lang_time = time.time() - t_lang
        print(f"  Langevin done in {lang_time:.0f}s ({lang_time/3600:.1f}h)")

        x_tau = snapshots[args.T]  # (N, d_latent) CPU

        # 3. CLIP embed t=0 and t=τ*
        print("  CLIP embedding...")
        t_embed = time.time()
        z0 = embedder.embed(x0_traj.numpy(), latent_shape=latent_shape,
                            batch_size=args.clip_batch_size)  # (N, 768)
        z_tau = embedder.embed(x_tau.numpy(), latent_shape=latent_shape,
                               batch_size=args.clip_batch_size)  # (N, 768)
        embed_time = time.time() - t_embed
        print(f"  CLIP done in {embed_time:.0f}s, z0={z0.shape}, z_tau={z_tau.shape}")

        # 4. Compute C(τ*) in CLIP space
        mu_clip = z0.mean(axis=0)
        z0_c = z0 - mu_clip
        z_tau_c = z_tau - mu_clip
        C_tau = (z_tau_c.T @ z0_c) / args.N
        C_tau_sym = 0.5 * (C_tau + C_tau.T)

        evals = np.linalg.eigvalsh(C_tau_sym)[::-1].copy()  # descending
        null_evals_all.append(evals[:args.n_eigvals])

        total_rep = time.time() - t0
        print(f"  λ₁={evals[0]:.6f}, λ₂={evals[1]:.6f}, "
              f"gap={evals[0]/max(abs(evals[1]),1e-12):.2f}")
        print(f"  Total replicate time: {total_rep:.0f}s ({total_rep/3600:.1f}h)")

    null_evals_arr = np.array(null_evals_all)

    # --- Save ---
    out_path = os.path.join(args.output_dir, f'sdxl_null_task{task_id:03d}.npz')
    np.savez(out_path,
             null_eigenvalues=null_evals_arr,
             task_id=task_id,
             seed_offset=seed_offset,
             reps_per_task=args.reps_per_task,
             N=args.N, T=args.T, dt=args.dt,
             guidance_scale=args.guidance_scale,
             n_eigvals=args.n_eigvals,
             var_f_obs=var_f,
             latent_shape=np.array(latent_shape))
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
