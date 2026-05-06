#!/usr/bin/env python3
"""SDXL elderly iso-Gaussian null via late-t SDXL score — one SLURM array task.

Implements an empirical iso-Gaussian Theorem 3 floor by:
  1. Sampling x_init ~ N(0, sigma_{t_null}^2 I) in latent space.
  2. Running canonical Langevin
        dX = (var_f/2) * score(X) dt + sqrt(var_f) dW
     with score = SDXLScore(unet, t=t_null, prompt=elderly, gs=7.5),
     var_f = sigma_{t_null}^2, T=2000 steps at dt=0.0025.
  3. VAE decode + CLIP embed at t=0 (init) and t=tau* (= T).

Saves z0 (N_per_task, 768) and z_tau (N_per_task, 768) for later bootstrap
subsampling.

CLI:
    python run_sdxl_iso_null.py --task_id $SLURM_ARRAY_TASK_ID \
        [--t_null 999] [--N_per_task 50] [--T 2000] [--dt 0.0025] \
        [--guidance_scale 7.5]
"""
import argparse
import gc
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'sdxl', 'scripts'))

from compute_sdxl_ssa_v2 import SDXLScore  # noqa: E402
from sdxl_components import RealCLIPEmbedder  # noqa: E402


def build_score_and_vae(device, t_null, guidance_scale=7.5,
                        latent_shape=(4, 128, 128), score_batch_size=16):
    """Load SDXL pipe, build SDXLScore at t=t_null, free unused components,
    and load a fresh fp32 VAE for the embedder.

    Returns (score_fn, vae, sigma_t_null).
    """
    from diffusers import StableDiffusionXLPipeline, DDPMScheduler, AutoencoderKL

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    print(f"Loading SDXL: {model_id}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to(device)

    prompt = "a portrait of an elderly person"
    negative_prompt = ""
    (prompt_embeds, negative_prompt_embeds,
     pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=prompt, negative_prompt=negative_prompt,
        device=device, num_images_per_prompt=1, do_classifier_free_guidance=True,
    )
    time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]],
                            dtype=torch.float16, device=device)
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    unet = pipe.unet

    del pipe.vae, pipe.text_encoder, pipe.text_encoder_2
    del pipe.tokenizer, pipe.tokenizer_2, pipe
    gc.collect(); torch.cuda.empty_cache()

    score_fn = SDXLScore(
        unet=unet, scheduler=scheduler, t=t_null,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        time_ids=time_ids,
        device=device, latent_shape=latent_shape,
        batch_size=score_batch_size,
        guidance_scale=guidance_scale,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    )
    sigma_t_null = float(score_fn.sigma_t)

    print("Loading fp32 VAE for CLIP embedder...")
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    ).to(device).eval()
    return score_fn, vae, sigma_t_null


def sample_iso_gaussian_latents(n, sigma, latent_dim, seed):
    """Sample N(0, sigma^2 I) in latent space. Returns (n, latent_dim) fp32."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(size=(n, latent_dim)).astype(np.float32) * sigma)


def run_langevin_inline(score_fn, x_init, var_f, T, dt, device, seed,
                        log_every=100):
    """Euler-Maruyama Langevin: dX = (var_f/2) score dt + sqrt(var_f) dW.

    Inlined here to avoid the random sample-with-replacement in the helper —
    we want every sampled null latent to be a unique trajectory.
    Returns (x0_cpu, x_tau_cpu).
    """
    sigma_f = float(np.sqrt(var_f))
    sqrt_dt = float(np.sqrt(dt))

    x = torch.from_numpy(x_init).to(device)
    x0_cpu = x.clone().cpu()

    torch.manual_seed(seed)
    t_start = time.time()
    for t in range(1, T + 1):
        grad = score_fn(x)
        x = x + 0.5 * var_f * dt * grad + sigma_f * sqrt_dt * torch.randn_like(x)
        if t % log_every == 0 or t == T:
            elapsed = time.time() - t_start
            rate = t / elapsed if elapsed > 0 else 0.0
            eta_h = (T - t) / rate / 3600 if rate > 0 else 0.0
            xn = x.norm(dim=-1).mean().item()
            gn = grad.norm(dim=-1).mean().item()
            print(f"    Langevin step {t}/{T}: ||x||={xn:.2f} ||score||={gn:.2f} "
                  f"{rate:.2f} steps/s ETA {eta_h:.2f}h", flush=True)
    return x0_cpu, x.clone().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=None)
    parser.add_argument('--t_null', type=int, default=999,
                        help='Score-evaluation timestep (large = iso-Gaussian).')
    parser.add_argument('--N_per_task', type=int, default=50)
    parser.add_argument('--T', type=int, default=2000)
    parser.add_argument('--dt', type=float, default=0.0025)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--base_seed', type=int, default=9000)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--score_batch_size', type=int, default=16)
    parser.add_argument('--clip_batch_size', type=int, default=4)
    parser.add_argument('--var_f_mode', type=str, default='sigma_t_null_sq',
                        choices=['sigma_t_null_sq', 'real'],
                        help="'sigma_t_null_sq' (default, matches iso-Gaussian "
                             "stationary variance); 'real' uses var_f from real "
                             "x0 latents (=0.32) — same physical-time scaling but "
                             "different mixing rate.")
    parser.add_argument('--real_var_f_path', type=str,
                        default='../../sdxl/results_da_pcavda/clip_da_latents_elderly_gs7p5.npz')
    args = parser.parse_args()

    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    seed = args.base_seed + 1000 * task_id

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Task {task_id}: device={device}, t_null={args.t_null}, "
          f"N_per_task={args.N_per_task}, T={args.T}, dt={args.dt}, seed={seed}")

    latent_shape = (4, 128, 128)
    latent_dim = int(np.prod(latent_shape))

    print("Building SDXLScore + fp32 VAE...")
    t_build = time.time()
    score_fn, vae, sigma_t_null = build_score_and_vae(
        device, t_null=args.t_null, guidance_scale=args.guidance_scale,
        latent_shape=latent_shape, score_batch_size=args.score_batch_size,
    )
    print(f"  build done in {time.time()-t_build:.0f}s, sigma_{{t_null}}={sigma_t_null:.6f}")

    if args.var_f_mode == 'sigma_t_null_sq':
        var_f = sigma_t_null ** 2
        print(f"  var_f = sigma_{{t_null}}^2 = {var_f:.6f}")
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        rp = args.real_var_f_path if os.path.isabs(args.real_var_f_path) else os.path.join(here, args.real_var_f_path)
        x0_obs = np.load(rp)['x0']
        var_f = float(x0_obs.var(axis=0).mean())
        print(f"  var_f = real-run var_f = {var_f:.6f}")

    print("Sampling iso-Gaussian initial latents N(0, sigma_t_null^2 I)...")
    x_init = sample_iso_gaussian_latents(
        args.N_per_task, sigma_t_null, latent_dim, seed)
    print(f"  x_init: {x_init.shape}, "
          f"per-dim var mean = {x_init.var(axis=0).mean():.6f}, "
          f"||x_init|| mean = {np.linalg.norm(x_init, axis=1).mean():.2f} "
          f"(expected ~{sigma_t_null * np.sqrt(latent_dim):.2f})")

    print(f"Running Langevin (T={args.T}, dt={args.dt})...")
    t_lang = time.time()
    x0_cpu, x_tau_cpu = run_langevin_inline(
        score_fn=score_fn, x_init=x_init, var_f=var_f,
        T=args.T, dt=args.dt, device=device, seed=seed + 50000,
    )
    lang_time = time.time() - t_lang
    print(f"  Langevin done in {lang_time:.0f}s ({lang_time/3600:.2f}h)")

    # Sanity for the trajectory itself: did it stay in the iso-Gaussian regime?
    init_norm = float(x0_cpu.norm(dim=-1).mean())
    final_norm = float(x_tau_cpu.norm(dim=-1).mean())
    print(f"  Norm check: ||x_0||_mean={init_norm:.2f} -> ||x_tau||_mean={final_norm:.2f} "
          f"(if final << init, dynamics shrunk; if final ~ init, OU-like)")

    print("Building CLIP embedder...")
    embedder = RealCLIPEmbedder(vae=vae, device=device, clip_batch_size=args.clip_batch_size)

    print("CLIP embedding (z0, z_tau)...")
    t_embed = time.time()
    z0 = embedder.embed(x0_cpu.numpy(), latent_shape=latent_shape,
                        batch_size=args.clip_batch_size)
    z_tau = embedder.embed(x_tau_cpu.numpy(), latent_shape=latent_shape,
                           batch_size=args.clip_batch_size)
    print(f"  CLIP done in {time.time()-t_embed:.0f}s, z0={z0.shape}, z_tau={z_tau.shape}")

    # Subset C_hat_sym(tau*) eigenvalue — small-N noisy but useful smoke test.
    z0_c = z0 - z0.mean(axis=0)
    z_tau_c = z_tau - z_tau.mean(axis=0)
    C = z_tau_c.T @ z0_c / args.N_per_task
    Csym = 0.5 * (C + C.T)
    evals = np.linalg.eigvalsh(Csym)
    lam_max_subset = float(evals[-1])
    lam_min_subset = float(evals[0])
    print(f"  subset top  eigvals: lam_max = {lam_max_subset:.6e}")
    print(f"                       lam_min = {lam_min_subset:.6e}")
    print(f"  REFERENCE: real lambda_1^obs = 9.96e-3 (must be much larger than null)")
    print(f"  z0   norm mean: {np.linalg.norm(z0, axis=1).mean():.4f} "
          f"(L2-normalized, expect 1.0)")
    print(f"  z_tau norm mean: {np.linalg.norm(z_tau, axis=1).mean():.4f}")

    out_path = os.path.join(args.output_dir, f'task_{task_id:03d}.npz')
    np.savez_compressed(
        out_path,
        z0=z0.astype(np.float32),
        z_tau=z_tau.astype(np.float32),
        task_id=np.array(task_id),
        seed=np.array(seed),
        N_per_task=np.array(args.N_per_task),
        T=np.array(args.T), dt=np.array(args.dt),
        guidance_scale=np.array(args.guidance_scale),
        t_null=np.array(args.t_null),
        sigma_t_null=np.array(sigma_t_null),
        var_f=np.array(var_f),
        var_f_mode=np.array(args.var_f_mode),
        latent_shape=np.array(latent_shape),
        lam_max_subset=np.array(lam_max_subset),
        init_norm_mean=np.array(init_norm),
        final_norm_mean=np.array(final_norm),
        langevin_seconds=np.array(lang_time),
    )
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
