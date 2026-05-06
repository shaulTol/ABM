"""
run_ssa_da_clip.py — Full SSA + DA pipeline in real CLIP space (768D).

Generates latents via SDXL (+ControlNet), runs Langevin dynamics,
embeds all snapshots via VAE decode + CLIP ViT-L/14, computes SSA + DA,
then interprets DA directions with text probes.

Usage:
    python scripts/run_ssa_da_clip.py \
        --model_id stabilityai/sdxl-turbo \
        --name clip_da_turbo_face \
        --prompt "A photo of a human face" \
        --guidance_scale 0.0 --gen_steps 2 \
        --n_samples 200 --n_traj 200 \
        --T 500 --dt 0.0001 --t_score 1 \
        --output_dir results_clip_da

    # With ControlNet:
    python scripts/run_ssa_da_clip.py \
        --model_id stabilityai/sdxl-turbo \
        --name clip_da_cn_dps \
        --prompt "A photo of a human face" \
        --guidance_scale 0.0 --gen_steps 2 \
        --controlnet_model_id xinsir/controlnet-scribble-sdxl-1.0 \
        --control_image_path path/to/scribble.png \
        --controlnet_scale 0.5 \
        --n_samples 200 --n_traj 200 \
        --T 500 --dt 0.0001 \
        --output_dir results_clip_da

    # Resume from checkpoint (skips generation + Langevin):
    python scripts/run_ssa_da_clip.py \
        --model_id stabilityai/stable-diffusion-xl-base-1.0 \
        --name v9_resume \
        --resume_checkpoint results_clip_da/langevin_ckpt_v9.npz \
        --clip_batch_size 1 \
        --output_dir results_clip_da
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)

from sdxl_components import (
    LatentCLIPEmbedder, RealCLIPEmbedder, TextProbeAnalyzer,
    compute_ssa_da, decode_latents_to_pil, make_topbottom_grid,
    plot_rho_decay, plot_gap_profile, DEFAULT_PROBE_PAIRS,
)
from compute_sdxl_ssa_v2 import (
    SDXLScore, generate_latents, run_langevin_latent_space,
    compute_knn_entropy,
)


def pca_project(X, k):
    """PCA-project (N, D) data to (N, k). Returns projected data and eigenvalues."""
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / X.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return Xc @ eigvecs[:, :k], eigvals[:k]


def gaussian_entropy(eigvals):
    """H_gauss = 0.5 * sum(log(2*pi*e * lambda_i))."""
    eigvals = np.maximum(eigvals, 1e-30)
    return 0.5 * np.sum(np.log(2 * np.pi * np.e * eigvals))


def kde_entropy(X_proj, n_eval=5000, seed=42):
    """KDE-based entropy via Monte Carlo on KDE samples."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(X_proj.T)
    rng = np.random.RandomState(seed)
    samples = kde.resample(n_eval, seed=rng)
    log_pdf = kde.logpdf(samples)
    return float(-np.mean(log_pdf))


def parse_args():
    p = argparse.ArgumentParser(
        description="SSA + DA in real CLIP space (768D) for SDXL"
    )
    # Model
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--prompt", type=str, default="A photo of a human face")
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--gen_steps", type=int, default=50)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    # Langevin
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--n_traj", type=int, default=200)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--dt", type=float, default=0.0001)
    p.add_argument("--t_score", type=int, default=1)
    p.add_argument("--score_batch_size", type=int, default=128)
    p.add_argument("--gen_batch_size", type=int, default=4)
    # ControlNet
    p.add_argument("--controlnet_model_id", type=str, default=None)
    p.add_argument("--control_image_path", type=str, default=None)
    p.add_argument("--controlnet_scale", type=float, default=0.8)
    # CLIP
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--clip_batch_size", type=int, default=16)
    # Options
    p.add_argument("--compute_entropy", action="store_true")
    p.add_argument("--entropy_k", type=int, default=10)
    p.add_argument("--also_compute_lc", action="store_true",
                   help="Also compute SSA/DA in Latent-CLIP space for comparison")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_checkpoint", type=str, default=None,
                   help="Path to .npz checkpoint from a previous run (skips generation+Langevin)")
    p.add_argument("--resume_x0", type=str, default=None,
                   help="Path to latents npz with x0 key. Skips generation, re-runs Langevin with current dt/T.")
    p.add_argument("--ssa_min_lag", type=int, default=0,
                   help="Only include lags >= this value when computing SSA mean")
    p.add_argument("--ssa_only", action="store_true",
                   help="Skip DA eigendecomposition, text probe cosines on DA, and DA1 grid")
    p.add_argument("--freeconv_stop_M", type=int, default=None,
                   help="Apply free-conv DA stopping: stop when lambda_2(C(tau)) "
                        "< lambda_plus(tau) for M consecutive lags. tau* = lag "
                        "just before the first crossing. Saves full C(tau*) "
                        "matrix and full eigenspectrum for distribution analysis.")
    p.add_argument("--snapshot_schedule", type=str, default="linear",
                   choices=["linear", "log_and_fixed"],
                   help="Snapshot schedule: 'linear' (20 evenly spaced) or "
                        "'log_and_fixed' (log2 + every-20 union, ~108 points)")
    p.add_argument("--no_save_checkpoint", action="store_true",
                   help="Skip saving large Langevin checkpoint to save disk")
    p.add_argument("--clip_outlier_sigma", type=float, default=None,
                   help="Remove samples with CLIP L2 distance > k*sigma from mean. "
                        "When set, saves both full and filtered results.")
    return p.parse_args()


def compute_snapshot_lags(T, schedule="linear"):
    """Compute snapshot lag values based on schedule type.

    Args:
        T: total Langevin steps
        schedule: 'linear' (20 evenly spaced) or 'log_and_fixed' (log2 + every-20)

    Returns:
        sorted list of lag values
    """
    if schedule == "log_and_fixed":
        log_lags = [2**i for i in range(int(np.log2(T)) + 1) if 2**i <= T]
        if T not in log_lags:
            log_lags.append(T)
        fixed_lags = list(range(20, T + 1, 20))
        snapshot_lags = sorted(set(log_lags + fixed_lags))
    else:  # linear
        snapshot_lags = sorted(set(int(round(x)) for x in np.linspace(10, T, 20)))
        if T not in snapshot_lags:
            snapshot_lags.append(T)
        snapshot_lags = sorted(snapshot_lags)
    return snapshot_lags


def run_stages_1_2(args, output_dir, device, latent_shape, use_controlnet):
    """Stage 1 (generate latents) + Stage 2 (Langevin). Returns x0, snapshots, metadata."""

    # --- Wandb (optional) ---
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project="sdxl-ssa",
            name=f"clip_{args.name}",
            config=vars(args),
            tags=["sdxl", "clip-mode"],
            dir=os.environ.get("WANDB_DIR", "."),
        )
        print(f"  wandb run: {wandb_run.url}")
    except Exception as e:
        print(f"  wandb init failed ({e}), continuing without logging")

    # =====================================================================
    # Stage 1: Generate latents
    # =====================================================================
    controlnet = None
    control_pil = None
    control_image_tensor = None
    if use_controlnet:
        print("\n=== Loading ControlNet ===")
        from diffusers import ControlNetModel
        controlnet = ControlNetModel.from_pretrained(
            args.controlnet_model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device).eval()

        from PIL import Image
        control_pil = Image.open(args.control_image_path).convert("RGB")
        control_pil = control_pil.resize((args.width, args.height))
        import torchvision.transforms as T
        control_image_tensor = T.ToTensor()(control_pil).unsqueeze(0).to(device)

    print("\n=== Loading SDXL pipeline ===")
    if use_controlnet:
        from diffusers import StableDiffusionXLControlNetPipeline
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            args.model_id, controlnet=controlnet,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        )
    else:
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        )
    pipe = pipe.to(device)
    unet = pipe.unet
    scheduler = pipe.scheduler
    unet.eval()

    print("\n=== Encoding prompt ===")
    (prompt_embeds, negative_prompt_embeds,
     pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=args.prompt, device=device, num_images_per_prompt=1,
        do_classifier_free_guidance=True, negative_prompt="",
    )
    time_ids = torch.tensor(
        [[args.height, args.width, 0, 0, args.height, args.width]],
        dtype=torch.float16, device=device,
    )

    print(f"\n=== Generating {args.n_samples} latents ===")
    t0 = time.time()
    gen_kwargs = {}
    if use_controlnet:
        gen_kwargs["control_image"] = control_pil
        gen_kwargs["controlnet_conditioning_scale"] = args.controlnet_scale
    latents = generate_latents(
        pipe, args.prompt, args.n_samples,
        guidance_scale=args.guidance_scale, gen_steps=args.gen_steps,
        height=args.height, width=args.width,
        batch_size=args.gen_batch_size, seed=args.seed,
        **gen_kwargs,
    )
    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.1f}s")

    latents_flat = latents.view(latents.shape[0], -1).float()
    var_f = latents_flat.var(dim=0).mean().item()
    print(f"  var_f = {var_f:.6f}")

    # =====================================================================
    # Stage 2: Langevin dynamics
    # =====================================================================
    cn_score_kwargs = {}
    if use_controlnet:
        cn_score_kwargs["controlnet"] = controlnet
        cn_score_kwargs["control_image"] = control_image_tensor
        cn_score_kwargs["controlnet_conditioning_scale"] = args.controlnet_scale

    score_fn = SDXLScore(
        unet=unet, scheduler=scheduler, t=args.t_score,
        prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
        time_ids=time_ids, device=device, latent_shape=latent_shape,
        batch_size=args.score_batch_size, guidance_scale=args.guidance_scale,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        **cn_score_kwargs,
    )

    snapshot_lags = compute_snapshot_lags(args.T, args.snapshot_schedule)

    print(f"\n=== Langevin dynamics ({args.n_traj} traj x {args.T} steps) ===")
    print(f"  Snapshot schedule: {args.snapshot_schedule} ({len(snapshot_lags)} points)")
    print(f"  Snapshot lags: {snapshot_lags}")
    t0 = time.time()
    snapshots, x0_flat = run_langevin_latent_space(
        score_fn=score_fn, samples_flat=latents_flat, var_f=var_f,
        n_traj=args.n_traj, T=args.T, dt=args.dt,
        snapshot_lags=snapshot_lags, device=device, seed=args.seed,
        wandb_run=wandb_run,
    )
    langevin_time = time.time() - t0
    print(f"  Langevin completed in {langevin_time:.1f}s ({langevin_time/3600:.1f}h)")

    # =====================================================================
    # Save checkpoint (so CLIP embedding can resume if it OOMs)
    # =====================================================================
    if not args.no_save_checkpoint:
        ckpt_path = os.path.join(output_dir, f"langevin_ckpt_{args.name}.npz")
        ckpt_data = {
            "x0_flat": x0_flat.numpy(),
            "snapshot_lags": np.array(snapshot_lags),
            "var_f": np.array(var_f),
            "gen_time": np.array(gen_time),
            "langevin_time": np.array(langevin_time),
            "snapshot_dtype_on_disk": np.array("float16"),
        }
        for lag, snap in snapshots.items():
            ckpt_data[f"snapshot_{lag}"] = snap.numpy().astype(np.float16)
        np.savez_compressed(ckpt_path, **ckpt_data)
        print(f"  Saved Langevin checkpoint (snapshots fp16): {ckpt_path}")
    else:
        print("  Skipping checkpoint save (--no_save_checkpoint)")

    # Free GPU for CLIP
    print("\n=== Freeing GPU for CLIP embedding ===")
    del score_fn
    if controlnet is not None:
        controlnet = controlnet.to("cpu")
    vae = pipe.vae.to(device).float().eval()
    pipe.unet = pipe.unet.to("cpu")
    torch.cuda.empty_cache()

    x0_np = x0_flat.numpy()
    snapshots_np = {lag: snap.numpy() for lag, snap in snapshots.items()}

    return vae, x0_np, snapshots_np, snapshot_lags, var_f, gen_time, langevin_time, wandb_run


def run_stages_1_2_resume(args, output_dir, device, use_controlnet):
    """Resume from checkpoint — load VAE only, skip generation + Langevin."""
    print(f"=== RESUMING from checkpoint: {args.resume_checkpoint} ===")
    ckpt = np.load(args.resume_checkpoint, allow_pickle=True)
    x0_np = ckpt["x0_flat"]
    snapshot_lags = ckpt["snapshot_lags"].tolist()
    snapshots_np = {int(lag): ckpt[f"snapshot_{lag}"]
                    for lag in snapshot_lags if f"snapshot_{lag}" in ckpt}
    var_f = float(ckpt["var_f"])
    gen_time = float(ckpt.get("gen_time", 0))
    langevin_time = float(ckpt.get("langevin_time", 0))
    print(f"  Loaded x0: {x0_np.shape}, {len(snapshots_np)} snapshots, var_f={var_f:.6f}")
    print(f"  Previous gen_time={gen_time:.1f}s, langevin_time={langevin_time:.1f}s")

    # Load only VAE for decode (no UNet needed)
    print("\n=== Loading SDXL VAE for decode ===")
    if use_controlnet:
        from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
        controlnet = ControlNetModel.from_pretrained(
            args.controlnet_model_id,
            torch_dtype=torch.float16, use_safetensors=True,
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            args.model_id, controlnet=controlnet,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        )
    else:
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        )
    vae = pipe.vae.to(device).float().eval()
    # Don't even load UNet to GPU
    del pipe.unet
    del pipe
    torch.cuda.empty_cache()

    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project="sdxl-ssa",
            name=f"clip_{args.name}_resume",
            config=vars(args),
            tags=["sdxl", "clip-mode", "resume"],
            dir=os.environ.get("WANDB_DIR", "."),
        )
    except Exception:
        pass

    return vae, x0_np, snapshots_np, snapshot_lags, var_f, gen_time, langevin_time, wandb_run


def run_stages_1_2_resume_x0(args, output_dir, device, latent_shape, use_controlnet):
    """Resume from x0 latents — reload UNet for score, re-run Langevin with new dt/T."""
    print(f"=== RESUMING x0 from: {args.resume_x0} ===")
    x0_data = np.load(args.resume_x0, allow_pickle=True)
    x0_flat_np = x0_data["x0"]  # (N, d_latent)
    print(f"  Loaded x0: {x0_flat_np.shape}")

    x0_flat = torch.from_numpy(x0_flat_np).float()
    var_f = x0_flat.var(dim=0).mean().item()
    print(f"  var_f = {var_f:.6f}")

    # --- Wandb ---
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project="sdxl-ssa",
            name=f"clip_{args.name}",
            config=vars(args),
            tags=["sdxl", "clip-mode", "resume-x0"],
            dir=os.environ.get("WANDB_DIR", "."),
        )
        print(f"  wandb run: {wandb_run.url}")
    except Exception as e:
        print(f"  wandb init failed ({e}), continuing without logging")

    # --- Load pipeline for score ---
    controlnet = None
    control_image_tensor = None
    if use_controlnet:
        print("\n=== Loading ControlNet ===")
        from diffusers import ControlNetModel
        controlnet = ControlNetModel.from_pretrained(
            args.controlnet_model_id,
            torch_dtype=torch.float16, use_safetensors=True,
        ).to(device).eval()
        from PIL import Image as PILImage
        control_pil = PILImage.open(args.control_image_path).convert("RGB")
        control_pil = control_pil.resize((args.width, args.height))
        import torchvision.transforms as T
        control_image_tensor = T.ToTensor()(control_pil).unsqueeze(0).to(device)

    print("\n=== Loading SDXL pipeline (for score) ===")
    if use_controlnet:
        from diffusers import StableDiffusionXLControlNetPipeline
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            args.model_id, controlnet=controlnet,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        )
    else:
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        )
    pipe = pipe.to(device)
    unet = pipe.unet
    scheduler = pipe.scheduler
    unet.eval()

    print("\n=== Encoding prompt ===")
    (prompt_embeds, negative_prompt_embeds,
     pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=args.prompt, device=device, num_images_per_prompt=1,
        do_classifier_free_guidance=True, negative_prompt="",
    )
    time_ids = torch.tensor(
        [[args.height, args.width, 0, 0, args.height, args.width]],
        dtype=torch.float16, device=device,
    )

    # --- Score function ---
    cn_score_kwargs = {}
    if use_controlnet:
        cn_score_kwargs["controlnet"] = controlnet
        cn_score_kwargs["control_image"] = control_image_tensor
        cn_score_kwargs["controlnet_conditioning_scale"] = args.controlnet_scale

    score_fn = SDXLScore(
        unet=unet, scheduler=scheduler, t=args.t_score,
        prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
        time_ids=time_ids, device=device, latent_shape=latent_shape,
        batch_size=args.score_batch_size, guidance_scale=args.guidance_scale,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        **cn_score_kwargs,
    )

    # --- Langevin ---
    snapshot_lags = compute_snapshot_lags(args.T, args.snapshot_schedule)

    print(f"\n=== Langevin dynamics ({args.n_traj} traj x {args.T} steps, dt={args.dt}) ===")
    print(f"  Snapshot schedule: {args.snapshot_schedule} ({len(snapshot_lags)} points)")
    print(f"  Snapshot lags: {snapshot_lags}")
    gen_time = 0.0  # generation skipped
    t0 = time.time()
    snapshots, x0_out = run_langevin_latent_space(
        score_fn=score_fn, samples_flat=x0_flat, var_f=var_f,
        n_traj=args.n_traj, T=args.T, dt=args.dt,
        snapshot_lags=snapshot_lags, device=device, seed=args.seed,
        wandb_run=wandb_run,
    )
    langevin_time = time.time() - t0
    print(f"  Langevin completed in {langevin_time:.1f}s ({langevin_time/3600:.1f}h)")

    # --- Save checkpoint ---
    if not args.no_save_checkpoint:
        ckpt_path = os.path.join(output_dir, f"langevin_ckpt_{args.name}.npz")
        ckpt_data = {
            "x0_flat": x0_out.numpy(),
            "snapshot_lags": np.array(snapshot_lags),
            "var_f": np.array(var_f),
            "gen_time": np.array(gen_time),
            "langevin_time": np.array(langevin_time),
            "snapshot_dtype_on_disk": np.array("float16"),
        }
        for lag, snap in snapshots.items():
            ckpt_data[f"snapshot_{lag}"] = snap.numpy().astype(np.float16)
        np.savez_compressed(ckpt_path, **ckpt_data)
        print(f"  Saved Langevin checkpoint (snapshots fp16): {ckpt_path}")
    else:
        print("  Skipping checkpoint save (--no_save_checkpoint)")

    # --- Free GPU for CLIP ---
    print("\n=== Freeing GPU for CLIP embedding ===")
    del score_fn
    if controlnet is not None:
        controlnet = controlnet.to("cpu")
    vae = pipe.vae.to(device).float().eval()
    pipe.unet = pipe.unet.to("cpu")
    torch.cuda.empty_cache()

    x0_np = x0_out.numpy()
    snapshots_np = {lag: snap.numpy() for lag, snap in snapshots.items()}

    return vae, x0_np, snapshots_np, snapshot_lags, var_f, gen_time, langevin_time, wandb_run


def _run_analysis(
    args, output_dir, device, vae, x0_np, snapshots_np,
    z0_clip, z_snapshots, snapshot_lags,
    analyzer, use_controlnet,
    var_f, gen_time, langevin_time, embed_time,
    run_name, name_suffix, outlier_info, wandb_run,
):
    """Run stages 4-8 (SSA/DA, probes, entropy, save, viz) on given CLIP data."""
    from scipy.stats import gaussian_kde as _gaussian_kde

    # =====================================================================
    # Stage 4: Compute SSA + DA in CLIP space
    # =====================================================================
    print(f"\n=== Computing SSA{'' if args.ssa_only else ' + DA'} in CLIP space (768D) ===")
    t0 = time.time()
    results = compute_ssa_da(z0_clip, z_snapshots, snapshot_lags,
                             ssa_min_lag=args.ssa_min_lag, ssa_only=args.ssa_only,
                             dt=args.dt,
                             freeconv_stop_M=getattr(args, 'freeconv_stop_M', None))
    ssa_time = time.time() - t0
    print(f"  SSA{'' if args.ssa_only else '+DA'} computed in {ssa_time:.1f}s")

    rho0_check = np.sum(
        (z0_clip - z0_clip.mean(0)) ** 2, axis=-1
    ).mean()
    print(f"  C(0) trace check: {rho0_check:.6f}")

    # =====================================================================
    # Stage 5: Text probe analysis on DA directions
    # =====================================================================
    print("\n=== Text probe analysis ===")
    da1_cosines = None
    da2_cosines = None
    da1_scores = None
    da1_corr = None

    if not args.ssa_only:
        da1_cosines = analyzer.score_direction(results["da1_eigvec"])
        da2_cosines = analyzer.score_direction(results["da2_eigvec"])

        print("\n  DA1 direction cosines with text probes:")
        for name in analyzer.probe_names:
            print(f"    {name:<15} cos = {da1_cosines[name]:>7.4f}")

        print("\n  DA2 direction cosines with text probes:")
        for name in analyzer.probe_names:
            print(f"    {name:<15} cos = {da2_cosines[name]:>7.4f}")

        da1_scores = (z0_clip - results["mu"]) @ results["da1_eigvec"]
        da1_corr = analyzer.correlate_with_ranking(z0_clip, da1_scores)

        print("\n  DA1 per-sample Spearman with text probes:")
        for name in analyzer.probe_names:
            r = da1_corr[name]
            print(f"    {name:<15} rho = {r['rho']:>7.4f} (p={r['pvalue']:.2e})")
    else:
        print("  Skipping DA direction probes (--ssa_only)")

    # =====================================================================
    # Stage 5a: Gender-sliced SSA
    # =====================================================================
    print("\n=== Gender-sliced SSA ===")
    gender_axis = analyzer.probe_axes["gender"]
    z0_c = z0_clip - results["mu"]
    s0_gender = z0_c @ gender_axis
    s0_gender_c = s0_gender - s0_gender.mean()
    var_gender = np.mean(s0_gender_c ** 2)

    rho_sq_gender = []
    for lag in results["snapshot_lags"]:
        if lag not in z_snapshots:
            continue
        z_tau_c = z_snapshots[lag] - results["mu"]
        s_tau_gender = z_tau_c @ gender_axis
        s_tau_gender_c = s_tau_gender - s_tau_gender.mean()
        rho_g = np.mean(s0_gender_c * s_tau_gender_c) / max(var_gender, 1e-12)
        rho_sq_gender.append(rho_g ** 2)
    rho_sq_gender = np.array(rho_sq_gender)
    ssa_gender_sliced = float(np.mean(rho_sq_gender))
    print(f"  SSA_gender_sliced = {ssa_gender_sliced:.6f}")
    print(f"  SSA_full          = {results['ssa']:.6f}")
    print(f"  Ratio (sliced/full) = {ssa_gender_sliced / max(results['ssa'], 1e-12):.4f}")

    gender_kde = _gaussian_kde(s0_gender)
    gender_log_pdf = gender_kde.logpdf(s0_gender)
    gender_entropy_1d = float(-np.mean(gender_log_pdf))
    print(f"  H_gender_1d (KDE) = {gender_entropy_1d:.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_gh, ax_gh = plt.subplots(figsize=(6, 4))
        ax_gh.hist(s0_gender, bins=30, density=True, alpha=0.5, color="steelblue",
                   edgecolor="k", linewidth=0.5, label="histogram")
        xs = np.linspace(s0_gender.min() - 0.5, s0_gender.max() + 0.5, 200)
        ax_gh.plot(xs, gender_kde(xs), "r-", lw=2, label="KDE")
        ax_gh.set_xlabel("Gender score (CLIP)")
        ax_gh.set_ylabel("Density")
        ax_gh.set_title(f"{run_name}: Gender Distribution\nH={gender_entropy_1d:.3f}")
        ax_gh.legend()
        ax_gh.grid(True, alpha=0.3)
        fig_gh.tight_layout()
        ghist_path = os.path.join(output_dir, f"clip_da_{run_name}_gender_hist.png")
        fig_gh.savefig(ghist_path, dpi=150)
        plt.close(fig_gh)
        print(f"  Saved gender histogram: {ghist_path}")
    except Exception as e:
        print(f"  Gender histogram failed: {e}")

    # =====================================================================
    # Stage 5a2: Glasses-sliced SSA
    # =====================================================================
    ssa_glasses_sliced = None
    if "glasses" in analyzer.probe_axes:
        print("\n=== Glasses-sliced SSA ===")
        glasses_axis = analyzer.probe_axes["glasses"]
        s0_glasses = z0_c @ glasses_axis
        s0_glasses_c = s0_glasses - s0_glasses.mean()
        var_glasses = np.mean(s0_glasses_c ** 2)

        rho_sq_glasses = []
        for lag in results["snapshot_lags"]:
            if lag not in z_snapshots:
                continue
            z_tau_c = z_snapshots[lag] - results["mu"]
            s_tau_glasses = z_tau_c @ glasses_axis
            s_tau_glasses_c = s_tau_glasses - s_tau_glasses.mean()
            rho_g = np.mean(s0_glasses_c * s_tau_glasses_c) / max(var_glasses, 1e-12)
            rho_sq_glasses.append(rho_g ** 2)
        rho_sq_glasses = np.array(rho_sq_glasses)
        ssa_glasses_sliced = float(np.mean(rho_sq_glasses))
        print(f"  SSA_glasses_sliced = {ssa_glasses_sliced:.6f}")
        print(f"  SSA_full           = {results['ssa']:.6f}")
        print(f"  Ratio (sliced/full) = {ssa_glasses_sliced / max(results['ssa'], 1e-12):.4f}")

        glasses_kde = _gaussian_kde(s0_glasses)
        glasses_log_pdf = glasses_kde.logpdf(s0_glasses)
        glasses_entropy_1d = float(-np.mean(glasses_log_pdf))
        print(f"  H_glasses_1d (KDE) = {glasses_entropy_1d:.4f}")

        try:
            fig_gl, ax_gl = plt.subplots(figsize=(6, 4))
            ax_gl.hist(s0_glasses, bins=30, density=True, alpha=0.5, color="orange",
                       edgecolor="k", linewidth=0.5, label="histogram")
            xs_gl = np.linspace(s0_glasses.min() - 0.5, s0_glasses.max() + 0.5, 200)
            ax_gl.plot(xs_gl, glasses_kde(xs_gl), "r-", lw=2, label="KDE")
            ax_gl.set_xlabel("Glasses score (CLIP)")
            ax_gl.set_ylabel("Density")
            ax_gl.set_title(f"{run_name}: Glasses Distribution\nH={glasses_entropy_1d:.3f}")
            ax_gl.legend()
            ax_gl.grid(True, alpha=0.3)
            fig_gl.tight_layout()
            glhist_path = os.path.join(output_dir, f"clip_da_{run_name}_glasses_hist.png")
            fig_gl.savefig(glhist_path, dpi=150)
            plt.close(fig_gl)
            print(f"  Saved glasses histogram: {glhist_path}")
        except Exception as e:
            print(f"  Glasses histogram failed: {e}")

    # =====================================================================
    # Stage 5b: Optional LC comparison (only for full data)
    # =====================================================================
    lc_results = None
    cross_rho = None
    if args.also_compute_lc and not name_suffix:
        print("\n=== Also computing SSA+DA in Latent-CLIP space (512D) ===")
        lc_embedder = LatentCLIPEmbedder(device)
        z0_lc = lc_embedder.embed(x0_np)
        z_snaps_lc = {}
        for lag in snapshot_lags:
            if lag in snapshots_np:
                z_snaps_lc[lag] = lc_embedder.embed(snapshots_np[lag])
        lc_results = compute_ssa_da(z0_lc, z_snaps_lc, snapshot_lags, dt=args.dt)

        from scipy.stats import spearmanr
        da1_clip_scores = (z0_clip - results["mu"]) @ results["da1_eigvec"]
        da1_lc_scores = (z0_lc - lc_results["mu"]) @ lc_results["da1_eigvec"]
        cross_rho, cross_p = spearmanr(da1_clip_scores, da1_lc_scores)
        print(f"\n  Cross-space DA1 Spearman: rho={cross_rho:.4f} (p={cross_p:.2e})")

    # =====================================================================
    # Stage 6: Full entropy suite (only for full data)
    # =====================================================================
    entropy = None
    entropy_suite = {}
    if args.compute_entropy and not name_suffix:
        print("\n=== Computing entropy suite in CLIP PCA space ===")

        k_pca_gauss = [5, 10, 20, 50]
        max_k = max(k_pca_gauss)
        _, all_eigvals = pca_project(z0_clip, max_k)
        gauss_results = {}
        for k in k_pca_gauss:
            h = gaussian_entropy(all_eigvals[:k])
            gauss_results[str(k)] = float(h)
            print(f"  H_gauss(k={k}): {h:.4f}")
        entropy_suite["gaussian"] = gauss_results

        knn_results = {}
        for k in k_pca_gauss:
            X_proj, _ = pca_project(z0_clip, k)
            h = compute_knn_entropy(torch.from_numpy(X_proj), k=args.entropy_k)
            knn_results[str(k)] = float(h)
            print(f"  H_knn(k_pca={k}): {h:.4f}")
        entropy_suite["knn"] = knn_results

        k_pca_kde = [2, 5]
        kde_results = {}
        for k in k_pca_kde:
            X_proj, _ = pca_project(z0_clip, k)
            h = kde_entropy(X_proj)
            kde_results[str(k)] = float(h)
            print(f"  H_kde(k_pca={k}): {h:.4f}")
        entropy_suite["kde"] = kde_results

        entropy = compute_knn_entropy(torch.from_numpy(z0_clip), k=args.entropy_k)
        entropy_suite["knn_full_768"] = float(entropy)
        print(f"  H_knn(full 768D): {entropy:.4f}")

    # =====================================================================
    # Stage 7: Save results
    # =====================================================================
    json_results = {
        "name": run_name,
        "model_id": args.model_id,
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "gen_steps": args.gen_steps,
        "height": args.height,
        "width": args.width,
        "controlnet_model_id": args.controlnet_model_id,
        "control_image_path": args.control_image_path,
        "controlnet_scale": args.controlnet_scale if use_controlnet else None,
        "clip_model": args.clip_model,
        "n_samples": int(z0_clip.shape[0]),
        "n_samples_original": args.n_samples,
        "n_traj": args.n_traj,
        "T": args.T,
        "dt": args.dt,
        "t_score": args.t_score,
        "var_f": var_f,
        "embed_dim": int(results["embed_dim"]),
        "ssa_clip": results["ssa"],
        "ssa_clip_all": results.get("ssa_all"),
        "ssa_min_lag": args.ssa_min_lag,
        "ssa_only": args.ssa_only,
        "snapshot_schedule": args.snapshot_schedule,
        "entropy_clip": entropy,
        "ssa_gender_sliced": ssa_gender_sliced,
        "rho_sq_gender": rho_sq_gender.tolist(),
        "gender_entropy_1d": gender_entropy_1d,
        "ssa_glasses_sliced": ssa_glasses_sliced,
        "entropy_suite": entropy_suite,
        "gen_time_s": gen_time,
        "langevin_time_s": langevin_time,
        "embed_time_s": embed_time,
        "ssa_time_s": ssa_time,
    }

    if outlier_info is not None:
        json_results.update(outlier_info)

    # Partitioned SSA for log_and_fixed schedule
    if args.snapshot_schedule == "log_and_fixed":
        used_lags = np.array(results["snapshot_lags"])
        rho_sq = results["rho_sq"]
        log_lags = [2**i for i in range(int(np.log2(args.T)) + 1) if 2**i <= args.T]
        if args.T not in log_lags:
            log_lags.append(args.T)
        fixed_lags = list(range(20, args.T + 1, 20))
        log_mask = np.isin(used_lags, log_lags)
        fixed_mask = np.isin(used_lags, fixed_lags)
        # Ŝ = Δt × Σ ρ² (Riemann sum, truncated at noise floor 1/N)
        N = args.n_samples
        nf = 1.0 / N
        delta_fixed20 = 20 * args.dt
        if fixed_mask.any():
            rho_f20 = rho_sq[fixed_mask]
            lags_f20 = used_lags[fixed_mask]
            below_nf = rho_f20 < nf
            if below_nf.any():
                trunc_idx = np.argmax(below_nf)
                shat_fixed20 = float(delta_fixed20 * rho_f20[:trunc_idx].sum())
                tau_star_fixed20 = float(lags_f20[trunc_idx] * args.dt)
                hit_floor_fixed20 = True
            else:
                shat_fixed20 = float(delta_fixed20 * rho_f20.sum())
                tau_star_fixed20 = float(lags_f20[-1] * args.dt)
                hit_floor_fixed20 = False
        else:
            shat_fixed20 = None
            tau_star_fixed20 = None
            hit_floor_fixed20 = None
        json_results["ssa_log"] = float(np.mean(rho_sq[log_mask])) if log_mask.any() else None
        json_results["ssa_fixed20"] = shat_fixed20
        json_results["tau_star_fixed20"] = tau_star_fixed20
        json_results["hit_floor_fixed20"] = hit_floor_fixed20
        json_results["rho_sq_all"] = rho_sq.tolist()
        json_results["snapshot_lags_all"] = used_lags.tolist()
        json_results["log_lags"] = log_lags
        json_results["fixed20_lags"] = fixed_lags

    if not args.ssa_only:
        json_results.update({
            "da1_optimal_lag": results["da1_optimal_lag"],
            "da1_gap": results["da1_gap"],
            "da1_cosines": da1_cosines,
            "da1_correlations": {k: v for k, v in da1_corr.items()},
            "da2_optimal_lag": results["da2_optimal_lag"],
            "da2_gap": results["da2_gap"],
            "da2_cosines": da2_cosines,
        })
        if "tau_star_freeconv" in results:
            json_results.update({
                "tau_star_freeconv": int(results["tau_star_freeconv"]),
                "freeconv_M": int(results["freeconv_M"]),
                "freeconv_sigma_f_sq": float(results["freeconv_sigma_f_sq"]),
                "freeconv_gamma": float(results["freeconv_gamma"]),
                "freeconv_first_cross_idx": int(results["freeconv_first_cross_idx"]),
            })

    if lc_results is not None:
        json_results["ssa_lc"] = lc_results["ssa"]
        json_results["da1_lc_gap"] = lc_results["da1_gap"]
        json_results["cross_space_da1_spearman"] = float(cross_rho)

    json_path = os.path.join(output_dir, f"clip_da_{run_name}.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    npz_data = {
        "rho_sq": results["rho_sq"],
        "snapshot_lags": np.array(results["snapshot_lags"]),
        "mu_clip": results["mu"],
        "rho_sq_gender": rho_sq_gender,
    }
    if not args.ssa_only:
        npz_data["gap12"] = results["gap12"]
        npz_data["gap23"] = results["gap23"]
        npz_data["da1_eigvec"] = results["da1_eigvec"]
        npz_data["da2_eigvec"] = results["da2_eigvec"]
        for i, lag in enumerate(results["snapshot_lags"]):
            npz_data[f"eigenvalues_lag{lag}"] = results["eigenvalues"][i]
            npz_data[f"eigvecs_lag{lag}"] = results["top_eigvecs"][i]
        # Free-conv stopping outputs
        if "tau_star_freeconv" in results:
            npz_data["tau_star_freeconv"] = np.int64(results["tau_star_freeconv"])
            npz_data["freeconv_M"] = np.int64(results["freeconv_M"])
            npz_data["freeconv_lambda_plus_profile"] = results["freeconv_lambda_plus_profile"]
            npz_data["freeconv_lambda2_profile"] = results["freeconv_lambda2_profile"]
            npz_data["freeconv_sigma_f_sq"] = np.float64(results["freeconv_sigma_f_sq"])
            npz_data["freeconv_gamma"] = np.float64(results["freeconv_gamma"])
            npz_data["freeconv_first_cross_idx"] = np.int64(results["freeconv_first_cross_idx"])
            npz_data["C_tau_star"] = results["C_tau_star"]
            npz_data["eigenvalues_full_at_tau_star"] = results["eigenvalues_full_at_tau_star"]
            npz_data["eigvecs_full_topk_at_tau_star"] = results["eigvecs_full_topk_at_tau_star"]

    npz_path = os.path.join(output_dir, f"clip_da_{run_name}.npz")
    np.savez_compressed(npz_path, **npz_data)
    print(f"Saved NPZ: {npz_path}")

    # Save latents only for full run
    if not name_suffix:
        vis_data = {"x0": x0_np}
        if not args.ssa_only:
            for key_lag in [results["da1_optimal_lag"], results["da2_optimal_lag"]]:
                if key_lag in snapshots_np:
                    vis_data[f"x_lag{key_lag}"] = snapshots_np[key_lag]
        vis_path = os.path.join(output_dir, f"clip_da_latents_{run_name}.npz")
        np.savez_compressed(vis_path, **vis_data)
        print(f"Saved latents: {vis_path}")

    if wandb_run is not None:
        log_data = {"ssa_clip": results["ssa"], "entropy_clip": entropy}
        if not args.ssa_only:
            log_data["da1_gap"] = results["da1_gap"]
            log_data["da1_optimal_lag"] = results["da1_optimal_lag"]
        wandb_run.log(log_data)
        wandb_run.finish()

    # =====================================================================
    # Stage 8: Visualization
    # =====================================================================
    print("\n=== Generating visualizations ===")

    latent_shape = (4, args.height // 8, args.width // 8)
    if not args.ssa_only and not name_suffix:
        try:
            grid_path = os.path.join(output_dir, f"clip_da_{run_name}_da1_grid.png")
            make_topbottom_grid(vae, x0_np, da1_scores, n=8, path=grid_path,
                                latent_shape=latent_shape, device=device)
        except Exception as e:
            print(f"  DA1 grid failed: {e}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if args.ssa_only:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            plot_rho_decay(results["rho_sq"], results["snapshot_lags"],
                           label="CLIP (768D)", ax=ax)
            if args.ssa_min_lag > 0:
                ax.axvline(args.ssa_min_lag, color="gray", ls="--", alpha=0.5,
                           label=f"ssa_min_lag={args.ssa_min_lag}")
                ax.legend()
            fig.suptitle(f"SSA: {run_name} (dt={args.dt}, T={args.T})")
            fig.tight_layout()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            plot_rho_decay(results["rho_sq"], results["snapshot_lags"],
                           label="CLIP (768D)", ax=axes[0])
            if lc_results is not None:
                plot_rho_decay(lc_results["rho_sq"], lc_results["snapshot_lags"],
                               label="LC (512D)", ax=axes[0])
            plot_gap_profile(results["gap12"], results["gap23"],
                             results["snapshot_lags"], ax=axes[1])
            fig.suptitle(f"SSA + DA: {run_name}")
            fig.tight_layout()
        fig_path = os.path.join(output_dir, f"clip_da_{run_name}_curves.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Saved curves: {fig_path}")
    except Exception as e:
        print(f"  Plot failed: {e}")

    total_time = gen_time + langevin_time + embed_time + ssa_time
    print(f"\n{'='*60}")
    print(f"Summary: {run_name} (CLIP-mode{'  SSA-only' if args.ssa_only else ''})")
    print(f"  N = {z0_clip.shape[0]}" + (f" (filtered from {args.n_samples})" if name_suffix else ""))
    print(f"  Model: {args.model_id}")
    print(f"  Guidance: {args.guidance_scale}")
    print(f"  dt={args.dt}, T={args.T}")
    print(f"  SSA_CLIP = {results['ssa']:.6f}" + (f" (lags >= {args.ssa_min_lag})" if args.ssa_min_lag > 0 else ""))
    if args.ssa_min_lag > 0:
        print(f"  SSA_CLIP_all = {results.get('ssa_all', 'N/A')}")
    print(f"  SSA_gender_sliced = {ssa_gender_sliced:.6f}")
    print(f"  H_gender_1d = {gender_entropy_1d:.4f}")
    if ssa_glasses_sliced is not None:
        print(f"  SSA_glasses_sliced = {ssa_glasses_sliced:.6f}")
    if entropy is not None:
        print(f"  Entropy (kNN full) = {entropy:.4f}")
    if not args.ssa_only:
        print(f"  DA1: gap={results['da1_gap']:.2f}x at lag={results['da1_optimal_lag']}")
        print(f"  DA2: gap={results['da2_gap']:.2f}x at lag={results['da2_optimal_lag']}")
        print(f"  DA1 top cosine: {max(da1_cosines.items(), key=lambda x: abs(x[1]))}")
    if lc_results is not None:
        print(f"  SSA_LC = {lc_results['ssa']:.6f}")
        print(f"  Cross-space DA1 Spearman: {cross_rho:.4f}")
    if outlier_info is not None:
        print(f"  Outliers removed: {outlier_info['n_outliers_removed']}")
    print(f"  Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(EXP_DIR, "results_clip_da")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_shape = (4, args.height // 8, args.width // 8)
    d_latent = int(np.prod(latent_shape))

    use_controlnet = args.controlnet_model_id is not None
    if use_controlnet and args.control_image_path is None:
        raise ValueError("--control_image_path required when using --controlnet_model_id")

    print(f"=== run_ssa_da_clip.py (CLIP-mode SSA+DA) ===")
    print(f"Model: {args.model_id}")
    print(f"Name: {args.name}")
    print(f"Device: {device}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Guidance scale: {args.guidance_scale}")
    if use_controlnet:
        print(f"ControlNet: {args.controlnet_model_id}")
        print(f"Control image: {args.control_image_path}")
        print(f"ControlNet scale: {args.controlnet_scale}")
    print(f"Latent shape: {latent_shape} ({d_latent}D)")
    print(f"Config: n_samples={args.n_samples}, n_traj={args.n_traj}, "
          f"T={args.T}, dt={args.dt}, t_score={args.t_score}")
    print(f"CLIP model: {args.clip_model}")

    # =====================================================================
    # Stages 1+2: Generate + Langevin (or resume)
    # =====================================================================
    if args.resume_checkpoint is not None:
        (vae, x0_np, snapshots_np, snapshot_lags, var_f,
         gen_time, langevin_time, wandb_run) = run_stages_1_2_resume(
            args, output_dir, device, use_controlnet)
    elif args.resume_x0 is not None:
        (vae, x0_np, snapshots_np, snapshot_lags, var_f,
         gen_time, langevin_time, wandb_run) = run_stages_1_2_resume_x0(
            args, output_dir, device, latent_shape, use_controlnet)
    else:
        (vae, x0_np, snapshots_np, snapshot_lags, var_f,
         gen_time, langevin_time, wandb_run) = run_stages_1_2(
            args, output_dir, device, latent_shape, use_controlnet)

    # =====================================================================
    # Stage 3: CLIP embedding
    # =====================================================================
    print("\n=== Loading real CLIP embedder ===")
    clip_embedder = RealCLIPEmbedder(
        vae, clip_model_name=args.clip_model, device=device,
        clip_batch_size=args.clip_batch_size,
    )

    print(f"\n=== Embedding x0 ({x0_np.shape[0]} samples) via real CLIP ===")
    t0 = time.time()
    z0_clip = clip_embedder.embed(x0_np, latent_shape=latent_shape)  # (N, 768)
    print(f"  z0 embedded in {time.time() - t0:.1f}s, shape={z0_clip.shape}")

    # Determinism check (fp16 VAE has minor numerical noise, use atol=1e-3)
    z0_check = clip_embedder.embed(x0_np[:1], latent_shape=latent_shape)
    max_diff = np.abs(z0_clip[:1] - z0_check).max()
    print(f"  Determinism check: max_diff={max_diff:.6f}", end="")
    if max_diff < 1e-3:
        print(" PASSED")
    else:
        print(f" WARNING: max_diff={max_diff:.6f} > 1e-3 (fp16 noise, continuing)")

    # Embed all snapshots
    z_snapshots = {}
    print(f"\n=== Embedding {len(snapshot_lags)} snapshots via real CLIP ===")
    t0 = time.time()
    for lag in snapshot_lags:
        if lag not in snapshots_np:
            continue
        z_snap = clip_embedder.embed(snapshots_np[lag], latent_shape=latent_shape)
        z_snapshots[lag] = z_snap
        print(f"  lag={lag}: embedded, shape={z_snap.shape}")
    embed_time = time.time() - t0
    print(f"  All snapshots embedded in {embed_time:.1f}s")

    # =====================================================================
    # Stage 4-8: Analysis loop (full data, then filtered if requested)
    # =====================================================================
    # Build list of (z0, z_snaps, name_suffix, outlier_info) to analyze
    analysis_runs = [
        (z0_clip, z_snapshots, "", None),  # full data, always
    ]

    if args.clip_outlier_sigma is not None:
        z0_mean = z0_clip.mean(axis=0)
        dists = np.linalg.norm(z0_clip - z0_mean, axis=1)
        mean_dist = dists.mean()
        std_dist = dists.std()
        threshold = mean_dist + args.clip_outlier_sigma * std_dist
        good_mask = dists <= threshold
        n_removed = int((~good_mask).sum())
        outlier_indices = np.where(~good_mask)[0].tolist()
        print(f"\n=== Outlier removal: {n_removed}/{len(z0_clip)} removed "
              f"(L2 > mean+{args.clip_outlier_sigma}*sigma = {mean_dist:.4f}+{args.clip_outlier_sigma}*{std_dist:.4f} = {threshold:.4f}) ===")
        if n_removed > 0:
            print(f"  Outlier indices: {outlier_indices}")
            print(f"  Outlier L2 distances: {dists[~good_mask].tolist()}")

        if good_mask.sum() < 10:
            print(f"  WARNING: Only {good_mask.sum()} samples remain after filtering, skipping filtered analysis")
        else:
            z0_filtered = z0_clip[good_mask]
            z_snaps_filtered = {lag: z[good_mask] for lag, z in z_snapshots.items()}
            outlier_info = {
                "n_outliers_removed": n_removed,
                "outlier_indices": outlier_indices,
                "outlier_sigma": args.clip_outlier_sigma,
                "outlier_threshold": float(threshold),
                "outlier_mean_dist": float(mean_dist),
                "outlier_std_dist": float(std_dist),
                "n_remaining": int(good_mask.sum()),
            }
            analysis_runs.append(
                (z0_filtered, z_snaps_filtered, "_filtered", outlier_info)
            )

    # Text probe analyzer (shared across runs)
    print("\n=== Loading text probe analyzer ===")
    analyzer = TextProbeAnalyzer(clip_model_name=args.clip_model, device=device)
    analyzer.define_probes()

    for run_z0, run_z_snaps, name_suffix, outlier_info in analysis_runs:
        run_name = args.name + name_suffix
        run_label = "FILTERED" if name_suffix else "FULL"
        print(f"\n{'='*60}")
        print(f"=== Analysis pass: {run_label} (N={len(run_z0)}) ===")
        print(f"{'='*60}")

        _run_analysis(
            args=args, output_dir=output_dir, device=device, vae=vae,
            x0_np=x0_np, snapshots_np=snapshots_np,
            z0_clip=run_z0, z_snapshots=run_z_snaps,
            snapshot_lags=snapshot_lags,
            analyzer=analyzer, use_controlnet=use_controlnet,
            var_f=var_f, gen_time=gen_time, langevin_time=langevin_time,
            embed_time=embed_time,
            run_name=run_name, name_suffix=name_suffix,
            outlier_info=outlier_info,
            wandb_run=wandb_run if not name_suffix else None,
        )


if __name__ == "__main__":
    main()
