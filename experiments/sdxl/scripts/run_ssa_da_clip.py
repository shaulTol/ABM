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
    return p.parse_args()


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
    # Stage 1: Generate latents (same as v2)
    # =====================================================================

    # Load ControlNet
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

    # Load SDXL pipeline
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

    # Encode prompt
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

    # Generate latents
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

    snapshot_lags = sorted(set(int(round(x)) for x in np.linspace(10, args.T, 20)))
    if args.T not in snapshot_lags:
        snapshot_lags.append(args.T)
    snapshot_lags = sorted(snapshot_lags)

    print(f"\n=== Langevin dynamics ({args.n_traj} traj x {args.T} steps) ===")
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
    # Stage 3: Free GPU for CLIP, embed all snapshots
    # =====================================================================
    print("\n=== Freeing GPU for CLIP embedding ===")
    del score_fn
    if controlnet is not None:
        controlnet = controlnet.to("cpu")
    # Keep VAE on GPU (needed for decode), offload UNet
    vae = pipe.vae.to(device).float().eval()
    pipe.unet = pipe.unet.to("cpu")
    torch.cuda.empty_cache()

    # Convert snapshots to numpy (already on CPU)
    x0_np = x0_flat.numpy()
    snapshots_np = {lag: snap.numpy() for lag, snap in snapshots.items()}

    # Build CLIP embedder (reuses VAE already on GPU)
    print("\n=== Loading real CLIP embedder ===")
    clip_embedder = RealCLIPEmbedder(
        vae, clip_model_name=args.clip_model, device=device,
        clip_batch_size=args.clip_batch_size,
    )

    # Embed x0
    print(f"\n=== Embedding x0 ({x0_np.shape[0]} samples) via real CLIP ===")
    t0 = time.time()
    z0_clip = clip_embedder.embed(x0_np)  # (N, 768)
    print(f"  z0 embedded in {time.time() - t0:.1f}s, shape={z0_clip.shape}")

    # Determinism check (fp16 VAE has minor numerical noise, use atol=1e-3)
    z0_check = clip_embedder.embed(x0_np[:1])
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
        z_snap = clip_embedder.embed(snapshots_np[lag])
        z_snapshots[lag] = z_snap
        print(f"  lag={lag}: embedded, shape={z_snap.shape}")
    embed_time = time.time() - t0
    print(f"  All snapshots embedded in {embed_time:.1f}s")

    # =====================================================================
    # Stage 4: Compute SSA + DA in CLIP space
    # =====================================================================
    print("\n=== Computing SSA + DA in CLIP space (768D) ===")
    t0 = time.time()
    results = compute_ssa_da(z0_clip, z_snapshots, snapshot_lags)
    ssa_time = time.time() - t0
    print(f"  SSA+DA computed in {ssa_time:.1f}s")

    # Verify rho(0) = 1.0
    rho0_check = np.sum(
        (z0_clip - z0_clip.mean(0)) ** 2, axis=-1
    ).mean()
    print(f"  C(0) trace check: {rho0_check:.6f}")

    # =====================================================================
    # Stage 5: Text probe analysis on DA directions
    # =====================================================================
    print("\n=== Text probe analysis ===")
    analyzer = TextProbeAnalyzer(clip_model_name=args.clip_model, device=device)
    analyzer.define_probes()

    # Direct cosine: DA direction vs text probes (works because both in CLIP space)
    da1_cosines = analyzer.score_direction(results["da1_eigvec"])
    da2_cosines = analyzer.score_direction(results["da2_eigvec"])

    print("\n  DA1 direction cosines with text probes:")
    for name in analyzer.probe_names:
        print(f"    {name:<15} cos = {da1_cosines[name]:>7.4f}")

    print("\n  DA2 direction cosines with text probes:")
    for name in analyzer.probe_names:
        print(f"    {name:<15} cos = {da2_cosines[name]:>7.4f}")

    # Per-sample: project z0 onto DA1, correlate with probe scores
    da1_scores = (z0_clip - results["mu"]) @ results["da1_eigvec"]
    da1_corr = analyzer.correlate_with_ranking(z0_clip, da1_scores)

    print("\n  DA1 per-sample Spearman with text probes:")
    for name in analyzer.probe_names:
        r = da1_corr[name]
        print(f"    {name:<15} rho = {r['rho']:>7.4f} (p={r['pvalue']:.2e})")

    # =====================================================================
    # Stage 5a: Gender-sliced SSA
    # =====================================================================
    print("\n=== Gender-sliced SSA ===")
    gender_axis = analyzer.probe_axes["gender"]  # (768,) normalized
    z0_c = z0_clip - results["mu"]
    s0_gender = z0_c @ gender_axis  # (N,)
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

    # =====================================================================
    # Stage 5b: Optional LC comparison
    # =====================================================================
    lc_results = None
    if args.also_compute_lc:
        print("\n=== Also computing SSA+DA in Latent-CLIP space (512D) ===")
        lc_embedder = LatentCLIPEmbedder(device)
        z0_lc = lc_embedder.embed(x0_np)
        z_snaps_lc = {}
        for lag in snapshot_lags:
            if lag in snapshots_np:
                z_snaps_lc[lag] = lc_embedder.embed(snapshots_np[lag])
        lc_results = compute_ssa_da(z0_lc, z_snaps_lc, snapshot_lags)

        # Cross-space DA1 comparison (per-sample projection correlation)
        from scipy.stats import spearmanr
        da1_clip_scores = (z0_clip - results["mu"]) @ results["da1_eigvec"]
        da1_lc_scores = (z0_lc - lc_results["mu"]) @ lc_results["da1_eigvec"]
        cross_rho, cross_p = spearmanr(da1_clip_scores, da1_lc_scores)
        print(f"\n  Cross-space DA1 Spearman: rho={cross_rho:.4f} (p={cross_p:.2e})")

    # =====================================================================
    # Stage 6: Entropy (optional)
    # =====================================================================
    entropy = None
    if args.compute_entropy:
        print("\n=== Computing kNN entropy in CLIP space ===")
        entropy = compute_knn_entropy(torch.from_numpy(z0_clip), k=args.entropy_k)
        print(f"  kNN entropy (k={args.entropy_k}): {entropy:.4f}")

    # =====================================================================
    # Stage 7: Visualization
    # =====================================================================
    print("\n=== Generating visualizations ===")

    # Top/bottom grid for DA1
    grid_path = os.path.join(output_dir, f"clip_da_{args.name}_da1_grid.png")
    make_topbottom_grid(vae, x0_np, da1_scores, n=8, path=grid_path, device=device)

    # Rho decay + gap profile plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_rho_decay(results["rho_sq"], results["snapshot_lags"],
                       label="CLIP (768D)", ax=axes[0])
        if lc_results is not None:
            plot_rho_decay(lc_results["rho_sq"], lc_results["snapshot_lags"],
                           label="LC (512D)", ax=axes[0])
        plot_gap_profile(results["gap12"], results["gap23"],
                         results["snapshot_lags"], ax=axes[1])
        fig.suptitle(f"SSA + DA: {args.name}")
        fig.tight_layout()
        fig_path = os.path.join(output_dir, f"clip_da_{args.name}_curves.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Saved curves: {fig_path}")
    except Exception as e:
        print(f"  Plot failed: {e}")

    # =====================================================================
    # Stage 8: Save results
    # =====================================================================
    json_results = {
        "name": args.name,
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
        "n_samples": args.n_samples,
        "n_traj": args.n_traj,
        "T": args.T,
        "dt": args.dt,
        "t_score": args.t_score,
        "var_f": var_f,
        "embed_dim": int(results["embed_dim"]),
        "ssa_clip": results["ssa"],
        "entropy_clip": entropy,
        "da1_optimal_lag": results["da1_optimal_lag"],
        "da1_gap": results["da1_gap"],
        "da1_cosines": da1_cosines,
        "da1_correlations": {k: v for k, v in da1_corr.items()},
        "da2_optimal_lag": results["da2_optimal_lag"],
        "da2_gap": results["da2_gap"],
        "da2_cosines": da2_cosines,
        "ssa_gender_sliced": ssa_gender_sliced,
        "rho_sq_gender": rho_sq_gender.tolist(),
        "gen_time_s": gen_time,
        "langevin_time_s": langevin_time,
        "embed_time_s": embed_time,
        "ssa_time_s": ssa_time,
    }

    if lc_results is not None:
        json_results["ssa_lc"] = lc_results["ssa"]
        json_results["da1_lc_gap"] = lc_results["da1_gap"]
        json_results["cross_space_da1_spearman"] = float(cross_rho)

    json_path = os.path.join(output_dir, f"clip_da_{args.name}.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Save curves + eigenvectors
    npz_data = {
        "rho_sq": results["rho_sq"],
        "snapshot_lags": np.array(results["snapshot_lags"]),
        "gap12": results["gap12"],
        "gap23": results["gap23"],
        "mu_clip": results["mu"],
        "da1_eigvec": results["da1_eigvec"],
        "da2_eigvec": results["da2_eigvec"],
        "rho_sq_gender": rho_sq_gender,
    }
    for i, lag in enumerate(results["snapshot_lags"]):
        npz_data[f"eigenvalues_lag{lag}"] = results["eigenvalues"][i]
        npz_data[f"eigvecs_lag{lag}"] = results["top_eigvecs"][i]

    npz_path = os.path.join(output_dir, f"clip_da_{args.name}.npz")
    np.savez_compressed(npz_path, **npz_data)
    print(f"Saved NPZ: {npz_path}")

    # Save latent snapshots
    vis_data = {"x0": x0_np}
    for key_lag in [results["da1_optimal_lag"], results["da2_optimal_lag"]]:
        if key_lag in snapshots_np:
            vis_data[f"x_lag{key_lag}"] = snapshots_np[key_lag]
    vis_path = os.path.join(output_dir, f"clip_da_latents_{args.name}.npz")
    np.savez_compressed(vis_path, **vis_data)
    print(f"Saved latents: {vis_path}")

    # --- Wandb ---
    if wandb_run is not None:
        wandb_run.log({
            "ssa_clip": results["ssa"],
            "entropy_clip": entropy,
            "da1_gap": results["da1_gap"],
            "da1_optimal_lag": results["da1_optimal_lag"],
        })
        wandb_run.finish()

    # --- Summary ---
    total_time = gen_time + langevin_time + embed_time + ssa_time
    print(f"\n{'='*60}")
    print(f"Summary: {args.name} (CLIP-mode)")
    print(f"  Model: {args.model_id}")
    print(f"  Guidance: {args.guidance_scale}")
    print(f"  SSA_CLIP = {results['ssa']:.6f}")
    print(f"  SSA_gender_sliced = {ssa_gender_sliced:.6f}")
    if entropy is not None:
        print(f"  Entropy = {entropy:.4f}")
    print(f"  DA1: gap={results['da1_gap']:.2f}x at lag={results['da1_optimal_lag']}")
    print(f"  DA2: gap={results['da2_gap']:.2f}x at lag={results['da2_optimal_lag']}")
    print(f"  DA1 top cosine: {max(da1_cosines.items(), key=lambda x: abs(x[1]))}")
    if lc_results is not None:
        print(f"  SSA_LC = {lc_results['ssa']:.6f}")
        print(f"  Cross-space DA1 Spearman: {cross_rho:.4f}")
    print(f"  Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")


if __name__ == "__main__":
    main()
