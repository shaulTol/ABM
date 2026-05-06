"""
compute_sdxl_ssa_v2.py — Compute SSA + DA + kNN entropy for SDXL models in latent-CLIP space.

v2 optimizations over v1:
- UNet stays in fp16 (2x faster on tensor cores), only Langevin state is fp32
- Larger score_batch_size (128 default, fits in fp16 VRAM)
- Better progress logging (flush=True, rate + ETA)
- Linear snapshot lag spacing (more resolution at longer lags)
- wandb integration for live monitoring
- Optional torch.compile for UNet

Pipeline per run (one model + one guidance scale):
1. Load SDXL pipeline (Base or Turbo)
2. Generate N latents via pipeline(output_type="latent")
3. Compute var_f from latents
4. kNN entropy in Latent-CLIP 512D
5. Build SDXLScore (Tweedie + CFG in latent space)
6. Langevin in flat 16384D latent space, snapshot at linear lags
7. Project snapshots to CLIP via LatentCLIPEmbedder
8. Compute C_clip(tau), eigendecompose, DA gap profiles, SSA

Usage:
    # Quick test:
    python scripts/compute_sdxl_ssa_v2.py \
        --model_id stabilityai/stable-diffusion-xl-base-1.0 \
        --name test --prompt "A photo of a human face" \
        --n_samples 50 --n_traj 50 --T 100 --gen_steps 10

    # Full guidance sweep (SLURM):
    python scripts/compute_sdxl_ssa_v2.py \
        --model_id stabilityai/stable-diffusion-xl-base-1.0 \
        --name sdxl_base_w7.5 --prompt "A photo of a human face" \
        --guidance_scale 7.5 --gen_steps 50 \
        --n_samples 2000 --n_traj 2000 --T 5000 --dt 0.01 --t_score 1
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
REPO_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))


# ---------------------------------------------------------------------------
# SDXL Score Wrapper (Tweedie + CFG in latent space) — fp16 UNet, fp32 I/O
# ---------------------------------------------------------------------------

class SDXLScore:
    """Wraps an SDXL UNet as score function via Tweedie formula with CFG.

    score(x) = -eps_theta(x, t, cond) / sigma_t

    With CFG (guidance_scale > 1):
        eps = eps_uncond + w * (eps_cond - eps_uncond)

    The UNet runs in its native dtype (fp16) for speed.
    Input x is cast fp32->fp16 at the boundary, output score is cast back to fp32.
    """

    def __init__(self, unet, scheduler, t, prompt_embeds, pooled_prompt_embeds,
                 time_ids, device, latent_shape=(4, 64, 64), batch_size=128,
                 guidance_scale=0.0, negative_prompt_embeds=None,
                 negative_pooled_prompt_embeds=None,
                 controlnet=None, control_image=None,
                 controlnet_conditioning_scale=0.8):
        self.unet = unet
        self.device = device
        self.latent_shape = latent_shape
        self.d = int(np.prod(latent_shape))
        self.batch_size = batch_size
        self.guidance_scale = guidance_scale

        # ControlNet
        self.controlnet = controlnet
        self.control_image = control_image  # (1, 3, H, W) tensor on device
        self.controlnet_conditioning_scale = controlnet_conditioning_scale

        # UNet dtype (stays fp16)
        self.unet_dtype = next(unet.parameters()).dtype

        # Precompute noise schedule constants
        alpha_bar_t = scheduler.alphas_cumprod[t].item()
        self.sigma_t = np.sqrt(1.0 - alpha_bar_t)
        self.t_tensor = torch.tensor([t], dtype=torch.long, device=device)

        # Text conditioning — keep in UNet dtype (fp16)
        self.prompt_embeds = prompt_embeds.to(device=device, dtype=self.unet_dtype)
        self.pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=self.unet_dtype)
        self.time_ids = time_ids.to(device=device, dtype=self.unet_dtype)

        # For CFG: uncond embeddings
        self.do_cfg = guidance_scale > 1.0
        if self.do_cfg:
            assert negative_prompt_embeds is not None
            self.negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=self.unet_dtype)
            self.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device=device, dtype=self.unet_dtype)

        cn_str = f", controlnet_scale={controlnet_conditioning_scale}" if controlnet else ""
        print(f"  SDXLScore: t={t}, alpha_bar={alpha_bar_t:.6f}, "
              f"sigma_t={self.sigma_t:.6f}, guidance_scale={guidance_scale}, "
              f"batch_size={batch_size}, unet_dtype={self.unet_dtype}{cn_str}")

    @torch.no_grad()
    def __call__(self, x_flat):
        """Score function: (N, 16384) fp32 -> (N, 16384) fp32."""
        N = x_flat.shape[0]
        if N <= self.batch_size:
            return self._score_batch(x_flat)

        scores = []
        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            scores.append(self._score_batch(x_flat[start:end]))
        return torch.cat(scores, dim=0)

    def _get_controlnet_residuals(self, latent_input, t_input, embeds_input,
                                    added_cond_kwargs_input):
        """Run ControlNet to get residuals for UNet injection."""
        # Expand control image to match batch size
        control_img = self.control_image.expand(latent_input.shape[0], -1, -1, -1)
        control_img = control_img.to(dtype=self.unet_dtype)

        down_residuals, mid_residual = self.controlnet(
            latent_input, t_input,
            encoder_hidden_states=embeds_input,
            controlnet_cond=control_img,
            conditioning_scale=self.controlnet_conditioning_scale,
            added_cond_kwargs=added_cond_kwargs_input,
            return_dict=False,
        )
        return down_residuals, mid_residual

    def _score_batch(self, x_flat):
        """Score for a single batch. Casts fp32->fp16 for UNet, back to fp32."""
        N = x_flat.shape[0]
        # Cast to UNet dtype (fp16) at the boundary
        x_latent = x_flat.view(N, *self.latent_shape).to(self.unet_dtype)
        t_batch = self.t_tensor.expand(N)

        # Expand text embeddings to batch
        prompt_embeds = self.prompt_embeds.expand(N, -1, -1)
        pooled = self.pooled_prompt_embeds.expand(N, -1)
        time_ids = self.time_ids.expand(N, -1)

        added_cond_kwargs = {
            "text_embeds": pooled,
            "time_ids": time_ids,
        }

        if self.do_cfg:
            # Concat uncond + cond for CFG
            neg_embeds = self.negative_prompt_embeds.expand(N, -1, -1)
            neg_pooled = self.negative_pooled_prompt_embeds.expand(N, -1)
            neg_time_ids = self.time_ids.expand(N, -1)

            latent_input = torch.cat([x_latent, x_latent], dim=0)
            t_input = torch.cat([t_batch, t_batch], dim=0)
            embeds_input = torch.cat([neg_embeds, prompt_embeds], dim=0)
            pooled_input = torch.cat([neg_pooled, pooled], dim=0)
            time_ids_input = torch.cat([neg_time_ids, time_ids], dim=0)

            added_cond_kwargs_input = {
                "text_embeds": pooled_input,
                "time_ids": time_ids_input,
            }

            # ControlNet residuals (if available)
            cn_kwargs = {}
            if self.controlnet is not None:
                down_res, mid_res = self._get_controlnet_residuals(
                    latent_input, t_input, embeds_input, added_cond_kwargs_input)
                cn_kwargs["down_block_additional_residuals"] = down_res
                cn_kwargs["mid_block_additional_residual"] = mid_res

            eps_pred = self.unet(
                latent_input, t_input,
                encoder_hidden_states=embeds_input,
                added_cond_kwargs=added_cond_kwargs_input,
                return_dict=False,
                **cn_kwargs,
            )[0]

            eps_uncond, eps_cond = eps_pred.chunk(2)
            eps_pred = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
        else:
            # ControlNet residuals (if available)
            cn_kwargs = {}
            if self.controlnet is not None:
                down_res, mid_res = self._get_controlnet_residuals(
                    x_latent, t_batch, prompt_embeds, added_cond_kwargs)
                cn_kwargs["down_block_additional_residuals"] = down_res
                cn_kwargs["mid_block_additional_residual"] = mid_res

            eps_pred = self.unet(
                x_latent, t_batch,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
                **cn_kwargs,
            )[0]

        # Cast back to fp32 for Langevin numerics
        score = -eps_pred.float() / self.sigma_t
        return score.view(N, -1)


# ---------------------------------------------------------------------------
# Latent-CLIP Embedder
# ---------------------------------------------------------------------------

class LatentCLIPEmbedder:
    """Maps VAE latents (4x64x64) to CLIP space (512D) without pixel decoding."""

    def __init__(self, device, checkpoint_path=None):
        import latent_clip

        if checkpoint_path is None:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(
                "wendlerc/latent-clip-b-8-512-34b-80k", "checkpoints/epoch_34.pt"
            )

        self.model, _, _ = latent_clip.create_model_and_transforms(
            "Latent-ViT-B-8-512", pretrained=checkpoint_path, device=device,
        )
        self.model.eval()
        self.device = device
        self.embed_dim = 512
        print(f"  LatentCLIPEmbedder loaded: embed_dim={self.embed_dim}")

    @torch.no_grad()
    def embed(self, latents_flat, latent_shape=(4, 64, 64), batch_size=512):
        """Embed flat latent vectors into CLIP space.

        Args:
            latents_flat: (N, 16384) in SDXL latent range
            latent_shape: (4, 64, 64)
            batch_size: Process in batches

        Returns:
            (N, 512) L2-normalized CLIP embeddings
        """
        N = latents_flat.shape[0]
        embeddings = []

        # Undo VAE scaling factor (SDXL uses 0.13025)
        VAE_SCALE = 0.13025

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = latents_flat[start:end].to(self.device)

            # Reshape to (B, 4, 64, 64) and undo scaling
            z = batch.view(-1, *latent_shape) / VAE_SCALE

            # Latent-CLIP expects (B, 4, 64, 64)
            emb = self.model.encode_image(z)

            # L2 normalize
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu())

        return torch.cat(embeddings, dim=0)


# ---------------------------------------------------------------------------
# Langevin dynamics in latent space (snapshot-based) — v2 with better logging
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_langevin_latent_space(score_fn, samples_flat, var_f, n_traj,
                              T, dt, snapshot_lags, device, seed=42,
                              wandb_run=None):
    """Euler-Maruyama Langevin in flat 16384D latent space.

    Stores full state at specified snapshot lags (on CPU) for later
    CLIP projection and SSA/DA computation.

    Args:
        score_fn: SDXLScore callable (N, d) -> (N, d)
        samples_flat: (M, d) generated latents (flat) for initial conditions
        var_f: scalar diffusion coefficient
        n_traj: number of trajectories
        T: total Langevin steps
        dt: timestep
        snapshot_lags: sorted list of lag values at which to snapshot
        device: torch device
        seed: random seed
        wandb_run: optional wandb run for logging

    Returns:
        dict: lag -> (n_traj, d) tensor on CPU
        x0: (n_traj, d) initial positions on CPU
    """
    d = samples_flat.shape[1]
    sigma_f = np.sqrt(var_f)
    sqrt_dt = np.sqrt(dt)

    torch.manual_seed(seed)
    idx = torch.randint(0, samples_flat.shape[0], (n_traj,))
    x = samples_flat[idx].to(device)
    x0 = x.clone().cpu()

    snapshots = {}
    snap_set = set(snapshot_lags)
    t_start = time.time()

    for t in range(1, T + 1):
        grad = score_fn(x)
        drift = 0.5 * var_f * dt * grad
        noise = sigma_f * sqrt_dt * torch.randn_like(x)
        x = x + drift + noise

        if t in snap_set:
            snapshots[t] = x.clone().cpu()

        if t % 100 == 0 or t == T:
            x_norm = x.norm(dim=-1).mean().item()
            score_norm = grad.norm(dim=-1).mean().item()
            elapsed = time.time() - t_start
            rate = t / elapsed
            eta = (T - t) / rate / 3600 if rate > 0 else 0
            print(f"    Langevin step {t}/{T}, ||x||={x_norm:.2f}, "
                  f"||score||={score_norm:.2f}, {rate:.2f} steps/s, "
                  f"ETA {eta:.1f}h", flush=True)

            if wandb_run is not None and t % 100 == 0:
                wandb_run.log({
                    "langevin/step": t,
                    "langevin/x_norm": x_norm,
                    "langevin/score_norm": score_norm,
                    "langevin/steps_per_sec": rate,
                    "langevin/eta_hours": eta,
                }, step=t)

    return snapshots, x0


# ---------------------------------------------------------------------------
# SSA + DA from CLIP-projected snapshots
# ---------------------------------------------------------------------------

def compute_ssa_and_da(clip_embedder, x0_flat, snapshots, snapshot_lags,
                       latent_shape=(4, 64, 64)):
    """Compute SSA and DA from Langevin snapshots projected to CLIP space.

    Args:
        clip_embedder: LatentCLIPEmbedder instance
        x0_flat: (N, d) initial positions
        snapshots: dict lag -> (N, d) positions
        snapshot_lags: sorted list of lags
        latent_shape: (4, 64, 64)

    Returns:
        dict with SSA, rho_sq curve, DA eigenvalues/vectors/gaps, optimal lags
    """
    N = x0_flat.shape[0]

    # Embed x0
    print("  Embedding x0 into CLIP space...")
    z0_clip = clip_embedder.embed(x0_flat, latent_shape)  # (N, 512)
    mu_clip = z0_clip.mean(dim=0)  # (512,)
    z0_centered = z0_clip - mu_clip.unsqueeze(0)

    # tr(C(0))
    tr_c0 = (z0_centered ** 2).sum(dim=-1).mean().item()
    print(f"  tr(C_clip(0)) = {tr_c0:.6f}")
    if tr_c0 < 1e-12:
        print("  WARNING: tr(C(0)) near zero")
        return {"ssa": 0.0, "rho_sq": np.zeros(len(snapshot_lags))}

    rho_sq = []
    eigenvalues_list = []
    gap12_list = []
    gap23_list = []
    top_eigvecs_list = []
    d_clip = z0_clip.shape[1]

    for i, lag in enumerate(snapshot_lags):
        if lag not in snapshots:
            continue
        print(f"  Processing lag={lag}...", flush=True)

        # Embed x_tau
        z_tau_clip = clip_embedder.embed(snapshots[lag], latent_shape)
        z_tau_centered = z_tau_clip - mu_clip.unsqueeze(0)

        # tr(C(tau)) for SSA
        tr_c_tau = (z0_centered * z_tau_centered).sum(dim=-1).mean().item()
        rho = tr_c_tau / tr_c0
        rho_sq.append(rho ** 2)

        # Full C_clip(tau) for DA
        z0_c = z0_centered.float()
        z_tau_c = z_tau_centered.float()
        C_tau = (z_tau_c.T @ z0_c) / N  # (512, 512)
        C_tau = 0.5 * (C_tau + C_tau.T)

        eigvals, eigvecs = torch.linalg.eigh(C_tau)
        idx = eigvals.argsort(descending=True)
        eigvals = eigvals[idx].numpy()
        eigvecs = eigvecs[:, idx].numpy()

        eigenvalues_list.append(eigvals[:10])
        top_eigvecs_list.append(eigvecs[:, :3])

        abs_eig = np.abs(eigvals)
        g12 = abs_eig[0] / max(abs_eig[1], 1e-12)
        g23 = abs_eig[1] / max(abs_eig[2], 1e-12)
        gap12_list.append(g12)
        gap23_list.append(g23)

    print()
    rho_sq = np.array(rho_sq)
    ssa = float(np.mean(rho_sq))

    # DA: find optimal lag per direction (k=1,2,3)
    gap12 = np.array(gap12_list)
    gap23 = np.array(gap23_list)

    # DA1: max gap12
    opt1_idx = int(np.argmax(gap12))
    # DA2: max gap23
    opt2_idx = int(np.argmax(gap23))

    used_lags = [l for l in snapshot_lags if l in snapshots]

    result = {
        "ssa": ssa,
        "rho_sq": rho_sq,
        "snapshot_lags": used_lags,
        "eigenvalues": eigenvalues_list,
        "gap12": gap12,
        "gap23": gap23,
        "top_eigvecs": top_eigvecs_list,
        "mu_clip": mu_clip.numpy(),
        # DA1 optimal
        "da1_optimal_lag": used_lags[opt1_idx],
        "da1_gap": float(gap12[opt1_idx]),
        "da1_eigvec": top_eigvecs_list[opt1_idx][:, 0],  # (512,)
        # DA2 optimal
        "da2_optimal_lag": used_lags[opt2_idx],
        "da2_gap": float(gap23[opt2_idx]),
        "da2_eigvec": top_eigvecs_list[opt2_idx][:, 1],  # (512,)
    }

    print(f"  SSA = {ssa:.6f}")
    print(f"  DA1: gap={gap12[opt1_idx]:.2f}x at lag={used_lags[opt1_idx]}")
    print(f"  DA2: gap={gap23[opt2_idx]:.2f}x at lag={used_lags[opt2_idx]}")

    return result


# ---------------------------------------------------------------------------
# kNN Entropy in CLIP space
# ---------------------------------------------------------------------------

def compute_knn_entropy(clip_embeddings, k=10):
    """Compute kNN entropy estimate in CLIP space.

    H_knn = d/N * sum(log(r_k)) + log(N) + log(V_d) - digamma(k)
    where r_k is the distance to k-th neighbor and V_d is unit ball volume.
    """
    from scipy.special import digamma
    from scipy.spatial import cKDTree

    X = clip_embeddings.numpy().astype(np.float64)
    N, d = X.shape

    tree = cKDTree(X)
    dists, _ = tree.query(X, k=k + 1)  # +1 because self is included
    r_k = dists[:, k]  # k-th neighbor distance (0-indexed: [0]=self, [k]=k-th)

    # Clip tiny distances
    r_k = np.maximum(r_k, 1e-30)

    # Log volume of d-dimensional unit ball: V_d = pi^(d/2) / Gamma(d/2 + 1)
    from scipy.special import gammaln
    log_V_d = (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1)

    H = d * np.mean(np.log(r_k)) + np.log(N) + log_V_d - digamma(k)

    return float(H)


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_latents(pipe, prompt, n_samples, guidance_scale, gen_steps,
                     height, width, batch_size=4, seed=42,
                     control_image=None, controlnet_conditioning_scale=0.8):
    """Generate latents from SDXL pipeline (with optional ControlNet).

    Returns:
        latents: (N, 4, H//8, W//8) in latent space
    """
    device = pipe.device
    all_latents = []
    n_generated = 0

    # Extra kwargs for ControlNet pipeline
    cn_kwargs = {}
    if control_image is not None:
        cn_kwargs["image"] = control_image
        cn_kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale

    while n_generated < n_samples:
        bs = min(batch_size, n_samples - n_generated)
        generator = torch.Generator(device=device).manual_seed(seed + n_generated)

        result = pipe(
            prompt=prompt,
            num_images_per_prompt=bs,
            guidance_scale=guidance_scale,
            num_inference_steps=gen_steps,
            height=height,
            width=width,
            generator=generator,
            output_type="latent",
            **cn_kwargs,
        )

        latents = result.images  # (B, 4, H//8, W//8)
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents)
        all_latents.append(latents.cpu())
        n_generated += bs
        print(f"  Generated {n_generated}/{n_samples} latents", flush=True)

    return torch.cat(all_latents, dim=0)[:n_samples]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute SSA + DA + entropy for SDXL in latent-CLIP space (v2: fp16 UNet + wandb)"
    )
    p.add_argument("--model_id", type=str, required=True,
                   help="HuggingFace model ID (e.g. stabilityai/stable-diffusion-xl-base-1.0)")
    p.add_argument("--name", type=str, required=True,
                   help="Run name for output files")
    p.add_argument("--prompt", type=str, default="A photo of a human face")
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--gen_steps", type=int, default=50,
                   help="Number of denoising steps for generation")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--n_samples", type=int, default=2000,
                   help="Number of latent samples to generate")
    p.add_argument("--n_traj", type=int, default=2000,
                   help="Number of Langevin trajectories")
    p.add_argument("--T", type=int, default=2500,
                   help="Total Langevin steps")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--t_score", type=int, default=1,
                   help="Timestep t for Tweedie score (always use 1)")
    p.add_argument("--score_batch_size", type=int, default=128,
                   help="UNet batch size for score evaluation (128 OK in fp16)")
    p.add_argument("--gen_batch_size", type=int, default=4,
                   help="Batch size for sample generation")
    p.add_argument("--compute_entropy", action="store_true",
                   help="Compute kNN entropy in CLIP space")
    p.add_argument("--entropy_k", type=int, default=10)
    p.add_argument("--latent_clip_checkpoint", type=str, default=None,
                   help="Path to Latent-CLIP checkpoint (auto-download if None)")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile_unet", action="store_true",
                   help="Use torch.compile on UNet (experimental, adds compilation overhead)")
    # ControlNet conditioning
    p.add_argument("--controlnet_model_id", type=str, default=None,
                   help="HuggingFace ControlNet model ID (e.g. xinsir/controlnet-scribble-sdxl-1.0)")
    p.add_argument("--control_image_path", type=str, default=None,
                   help="Path to control image (e.g. scribble PNG)")
    p.add_argument("--controlnet_scale", type=float, default=0.8,
                   help="ControlNet conditioning scale")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or os.path.join(EXP_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_shape = (4, args.height // 8, args.width // 8)
    d_latent = int(np.prod(latent_shape))

    # Validate ControlNet args
    use_controlnet = args.controlnet_model_id is not None
    if use_controlnet and args.control_image_path is None:
        raise ValueError("--control_image_path required when using --controlnet_model_id")

    print(f"=== compute_sdxl_ssa_v2.py ===")
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
    print(f"Score batch size: {args.score_batch_size} (fp16 UNet)")

    # --- Wandb ---
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project="sdxl-ssa",
            name=args.name,
            config=vars(args),
            tags=["sdxl", "v2"],
            dir=os.environ.get("WANDB_DIR", "."),
        )
        print(f"  wandb run: {wandb_run.url}")
    except Exception as e:
        print(f"  wandb init failed ({e}), continuing without logging")

    # --- Load ControlNet (if requested) ---
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
        )
        controlnet = controlnet.to(device)
        controlnet.eval()
        cn_params = sum(p.numel() for p in controlnet.parameters())
        print(f"  ControlNet parameters: {cn_params:,} ({cn_params/1e6:.1f}M)")

        # Load and preprocess control image
        from PIL import Image
        control_pil = Image.open(args.control_image_path).convert("RGB")
        control_pil = control_pil.resize((args.width, args.height))
        print(f"  Control image: {args.control_image_path} -> {control_pil.size}")

        # Preprocess to tensor for SDXLScore (normalized to [0,1], shape (1,3,H,W))
        import torchvision.transforms as T
        control_image_tensor = T.ToTensor()(control_pil).unsqueeze(0).to(device)

    # --- Load SDXL pipeline ---
    print("\n=== Loading SDXL pipeline ===")
    if use_controlnet:
        from diffusers import StableDiffusionXLControlNetPipeline
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            args.model_id,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    else:
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    pipe = pipe.to(device)

    unet = pipe.unet
    scheduler = pipe.scheduler
    unet.eval()

    n_params = sum(p.numel() for p in unet.parameters())
    unet_dtype = next(unet.parameters()).dtype
    print(f"  UNet parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  UNet dtype: {unet_dtype}")

    # Optional torch.compile
    if args.compile_unet:
        print("  Compiling UNet with torch.compile...")
        t_compile = time.time()
        unet = torch.compile(unet, mode="reduce-overhead")
        print(f"  Compiled in {time.time() - t_compile:.1f}s")

    # --- Encode prompt ---
    print("\n=== Encoding prompt ===")
    (prompt_embeds, negative_prompt_embeds,
     pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=args.prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
    )
    print(f"  prompt_embeds: {prompt_embeds.shape}, dtype={prompt_embeds.dtype}")
    print(f"  pooled_prompt_embeds: {pooled_prompt_embeds.shape}")

    # Build time_ids for SDXL (original_size, crop, target_size)
    time_ids = torch.tensor(
        [[args.height, args.width, 0, 0, args.height, args.width]],
        dtype=torch.float16, device=device,
    )

    # --- Generate latents ---
    print(f"\n=== Generating {args.n_samples} latents ===")
    t0 = time.time()
    gen_kwargs = {}
    if use_controlnet:
        gen_kwargs["control_image"] = control_pil
        gen_kwargs["controlnet_conditioning_scale"] = args.controlnet_scale
    latents = generate_latents(
        pipe, args.prompt, args.n_samples,
        guidance_scale=args.guidance_scale,
        gen_steps=args.gen_steps,
        height=args.height, width=args.width,
        batch_size=args.gen_batch_size,
        seed=args.seed,
        **gen_kwargs,
    )
    gen_time = time.time() - t0
    print(f"  Generated {latents.shape[0]} latents in {gen_time:.1f}s")
    print(f"  Latent range: [{latents.min():.3f}, {latents.max():.3f}]")
    print(f"  Latent shape: {latents.shape}")

    # Flatten for Langevin (fp32 for numerical precision)
    latents_flat = latents.view(latents.shape[0], -1).float()  # (N, d)

    # Compute var_f
    var_f = latents_flat.var(dim=0).mean().item()
    print(f"  var_f = {var_f:.6f}")

    # --- Load Latent-CLIP ---
    print("\n=== Loading Latent-CLIP ===")
    clip_embedder = LatentCLIPEmbedder(device, checkpoint_path=args.latent_clip_checkpoint)

    # --- kNN Entropy ---
    entropy = None
    if args.compute_entropy:
        print("\n=== Computing kNN entropy in CLIP space ===")
        clip_samples = clip_embedder.embed(latents_flat, latent_shape)
        entropy = compute_knn_entropy(clip_samples, k=args.entropy_k)
        print(f"  kNN entropy (k={args.entropy_k}): {entropy:.4f}")
        del clip_samples

    # --- Build score function ---
    print(f"\n=== Building SDXLScore (t={args.t_score}, fp16 UNet) ===")
    # v2: Keep UNet in fp16 — SDXLScore handles dtype casting at boundaries
    cn_score_kwargs = {}
    if use_controlnet:
        cn_score_kwargs["controlnet"] = controlnet
        cn_score_kwargs["control_image"] = control_image_tensor
        cn_score_kwargs["controlnet_conditioning_scale"] = args.controlnet_scale
    score_fn = SDXLScore(
        unet=unet,
        scheduler=scheduler,
        t=args.t_score,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        time_ids=time_ids,
        device=device,
        latent_shape=latent_shape,
        batch_size=args.score_batch_size,
        guidance_scale=args.guidance_scale,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        **cn_score_kwargs,
    )

    # --- Sanity check: one Langevin step ---
    print("\n=== Sanity check ===")
    x_test = latents_flat[:min(10, args.n_traj)].to(device)
    grad_test = score_fn(x_test)
    sigma_f = np.sqrt(var_f)
    displacement = (0.5 * var_f * args.dt * grad_test +
                    sigma_f * np.sqrt(args.dt) * torch.randn_like(x_test))
    print(f"  Score norm: {grad_test.norm(dim=-1).mean():.2f}")
    print(f"  Displacement norm: {displacement.norm(dim=-1).mean():.4f}")
    del x_test, grad_test, displacement
    torch.cuda.empty_cache()

    # --- Langevin dynamics ---
    # v2: Linear snapshot lags (more resolution at longer lags)
    snapshot_lags = sorted(set(
        int(round(x)) for x in np.linspace(10, args.T, 20)
    ))
    # Ensure we include the endpoints
    if args.T not in snapshot_lags:
        snapshot_lags.append(args.T)
    snapshot_lags = sorted(snapshot_lags)
    print(f"\n=== Langevin dynamics ({args.n_traj} traj x {args.T} steps) ===")
    print(f"  Snapshot lags: {snapshot_lags}")

    t0 = time.time()
    snapshots, x0_flat = run_langevin_latent_space(
        score_fn=score_fn,
        samples_flat=latents_flat,
        var_f=var_f,
        n_traj=args.n_traj,
        T=args.T,
        dt=args.dt,
        snapshot_lags=snapshot_lags,
        device=device,
        seed=args.seed,
        wandb_run=wandb_run,
    )
    langevin_time = time.time() - t0
    print(f"  Langevin completed in {langevin_time:.1f}s ({langevin_time/3600:.1f}h)")

    # Free UNet + ControlNet from GPU (CLIP needs the memory)
    del score_fn
    if controlnet is not None:
        controlnet = controlnet.to("cpu")
    pipe = pipe.to("cpu")
    torch.cuda.empty_cache()

    # --- SSA + DA computation ---
    print("\n=== Computing SSA + DA in CLIP space ===")
    t0 = time.time()
    results = compute_ssa_and_da(
        clip_embedder=clip_embedder,
        x0_flat=x0_flat,
        snapshots=snapshots,
        snapshot_lags=snapshot_lags,
        latent_shape=latent_shape,
    )
    ssa_time = time.time() - t0
    print(f"  SSA+DA computed in {ssa_time:.1f}s")

    # --- Save results ---
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
        "n_samples": args.n_samples,
        "n_traj": args.n_traj,
        "T": args.T,
        "dt": args.dt,
        "t_score": args.t_score,
        "var_f": var_f,
        "ssa": results["ssa"],
        "entropy": entropy,
        "da1_optimal_lag": results["da1_optimal_lag"],
        "da1_gap": results["da1_gap"],
        "da2_optimal_lag": results["da2_optimal_lag"],
        "da2_gap": results["da2_gap"],
        "gen_time_s": gen_time,
        "langevin_time_s": langevin_time,
        "ssa_time_s": ssa_time,
        "version": "v2",
    }

    json_path = os.path.join(output_dir, f"sdxl_ssa_{args.name}.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Save curves + DA directions
    npz_data = {
        "rho_sq": results["rho_sq"],
        "snapshot_lags": np.array(results["snapshot_lags"]),
        "gap12": results["gap12"],
        "gap23": results["gap23"],
        "mu_clip": results["mu_clip"],
        "da1_eigvec": results["da1_eigvec"],
        "da2_eigvec": results["da2_eigvec"],
    }
    # Save eigenvalues at each lag
    for i, lag in enumerate(results["snapshot_lags"]):
        npz_data[f"eigenvalues_lag{lag}"] = results["eigenvalues"][i]
        npz_data[f"eigvecs_lag{lag}"] = results["top_eigvecs"][i]

    npz_path = os.path.join(output_dir, f"sdxl_da_{args.name}.npz")
    np.savez_compressed(npz_path, **npz_data)
    print(f"Saved NPZ: {npz_path}")

    # Save latent snapshots for visualization (x0 + DA extremes)
    vis_data = {"x0": x0_flat.numpy()}
    for key_lag in [results["da1_optimal_lag"], results["da2_optimal_lag"]]:
        if key_lag in snapshots:
            vis_data[f"x_lag{key_lag}"] = snapshots[key_lag].numpy()
    vis_path = os.path.join(output_dir, f"sdxl_latents_{args.name}.npz")
    np.savez_compressed(vis_path, **vis_data)
    print(f"Saved latents: {vis_path}")

    # --- Wandb final logging ---
    if wandb_run is not None:
        wandb_run.log({
            "ssa": results["ssa"],
            "entropy": entropy,
            "da1_gap": results["da1_gap"],
            "da1_optimal_lag": results["da1_optimal_lag"],
            "da2_gap": results["da2_gap"],
            "da2_optimal_lag": results["da2_optimal_lag"],
            "gen_time_s": gen_time,
            "langevin_time_s": langevin_time,
            "ssa_time_s": ssa_time,
        })
        wandb_run.finish()

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Summary: {args.name} (v2)")
    print(f"  Model: {args.model_id}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  SSA = {results['ssa']:.6f}")
    if entropy is not None:
        print(f"  Entropy = {entropy:.4f}")
    print(f"  DA1: gap={results['da1_gap']:.2f}x at lag={results['da1_optimal_lag']}")
    print(f"  DA2: gap={results['da2_gap']:.2f}x at lag={results['da2_optimal_lag']}")
    print(f"  rho(tau)^2 curve: {results['rho_sq']}")
    print(f"  Total time: {gen_time + langevin_time + ssa_time:.1f}s "
          f"({(gen_time + langevin_time + ssa_time)/3600:.1f}h)")


if __name__ == "__main__":
    main()
