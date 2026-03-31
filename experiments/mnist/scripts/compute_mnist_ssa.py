"""
compute_mnist_ssa.py — Compute SSA for each MNIST digit using class-conditional
DDPM Tweedie score in full 784D pixel space.

Pipeline per digit + noise level:
1. Load trained class-conditional UNet + scheduler
2. Wrap UNet as per-digit score function: score(x) = -eps(x,t,label) / sigma_t
3. Load N=5000 real MNIST samples of that digit (flat 784D, normalized [-1,1])
4. Run Langevin in 784D pixel space
5. Compute trace-based SSA: rho(h) = tr(C(h))/tr(C(0)), SSA = mean(rho^2[1:L])

Usage:
    # Single digit (for testing):
    python scripts/compute_mnist_ssa.py --model_dir outputs/mnist/model --digit 1

    # All digits (SLURM array):
    python scripts/compute_mnist_ssa.py --model_dir outputs/mnist/model --digit $SLURM_ARRAY_TASK_ID

    # Noise level sweep (digits 1 and 3):
    python scripts/compute_mnist_ssa.py --model_dir outputs/mnist/model --digit 1 --noise_levels 1,5,10,20,50,100
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
# DDPM Score Wrapper (class-conditional)
# ---------------------------------------------------------------------------

class MNISTDDPMScore:
    """Wraps a class-conditional DDPM UNet as score function via Tweedie formula.

    score(x) = -eps_theta(x, t, label) / sigma_t

    The UNet expects (N, 1, 28, 28) images and integer timestep + class label.
    This wrapper handles flat (N, 784) <-> image reshape.
    """

    def __init__(self, unet, scheduler, t, digit_label, device):
        self.unet = unet
        self.device = device
        self.digit_label = digit_label

        alpha_bar_t = scheduler.alphas_cumprod[t].item()
        self.sigma_t = np.sqrt(1.0 - alpha_bar_t)
        self.t_tensor = torch.tensor([t], dtype=torch.long, device=device)
        self.label_tensor = torch.tensor([digit_label], dtype=torch.long, device=device)

        print(f"  MNISTDDPMScore: t={t}, digit={digit_label}, "
              f"alpha_bar={alpha_bar_t:.6f}, sigma_t={self.sigma_t:.6f}")

    @torch.no_grad()
    def __call__(self, x_flat):
        """Score function: (N, 784) -> (N, 784)."""
        N = x_flat.shape[0]
        x_img = x_flat.view(N, 1, 28, 28)

        t_batch = self.t_tensor.expand(N)
        labels_batch = self.label_tensor.expand(N)
        eps_pred = self.unet(x_img, t_batch, class_labels=labels_batch,
                             return_dict=False)[0]

        score_img = -eps_pred / self.sigma_t
        return score_img.view(N, -1)


# ---------------------------------------------------------------------------
# Trace-based SSA in 784D pixel space
# ---------------------------------------------------------------------------

def compute_trace_ssa(trajectories, mu, max_lag, burnin=0):
    """Compute SSA from pixel-space trajectories using trace normalization.

    rho(h) = tr(C(h)) / tr(C(0))
    SSA = mean(rho^2[burnin+1 : max_lag])

    Args:
        trajectories: (n_traj, T+1, 784) pixel-space trajectories
        mu: (784,) class mean for centering
        max_lag: Maximum lag
        burnin: Skip first burnin lags

    Returns:
        ssa: float
        rho_sq: (max_lag,) array of rho(h)^2 values
    """
    n_traj, T_total, d = trajectories.shape

    # Center by class mean (NOT per-trajectory mean)
    centered = trajectories - mu.unsqueeze(0).unsqueeze(0)

    # tr(C(0)) = E[||x - mu||^2]
    tr_c0 = (centered ** 2).sum(dim=-1).mean().item()
    if tr_c0 < 1e-12:
        print("  WARNING: tr(C(0)) near zero — degenerate trajectories")
        return 0.0, np.zeros(max_lag)

    rho_sq = np.zeros(max_lag)
    n_lags = min(max_lag, T_total - 1)

    for h in range(1, n_lags + 1):
        x_t = centered[:, :T_total - h, :]
        x_th = centered[:, h:, :]
        tr_ch = (x_t * x_th).sum(dim=-1).mean().item()
        rho = tr_ch / tr_c0
        rho_sq[h - 1] = rho ** 2

    lag_start = burnin
    if lag_start >= n_lags:
        lag_start = 0
    ssa = float(np.mean(rho_sq[lag_start:n_lags]))

    return ssa, rho_sq


# ---------------------------------------------------------------------------
# Langevin dynamics in 784D
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_langevin_784d(score_fn, samples_flat, var_f, n_traj, T, dt,
                      device, seed=42):
    """Run Langevin dynamics in 784D pixel space.

    No projection needed — 784D is directly tractable for autocorrelation.

    Args:
        score_fn: MNISTDDPMScore callable (N, 784) -> (N, 784)
        samples_flat: (M, 784) real samples for initial conditions
        var_f: scalar diffusion coefficient
        n_traj: number of trajectories
        T: total Langevin steps
        dt: timestep
        device: torch device
        seed: random seed

    Returns:
        trajectories: (n_traj, T+1, 784) on CPU
    """
    sigma_f = np.sqrt(var_f)
    sqrt_dt = np.sqrt(dt)

    torch.manual_seed(seed)
    idx = torch.randint(0, samples_flat.shape[0], (n_traj,))
    x = samples_flat[idx].to(device)  # (n_traj, 784)

    # Store every step for autocorrelation
    # Memory: n_traj * (T+1) * 784 * 4 bytes
    # 500 * 2001 * 784 * 4 = ~3.0 GB — fits in RAM
    traj = torch.zeros(n_traj, T + 1, 784, device="cpu")
    traj[:, 0, :] = x.cpu()

    for t in range(1, T + 1):
        grad = score_fn(x)  # (n_traj, 784)
        drift = 0.5 * var_f * dt * grad
        noise = sigma_f * sqrt_dt * torch.randn_like(x)
        x = x + drift + noise
        # Clamp to prevent divergence (DDPM training range is [-1, 1])
        x = x.clamp(-1.5, 1.5)
        traj[:, t, :] = x.cpu()

        if t % 500 == 0 or t == T:
            x_norm = x.norm(dim=-1).mean().item()
            print(f"    Step {t}/{T}, ||x||={x_norm:.2f}", end="\r")

    print()
    return traj


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

@torch.no_grad()
def sanity_check(score_fn, samples_flat, var_f, dt, device, n_check=100):
    """Quick sanity check: 1 Langevin step, verify small displacement."""
    x = samples_flat[:n_check].to(device)
    grad = score_fn(x)
    sigma_f = np.sqrt(var_f)
    sqrt_dt = np.sqrt(dt)

    drift = 0.5 * var_f * dt * grad
    noise = sigma_f * sqrt_dt * torch.randn_like(x)
    x_new = x + drift + noise

    displacement = (x_new - x).norm(dim=-1)
    score_norm = grad.norm(dim=-1)

    print(f"  Sanity check ({n_check} samples, 1 step):")
    print(f"    Score norm:    mean={score_norm.mean():.2f}, "
          f"std={score_norm.std():.2f}")
    print(f"    Displacement:  mean={displacement.mean():.4f}, "
          f"std={displacement.std():.4f}")
    print(f"    x range after: [{x_new.min():.3f}, {x_new.max():.3f}]")

    if displacement.mean() > 1.0:
        print("  WARNING: Large displacement — consider reducing dt")
    if score_norm.mean() < 0.01:
        print("  WARNING: Score norms very small — model may not be trained")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-digit SSA using class-conditional DDPM Tweedie score"
    )
    p.add_argument("--model_dir", type=str, required=True,
                   help="Path to saved UNet model directory (with config.json + diffusion_pytorch_model.safetensors)")
    p.add_argument("--digit", type=int, required=True,
                   help="Digit class (0-9)")
    p.add_argument("--noise_levels", type=str, default="10",
                   help="Comma-separated noise levels t for score function")
    p.add_argument("--n_samples", type=int, default=5000,
                   help="Number of real MNIST samples per digit")
    p.add_argument("--n_traj", type=int, default=500,
                   help="Number of Langevin trajectories")
    p.add_argument("--T", type=int, default=2000,
                   help="Langevin trajectory length (steps)")
    p.add_argument("--dt", type=float, default=0.01,
                   help="Langevin timestep")
    p.add_argument("--max_lag", type=int, default=500,
                   help="Max autocorrelation lag")
    p.add_argument("--burnin", type=int, default=0,
                   help="Skip first burnin lags in SSA summation")
    p.add_argument("--data_dir", type=str, default="data",
                   help="MNIST data directory")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Results directory (default: EXP_DIR/results)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or os.path.join(EXP_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    noise_levels = [int(t) for t in args.noise_levels.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    digit = args.digit
    assert 0 <= digit <= 9, f"Invalid digit: {digit}"

    print(f"Model: {args.model_dir}")
    print(f"Digit: {digit}")
    print(f"Device: {device}")
    print(f"Noise levels: {noise_levels}")
    print(f"Config: n_samples={args.n_samples}, n_traj={args.n_traj}, "
          f"T={args.T}, dt={args.dt}, max_lag={args.max_lag}, burnin={args.burnin}")

    # --- Load model ---
    print("\n=== Loading UNet ===")
    from diffusers import UNet2DModel, DDPMScheduler
    unet = UNet2DModel.from_pretrained(args.model_dir).to(device)
    unet.eval()
    scheduler = DDPMScheduler.from_pretrained(args.model_dir)

    n_params = sum(p.numel() for p in unet.parameters())
    print(f"  UNet parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Class embeds: {unet.config.num_class_embeds}")

    # --- Load real MNIST samples for this digit ---
    print(f"\n=== Loading MNIST digit {digit} ===")
    from torchvision.datasets import MNIST
    from torchvision import transforms
    dataset = MNIST(root=args.data_dir, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),  # [-1, 1]
                    ]))

    digit_images = []
    for img, label in dataset:
        if label == digit:
            digit_images.append(img.view(-1))  # (784,)
        if len(digit_images) >= args.n_samples:
            break

    samples_flat = torch.stack(digit_images)  # (N, 784)
    print(f"  Loaded {len(samples_flat)} samples for digit {digit}")
    print(f"  Range: [{samples_flat.min():.3f}, {samples_flat.max():.3f}]")

    # Compute class mean and var_f
    mu = samples_flat.mean(dim=0)  # (784,)
    var_f = samples_flat.var(dim=0).mean().item()
    print(f"  var_f = {var_f:.6f} (sigma_f = {np.sqrt(var_f):.6f})")

    # --- Sweep noise levels ---
    results = {
        "model_dir": args.model_dir,
        "digit": digit,
        "n_params": n_params,
        "n_samples": len(samples_flat),
        "n_traj": args.n_traj,
        "T": args.T,
        "dt": args.dt,
        "max_lag": args.max_lag,
        "burnin": args.burnin,
        "var_f": var_f,
        "noise_levels": {},
    }

    all_rho_sq = {}

    for t_noise in noise_levels:
        print(f"\n{'='*60}")
        print(f"=== Digit {digit}, noise level t={t_noise} ===")
        t0 = time.time()

        score_fn = MNISTDDPMScore(unet, scheduler, t_noise, digit, device)

        # Sanity check (first noise level only)
        if t_noise == noise_levels[0]:
            sanity_check(score_fn, samples_flat, var_f, args.dt, device)

        # Run Langevin
        print(f"  Running Langevin ({args.n_traj} traj x {args.T} steps)...")
        traj = run_langevin_784d(
            score_fn=score_fn,
            samples_flat=samples_flat,
            var_f=var_f,
            n_traj=args.n_traj,
            T=args.T,
            dt=args.dt,
            device=device,
            seed=args.seed + t_noise + digit * 1000,
        )
        print(f"  Trajectory shape: {traj.shape}")

        # Compute SSA
        print(f"  Computing trace SSA (max_lag={args.max_lag}, burnin={args.burnin})...")
        ssa, rho_sq = compute_trace_ssa(
            traj, mu, max_lag=args.max_lag, burnin=args.burnin,
        )

        elapsed = time.time() - t0
        print(f"  SSA(digit={digit}, t={t_noise}) = {ssa:.6f}  [{elapsed:.1f}s]")

        results["noise_levels"][str(t_noise)] = {
            "ssa": ssa,
            "elapsed_s": elapsed,
        }
        all_rho_sq[str(t_noise)] = rho_sq

        del traj
        torch.cuda.empty_cache()

    # --- Save results ---
    json_path = os.path.join(output_dir, f"ssa_digit_{digit}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {json_path}")

    npz_path = os.path.join(output_dir, f"ssa_curves_digit_{digit}.npz")
    np.savez_compressed(npz_path, **{f"rho_sq_t{k}": v for k, v in all_rho_sq.items()})
    print(f"Saved curves: {npz_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary for digit {digit}")
    print(f"{'t':>5}  {'SSA':>10}  {'time':>6}")
    print("-" * 25)
    for t_noise in noise_levels:
        r = results["noise_levels"][str(t_noise)]
        print(f"{t_noise:>5}  {r['ssa']:>10.6f}  {r['elapsed_s']:>5.1f}s")


if __name__ == "__main__":
    main()
