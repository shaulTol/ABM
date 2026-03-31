"""
compute_mnist_da.py — Compute DA directions for each MNIST digit using
class-conditional DDPM Tweedie score in full 784D pixel space.

For each digit:
1. Load UNet, create per-digit score function
2. Build GenericLangevinSimulator with the DDPM score
3. Run da_convergence_profile() with geometric checkpoints (lag 10..10000)
4. Find optimal lag (peak eigenvalue gap)
5. Extract DA1 (784D vector, directly reshapeable to 28x28)
6. Compute PCA1 from within-class C(0)
7. Save DA/PCA directions + convergence profile

Usage:
    python scripts/compute_mnist_da.py --model_dir outputs/mnist/model --digit 1
    python scripts/compute_mnist_da.py --model_dir outputs/mnist/model --digit $SLURM_ARRAY_TASK_ID
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
sys.path.insert(0, REPO_ROOT)

from src.metrics.da_directions import da_convergence_profile
from src.simulation.generic_langevin import GenericLangevinSimulator


# ---------------------------------------------------------------------------
# DDPM Score Wrapper (same as in compute_mnist_ssa.py)
# ---------------------------------------------------------------------------

class MNISTDDPMScore:
    """Wraps a class-conditional DDPM UNet as score function via Tweedie formula."""

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
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute DA directions per digit via DDPM Tweedie score"
    )
    p.add_argument("--model_dir", type=str, required=True,
                   help="Path to saved UNet model directory")
    p.add_argument("--digit", type=int, required=True, help="Digit (0-9)")
    p.add_argument("--noise_level", type=int, default=10,
                   help="Noise level t for score function")
    p.add_argument("--n_samples", type=int, default=5000,
                   help="Number of real MNIST samples per digit")
    p.add_argument("--n_traj", type=int, default=5000,
                   help="Number of trajectories for DA estimation")
    p.add_argument("--n_checkpoints", type=int, default=25,
                   help="Number of geometric checkpoints")
    p.add_argument("--max_lag", type=int, default=10000,
                   help="Maximum lag for convergence profile")
    p.add_argument("--dt", type=float, default=0.01,
                   help="Langevin timestep")
    p.add_argument("--data_dir", type=str, default="data",
                   help="MNIST data directory")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Results directory (default: EXP_DIR/results_da)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or os.path.join(EXP_DIR, "results_da")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    digit = args.digit
    assert 0 <= digit <= 9

    print(f"Model: {args.model_dir}")
    print(f"Digit: {digit}")
    print(f"Device: {device}")
    print(f"Config: noise_level={args.noise_level}, n_traj={args.n_traj}, "
          f"dt={args.dt}, max_lag={args.max_lag}, n_checkpoints={args.n_checkpoints}")

    # --- Load model ---
    print("\n=== Loading UNet ===")
    from diffusers import UNet2DModel, DDPMScheduler
    unet = UNet2DModel.from_pretrained(args.model_dir).to(device)
    unet.eval()
    scheduler = DDPMScheduler.from_pretrained(args.model_dir)
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"  UNet parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # --- Load real MNIST samples ---
    print(f"\n=== Loading MNIST digit {digit} ===")
    from torchvision.datasets import MNIST
    from torchvision import transforms
    dataset = MNIST(root=args.data_dir, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]))

    digit_images = []
    for img, label in dataset:
        if label == digit:
            digit_images.append(img.view(-1))
        if len(digit_images) >= args.n_samples:
            break

    samples_flat = torch.stack(digit_images)  # (N, 784)
    print(f"  Loaded {len(samples_flat)} samples")

    # --- Create score function ---
    score_fn = MNISTDDPMScore(unet, scheduler, args.noise_level, digit, device)

    # --- Compute class statistics ---
    mu = samples_flat.mean(dim=0)
    var_f = samples_flat.var(dim=0).mean().item()
    print(f"  var_f = {var_f:.6f}")

    # --- Build GenericLangevinSimulator ---
    print("\n=== Building Langevin simulator ===")
    sim = GenericLangevinSimulator(
        score_fn=score_fn,
        samples=samples_flat,
        dt=args.dt,
        device=device,
        mu_f=mu,
        var_f=var_f,
    )

    # --- Compute PCA1 (within-class) ---
    print("\n=== Computing within-class PCA ===")
    centered = samples_flat - mu.unsqueeze(0)
    # Use SVD for efficiency (784x784 covariance would be fine too)
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    pca_evals = (S ** 2) / (len(samples_flat) - 1)
    pca_evecs = Vt.T  # (784, N) — columns are eigenvectors
    pca1 = pca_evecs[:, 0]
    print(f"  PCA1 eigenvalue: {pca_evals[0]:.4f}")
    print(f"  PCA2 eigenvalue: {pca_evals[1]:.4f}")
    print(f"  PCA variance ratio: {pca_evals[0]/pca_evals.sum():.4f}")

    # --- DA convergence profile ---
    print(f"\n=== DA convergence profile (digit {digit}) ===")
    checkpoints = np.unique(
        np.geomspace(10, args.max_lag, args.n_checkpoints).astype(int)
    ).tolist()
    print(f"  Checkpoints: {len(checkpoints)} lags from {checkpoints[0]} to {checkpoints[-1]}")
    print(f"  Physical time: {checkpoints[0]*args.dt:.1f} to {checkpoints[-1]*args.dt:.1f}")

    t0 = time.time()
    profile = da_convergence_profile(
        simulator=sim,
        checkpoints=checkpoints,
        n_trajectories=args.n_traj,
        dt=args.dt,
        device=device,
        seed=args.seed + digit,
        reference_direction=pca1,
    )
    elapsed = time.time() - t0

    opt_idx = profile['optimal_idx']
    opt_lag = profile['checkpoints'][opt_idx]
    opt_gap = profile['gap_profile'][opt_idx]
    opt_angle = profile['angles_deg'][opt_idx]

    print(f"\n  Convergence profile (computed in {elapsed:.1f}s):")
    print(f"  {'Lag':>6}  {'Phys t':>7}  {'Gap':>8}  {'Angle(DA1,PCA1)':>16}  {'lam1':>8}")
    print(f"  {'-'*55}")
    for i, cp in enumerate(profile['checkpoints']):
        marker = " <-- optimal" if i == opt_idx else ""
        print(f"  {cp:>6}  {cp*args.dt:>7.1f}  {profile['gap_profile'][i]:>8.2f}  "
              f"{profile['angles_deg'][i]:>16.1f}  "
              f"{profile['leading_eigenvalues'][i]:>8.4f}{marker}")

    print(f"\n  Optimal lag: {opt_lag} (phys_time={opt_lag*args.dt:.1f})")
    print(f"  Optimal gap: {opt_gap:.2f}x")
    print(f"  DA1 angle from PCA1: {opt_angle:.1f} deg")

    da_evals = profile['optimal_eigenvalues']
    da_evecs = profile['optimal_eigenvectors']

    # Align sign: make DA1 point same direction as PCA1
    if np.dot(da_evecs[:, 0], pca1.numpy()) < 0:
        da_evecs[:, 0] *= -1

    print(f"  DA1 eigenvalue: {da_evals[0]:.4f}")
    print(f"  DA2 eigenvalue: {da_evals[1]:.4f}")

    # --- Project samples onto DA1 and PCA1 ---
    da1_proj = (centered @ torch.from_numpy(da_evecs[:, 0]).float()).numpy()
    pca1_proj = (centered @ pca1).numpy()

    # --- Save results ---
    results = {
        "digit": digit,
        "noise_level": args.noise_level,
        "n_traj": args.n_traj,
        "dt": args.dt,
        "max_lag": args.max_lag,
        "var_f": var_f,
        "optimal_lag": int(opt_lag),
        "optimal_gap": float(opt_gap),
        "da1_angle_from_pca1": float(opt_angle),
        "da1_eigenvalue": float(da_evals[0]),
        "da2_eigenvalue": float(da_evals[1]),
        "pca1_eigenvalue": float(pca_evals[0].item()),
        "pca2_eigenvalue": float(pca_evals[1].item()),
        "elapsed_s": elapsed,
    }

    json_path = os.path.join(output_dir, f"da_digit_{digit}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {json_path}")

    # Save directions + profile
    npz_path = os.path.join(output_dir, f"da_digit_{digit}.npz")
    np.savez_compressed(
        npz_path,
        da_eigenvalues=da_evals,
        da_eigenvectors=da_evecs,
        pca_eigenvalues=pca_evals.numpy()[:20],
        pca_eigenvectors=pca_evecs.numpy()[:, :20],
        da1_proj=da1_proj,
        pca1_proj=pca1_proj,
        checkpoints=profile['checkpoints'],
        gap_profile=profile['gap_profile'],
        angles_deg=profile['angles_deg'],
        leading_eigenvalues=profile['leading_eigenvalues'],
        class_mean=mu.numpy(),
    )
    print(f"Saved directions: {npz_path}")


if __name__ == "__main__":
    main()
