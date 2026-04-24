"""
Marchenko-Pastur generalization for the time-lagged sample autocovariance
of the canonical OU diffusion under the isotropic Gaussian null.

Verifies the exact decomposition
    S(tau) = c_+(tau) * W_+ - c_-(tau) * W_-
with W_+ _|_ W_- independent Wishart matrices, c_pm(tau) = sigma^2 (1 pm e^{-tau/2})/2.

Three experiments:
  (1) Two samplings of S(tau) give indistinguishable spectra.
  (2) Empirical bulk matches the free-convolution prediction and departs from
      the "naive MP with sigma^2 <- rho(tau)^2" overlay used in the parent paper.
  (3) Top eigenvalue trajectory vs. the paper's Frobenius/Bai-Yin noise-floor
      bounds (using analytic_floor_lower/upper from noise_floor_utils).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Pull the paper's noise-floor formulas from the existing utility module
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "noise_floor"))
from noise_floor_utils import analytic_floor_lower, analytic_floor_upper  # noqa: E402

matplotlib.rcParams["figure.dpi"] = 120

OUT_REPO = ROOT / "figures" / "mp_generalization"
OUT_PAPER = Path(
    "/mnt/g/My Drive/Students/ShaulTokolovsky/MultimodalityMeasure/SSA/mp_figures"
)
OUT_REPO.mkdir(parents=True, exist_ok=True)
OUT_PAPER.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def sample_ou_pair(d: int, N: int, tau: float, sigma: float, rng: np.random.Generator):
    """Exact, discretization-free sample of (X_0, X_tau) pairs under OU null."""
    a = float(np.exp(-tau / 2.0))
    b = float(sigma * np.sqrt(1.0 - np.exp(-tau)))
    X0 = sigma * rng.standard_normal((d, N))
    xi = rng.standard_normal((d, N))
    Xt = a * X0 + b * xi
    return X0, Xt


def S_from_pair(X0: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    """Symmetrized empirical time-lagged autocovariance."""
    N = X0.shape[1]
    C = Xt @ X0.T / N
    return 0.5 * (C + C.T)


def sample_S_via_decomp(d: int, N: int, tau: float, sigma: float,
                        rng: np.random.Generator) -> np.ndarray:
    """S = c_+ W_+ - c_- W_- via INDEPENDENT Wishart matrices (Theorem 1)."""
    rho = float(np.exp(-tau / 2.0))
    c_plus = 0.5 * sigma**2 * (1.0 + rho)
    c_minus = 0.5 * sigma**2 * (1.0 - rho)
    U = rng.standard_normal((d, N))
    V = rng.standard_normal((d, N))
    Wp = U @ U.T / N
    Wm = V @ V.T / N
    return c_plus * Wp - c_minus * Wm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a)
    b = np.sort(b)
    grid = np.sort(np.concatenate([a, b]))
    Fa = np.searchsorted(a, grid, side="right") / len(a)
    Fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(Fa - Fb)))


def mp_density(eff_sigma2: float, gamma: float):
    lam_minus = eff_sigma2 * (1 - np.sqrt(gamma)) ** 2
    lam_plus = eff_sigma2 * (1 + np.sqrt(gamma)) ** 2

    def f(x):
        out = np.zeros_like(x)
        mask = (x > lam_minus) & (x < lam_plus)
        xm = x[mask]
        out[mask] = np.sqrt((lam_plus - xm) * (xm - lam_minus)) / (
            2 * np.pi * gamma * eff_sigma2 * xm
        )
        return out

    return f, lam_minus, lam_plus


def save_to_both(fig, name: str):
    for out in (OUT_REPO, OUT_PAPER):
        fig.savefig(out / name, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Experiment 1: two samplings give the same spectrum
# ---------------------------------------------------------------------------

def experiment_decomp_verification(d=100, N=2000, sigma=1.0, n_trials=50,
                                   rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng(0)
    tau_list = [0.0, 0.5, 1.0, 2.0, 5.0, 20.0]
    ncols = len(tau_list)
    fig, ax = plt.subplots(1, ncols, figsize=(2.8 * ncols, 3.0), sharey=False)

    for j, tau in enumerate(tau_list):
        eig_ou, eig_dec = [], []
        for _ in range(n_trials):
            X0, Xt = sample_ou_pair(d, N, tau, sigma, rng)
            eig_ou.append(np.linalg.eigvalsh(S_from_pair(X0, Xt)))
            eig_dec.append(np.linalg.eigvalsh(sample_S_via_decomp(d, N, tau, sigma, rng)))
        eig_ou = np.concatenate(eig_ou)
        eig_dec = np.concatenate(eig_dec)

        lo = min(eig_ou.min(), eig_dec.min())
        hi = max(eig_ou.max(), eig_dec.max())
        bins = np.linspace(lo, hi, 60)
        ax[j].hist(eig_ou, bins=bins, density=True, alpha=0.55, label="OU sampling")
        ax[j].hist(eig_dec, bins=bins, density=True, alpha=0.55, label="Wishart decomp")
        ax[j].set_title(rf"$\tau={tau}$")
        ax[j].set_xlabel("eigenvalue")
        ax[j].text(
            0.02, 0.97,
            f"KS = {ks_distance(eig_ou, eig_dec):.3f}",
            transform=ax[j].transAxes, fontsize=7, va="top", ha="left",
        )
        if j == 0:
            ax[j].set_ylabel("empirical density")
            ax[j].legend(fontsize=8, loc="upper right")

    fig.suptitle(
        rf"Verification of $S(\tau)=c_+ W_+ - c_- W_-$  "
        rf"($d={d}$, $N={N}$, $\gamma={d/N:.3f}$, {n_trials} trials)",
        y=1.03,
    )
    fig.tight_layout()
    save_to_both(fig, "decomp_verification.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2: bulk vs. naive MP, free-convolution ground truth
# ---------------------------------------------------------------------------

def experiment_bulk_vs_naive(d=100, N=2000, sigma=1.0,
                             d_gt=400, N_gt=8000,
                             gt_trials=30, n_trials=50,
                             rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng(1)
    tau_list = [0.0, 0.5, 1.0, 2.0, 5.0, 20.0]
    gamma = d / N

    fig, ax = plt.subplots(1, len(tau_list), figsize=(2.8 * len(tau_list), 3.0))
    for j, tau in enumerate(tau_list):
        eig_ou = np.concatenate([
            np.linalg.eigvalsh(S_from_pair(*sample_ou_pair(d, N, tau, sigma, rng)))
            for _ in range(n_trials)
        ])
        eig_gt = np.concatenate([
            np.linalg.eigvalsh(sample_S_via_decomp(d_gt, N_gt, tau, sigma, rng))
            for _ in range(gt_trials)
        ])

        lo = min(eig_ou.min(), eig_gt.min())
        hi = max(eig_ou.max(), eig_gt.max())
        bins = np.linspace(lo, hi, 60)
        ax[j].hist(eig_ou, bins=bins, density=True, alpha=0.55, color="tab:blue",
                   label=f"OU (d={d},N={N})")
        gt_hist, _ = np.histogram(eig_gt, bins=bins, density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        ax[j].plot(centers, gt_hist, color="tab:red", lw=2, label="free conv (MC)")

        rho = float(np.exp(-tau / 2.0))
        eff = (sigma * rho) ** 2
        if eff > 1e-12:
            fmp, lm, lp = mp_density(eff, gamma)
            xs = np.linspace(lm, lp, 400)
            ax[j].plot(xs, fmp(xs), "--", color="tab:green", lw=2,
                       label=r"naive MP, $\sigma^2=\rho^2$")
        ax[j].set_title(rf"$\tau={tau}$")
        ax[j].set_xlabel("eigenvalue")
        ax[j].text(
            0.02, 0.97,
            f"KS(OU, GT) = {ks_distance(eig_ou, eig_gt):.3f}",
            transform=ax[j].transAxes, fontsize=7, va="top", ha="left",
        )
        if j == 0:
            ax[j].set_ylabel("density")
            ax[j].legend(fontsize=7, loc="best")

    fig.suptitle(r"Bulk of $S(\tau)$: OU vs. free-conv ground truth vs. naive MP",
                 y=1.03)
    fig.tight_layout()
    save_to_both(fig, "bulk_vs_naive_MP.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 3: top-eigenvalue trajectory vs. paper's bounds
# ---------------------------------------------------------------------------

def experiment_edge_trajectory(sigma=1.0, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng(2)
    configs = [(100, 2000), (50, 500)]
    tau_grid = np.geomspace(1e-2, 30.0, 40)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5))

    for d, N in configs:
        gamma = d / N
        lam_max_emp = np.array([
            np.mean([
                np.linalg.eigvalsh(S_from_pair(*sample_ou_pair(d, N, tau, sigma, rng)))[-1]
                for _ in range(20)
            ])
            for tau in tau_grid
        ])

        d_gt, N_gt = 400, 8000
        lam_max_gt = np.array([
            np.linalg.eigvalsh(sample_S_via_decomp(d_gt, N_gt, tau, sigma, rng))[-1]
            for tau in tau_grid
        ])

        floor_upper = analytic_floor_upper(tau_grid, sigma**2, d, N)
        floor_lower = analytic_floor_lower(tau_grid, sigma**2, d, N)

        (line_emp,) = ax.plot(tau_grid, lam_max_emp, "o-", ms=4,
                              label=rf"empirical, $d={d}, N={N}$ ($\gamma={gamma:.2f}$)")
        ax.plot(tau_grid, lam_max_gt, "-", color=line_emp.get_color(), alpha=0.5,
                label=rf"free-conv MC ($d={d_gt}, N={N_gt}$)")
        ax.plot(tau_grid, floor_upper, ":", color=line_emp.get_color(),
                label=rf"paper upper, $\gamma={gamma:.2f}$")
        ax.plot(tau_grid, floor_lower, "--", color=line_emp.get_color(),
                label=rf"paper lower, $\gamma={gamma:.2f}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"lag $\tau$")
    ax.set_ylabel(r"$\lambda_{\max}(S(\tau))$")
    ax.set_title(r"Top eigenvalue of $S(\tau)$ vs. paper's noise-floor bounds")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_to_both(fig, "edge_trajectory.png")
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(42)
    print("[1/3] Decomposition verification ...")
    experiment_decomp_verification(rng=rng)
    print("[2/3] Bulk vs. naive MP ...")
    experiment_bulk_vs_naive(rng=rng)
    print("[3/3] Edge trajectory vs. paper's bounds ...")
    experiment_edge_trajectory(rng=rng)
    print(f"Done. Figures in\n  {OUT_REPO}\n  {OUT_PAPER}")


if __name__ == "__main__":
    main()
