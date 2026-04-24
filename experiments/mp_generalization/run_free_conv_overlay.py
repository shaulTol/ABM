"""
Demo: OU-sampled spectrum vs. analytic free-convolution density vs. naive MP.

Produces the figure (parent paper, Appendix D.3 caption):
    "Bulk of S(tau): simulated eigenvalues vs. analytic prediction".

For each tau in a grid:
  * Blue histogram : eigenvalues of S(tau) computed from direct OU sampling
                    of (X_0, X_tau) pairs, at modest (d, N).
  * Red solid line : analytic density rho_tau^gamma from the Stieltjes-cubic
                    theorem (free_conv_utils.density).
  * Green dashed   : the naive "MP with sigma^2 <- rho(tau)^2" overlay used
                    in the original parent-paper draft. Present to show it
                    disagrees at every tau > 0.

Unlike run_mp_generalization.experiment_bulk_vs_naive, which needs a
Monte-Carlo "ground truth" at larger (d_gt, N_gt) to obtain the target
bulk, this script uses the analytic density directly.

Usage:
    python3 run_free_conv_overlay.py
    python3 run_free_conv_overlay.py --d 100 --N 2000 --sigma 1.0
    python3 run_free_conv_overlay.py --taus 0 0.5 1 2 5 20 --save out.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# local imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "noise_floor"))
sys.path.insert(0, str(ROOT / "experiments" / "mp_generalization"))

from free_conv_utils import density as fc_density, edges as fc_edges  # noqa: E402
from run_mp_generalization import (  # noqa: E402
    sample_ou_pair, S_from_pair, mp_density,
)

matplotlib.rcParams["figure.dpi"] = 120

OUT_REPO = ROOT / "figures" / "mp_generalization"
OUT_PAPER = Path(
    "/mnt/g/My Drive/Students/ShaulTokolovsky/MultimodalityMeasure/SSA/mp_figures"
)


def run(taus, d, N, sigma, n_trials, seed, save_path):
    rng = np.random.default_rng(seed)
    gamma = d / N

    fig, axes = plt.subplots(
        1, len(taus), figsize=(2.9 * len(taus), 3.3),
        sharey=False,
    )
    if len(taus) == 1:
        axes = [axes]

    for j, tau in enumerate(taus):
        # --- simulate OU eigenvalues ---
        eigs = np.concatenate([
            np.linalg.eigvalsh(S_from_pair(*sample_ou_pair(d, N, tau, sigma, rng)))
            for _ in range(n_trials)
        ])

        # --- analytic free-conv edges + density grid ---
        if tau == 0.0:
            lm_th, lp_th = sigma**2 * (1 - np.sqrt(gamma))**2, sigma**2 * (1 + np.sqrt(gamma))**2
        else:
            lm_th, lp_th = fc_edges(tau, sigma, gamma=gamma)

        pad = 0.05 * (lp_th - lm_th)
        lo = min(eigs.min(), lm_th) - pad
        hi = max(eigs.max(), lp_th) + pad

        lam_grid = np.linspace(lm_th, lp_th, 500)
        rho_grid = fc_density(lam_grid, tau, sigma, gamma=gamma)

        # --- naive MP overlay (wrong, for contrast) ---
        rho = float(np.exp(-tau / 2.0))
        eff = (sigma * rho) ** 2

        # --- plot ---
        bins = np.linspace(lo, hi, 70)
        axes[j].hist(eigs, bins=bins, density=True, alpha=0.55,
                     color="tab:blue",
                     label=f"OU sim. (d={d},N={N})")
        axes[j].plot(lam_grid, rho_grid, "r-", lw=2.2,
                     label="analytic (Thm 1)")
        if eff > 1e-12:
            fmp, lm_mp, lp_mp = mp_density(eff, gamma)
            xs = np.linspace(lm_mp, lp_mp, 400)
            axes[j].plot(xs, fmp(xs), "--", color="tab:green", lw=1.8,
                         label=r"naive MP, $\sigma^2\leftarrow\rho^2$")

        # edge markers
        axes[j].axvline(lm_th, color="k", ls=":", lw=0.8)
        axes[j].axvline(lp_th, color="k", ls=":", lw=0.8)

        axes[j].set_title(rf"$\tau={tau}$")
        axes[j].set_xlabel(r"$\lambda$")
        if j == 0:
            axes[j].set_ylabel(r"density $\rho(\lambda)$")
            axes[j].legend(fontsize=7, loc="best")

    fig.suptitle(
        rf"Bulk of $S(\tau)$: simulation vs. analytic free-conv vs. naive MP  "
        rf"($\gamma = d/N = {gamma:.3f}$, $\sigma={sigma}$)",
        y=1.04,
    )
    fig.tight_layout()

    if save_path is None:
        save_path = OUT_REPO / "free_conv_overlay.png"
        OUT_REPO.mkdir(parents=True, exist_ok=True)
    save_path = Path(save_path)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"saved: {save_path}")
    # Also copy to paper figures dir if writable
    try:
        OUT_PAPER.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PAPER / save_path.name, bbox_inches="tight")
        print(f"also saved: {OUT_PAPER / save_path.name}")
    except Exception as e:  # noqa: BLE001
        print(f"note: could not copy to {OUT_PAPER}: {e}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taus", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 2.0, 5.0, 20.0],
                        help="tau values to plot (space-separated)")
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--n-trials", type=int, default=50,
                        help="OU sampling trials per panel")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    run(
        taus=args.taus, d=args.d, N=args.N, sigma=args.sigma,
        n_trials=args.n_trials, seed=args.seed, save_path=args.save,
    )
