#!/usr/bin/env python3
"""
Plot planted-spike variants: 2 rows × 3 columns.

Top row:    #{λ_j > λ_+(τ)}  — mode count above noise floor
Bottom row: λ_m(τ)/λ_m(0)    — m-th eigenvalue ratio (m = planted count for that column)

Vertical line: largest τ where BOTH conditions hold for that m:
  (i)  top m eigenvalues of Ĉ_sym(τ) exceed λ_+(τ)
  (ii) λ_k(τ)/λ_k(0) < 1/e for all k = 1, ..., m

Usage:
    python plot_spike_count_per_direction.py --npz fig_planted_spikes_variants_XXX.npz
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.noise_floor.free_conv_utils import edges as fc_edges, edge_tau0  # noqa: E402


def compute_tau_star(taus, topk_eigs, sigma_f, gamma, m):
    """Find largest τ where (i) top-m eigs > λ_+(τ) AND (ii) all λ_k(τ)/λ_k(0) < 1/e."""
    lam_at_0 = topk_eigs[0, :m]  # eigenvalues at τ=0
    inv_e = 1.0 / np.e
    best_tau = None

    for i, tau in enumerate(taus):
        if tau == 0.0:
            continue
        # Condition (i): top-m eigenvalues above noise floor
        if tau == 0.0:
            _, lp = edge_tau0(sigma_f, gamma)
        else:
            _, lp = fc_edges(tau, sigma=sigma_f, gamma=gamma)
        eigs_m = topk_eigs[i, :m]
        if not np.all(eigs_m > lp):
            continue
        # Condition (ii): all λ_k(τ)/λ_k(0) < 1/e
        ratios = eigs_m / lam_at_0
        if not np.all(ratios < inv_e):
            continue
        best_tau = tau

    return best_tau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True,
                        help="Path to npz from simulate_planted_spikes_variants.py")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)

    # Detect format: vf0_cfg0_ (multi-factor) or cfg0_ (single-factor)
    if "vf0_cfg0_taus" in data:
        pfx_fmt = "vf0_cfg{}_"
    elif "cfg0_taus" in data:
        pfx_fmt = "cfg{}_"
    else:
        raise ValueError(f"Unknown npz format. Keys: {sorted(data.keys())[:10]}")

    d = int(data["d"])
    N = int(data["N"])
    gamma = d / N

    # Estimate sigma_f from the data (same as simulation)
    sigma_iso = float(data["sigma_iso"])

    configs = []
    for ci in range(3):
        pfx = pfx_fmt.format(ci)
        taus = data[pfx + "taus"]
        counts = data[pfx + "counts"]
        label = str(data[pfx + "label"])
        k_sep = int(data[pfx + "k_sep"])
        k_var = int(data[pfx + "k_var"])
        k_total = k_sep + k_var

        if pfx + "topk_eigs" in data:
            topk_eigs = data[pfx + "topk_eigs"]  # (n_lags, k_total)
        else:
            raise ValueError(f"npz missing '{pfx}topk_eigs'. Re-run simulation with updated script.")

        # Reconstruct sigma_f for noise floor
        # sigma_iso^2 = c^2 + sigma_small^2. For "20 sep": all axes have sigma_iso^2
        # For "20 var" or mixed: need sigma_big from npz
        if "sigma_big" in data:
            sigma_big = float(data["sigma_big"])
        else:
            sigma_big = sigma_iso  # fallback
        sigma_f_sq = (k_sep * sigma_iso**2 + k_var * sigma_big**2
                      + (d - k_total) * sigma_iso**2) / d
        sigma_f = float(np.sqrt(sigma_f_sq))

        # Compute noise-floor edges at each tau
        lp_by_tau = []
        for tau in taus:
            if tau == 0.0:
                _, lp = edge_tau0(sigma_f, gamma)
            else:
                _, lp = fc_edges(tau, sigma=sigma_f, gamma=gamma)
            lp_by_tau.append(lp)
        lp_by_tau = np.array(lp_by_tau)

        # m = planted count for this column
        m = k_total

        # Compute τ* with per-direction criterion
        tau_star = compute_tau_star(taus, topk_eigs, sigma_f, gamma, m)

        configs.append({
            "label": label, "taus": taus, "counts": counts,
            "topk_eigs": topk_eigs, "k_total": k_total, "k_sep": k_sep,
            "k_var": k_var, "m": m, "tau_star": tau_star,
            "lp_by_tau": lp_by_tau, "sigma_f": sigma_f,
        })

    # ---- Figure: 2 rows × 3 cols ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))

    for ci, cfg in enumerate(configs):
        taus = cfg["taus"]
        mask = taus > 0
        m = cfg["m"]

        # Top row: count panel
        ax = axes[0, ci]
        ax.plot(taus[mask], cfg["counts"][mask], "o-", color="C0", lw=1.6, ms=5)
        ax.axhline(m, color="crimson", ls="--", lw=1.4,
                   label=f"planted $k = {m}$")
        if cfg["tau_star"] is not None:
            ax.axvline(cfg["tau_star"], color="red", ls=":", lw=1.4, alpha=0.7,
                       label=rf"$\tau^* = {cfg['tau_star']:.1f}$")
        ax.set_xscale("log")
        ax.set_title(cfg["label"], fontweight="bold")
        if ci == 0:
            ax.set_ylabel(r"$\#\{\lambda_j > \lambda_+(\tau)\}$")
        ax.set_ylim(-0.5, m + 5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom row: λ_m(τ)/λ_m(0)
        ax = axes[1, ci]
        topk = cfg["topk_eigs"]  # (n_lags, k_total)
        lam_m_ratio = topk[mask, m - 1] / topk[0, m - 1]  # m-th eigenvalue ratio
        ax.plot(taus[mask], lam_m_ratio, "o-", color="C0", lw=1.6, ms=5)
        ax.axhline(1.0 / np.e, color="gray", ls="--", lw=1.2, alpha=0.7,
                   label="$1/e$")
        if cfg["tau_star"] is not None:
            ax.axvline(cfg["tau_star"], color="red", ls=":", lw=1.4, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\tau$")
        if ci == 0:
            ax.set_ylabel(rf"$\lambda_{{{m}}}(\tau) / \lambda_{{{m}}}(0)$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    sigma_iso_val = float(data["sigma_iso"])
    sigma_big_val = float(data.get("sigma_big", sigma_iso_val))
    vf = sigma_big_val**2 / sigma_iso_val**2
    tau_max_val = float(taus[-1])
    fig.suptitle(
        rf"$\sigma_f^2 = \mathrm{{tr}}(\Sigma_f)/d$, "
        rf"variance factor = {vf:.0f}x, $\tau_{{\max}} = {tau_max_val:.0f}$",
        fontsize=12,
    )
    fig.tight_layout()

    out_path = args.output or str(Path(args.npz).with_name("fig_spike_count_per_direction.png"))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print τ* summary
    for cfg in configs:
        ts = cfg["tau_star"]
        print(f"  {cfg['label']}: m={cfg['m']}, τ*={ts}")


if __name__ == "__main__":
    main()
