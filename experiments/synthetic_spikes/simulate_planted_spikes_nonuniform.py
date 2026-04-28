"""
Non-uniform barrier-height variant of the planted-spike experiment.

Addresses the concern that the existing experiment has all k=20 metastable
axes at the *same* barrier height (c=3.0, sigma_s=0.3), which trivially
produces a wide flat plateau because all 20 slow eigenvalues are degenerate.

This script keeps the matched per-axis stationary variance V = c_i^2 + sigma_{s,i}^2
constant across axes (so PCA at tau=0 is still blind to the planted basis), but
varies c_i across axes to produce a *range* of barrier heights
B_i = c_i^2 / (2 sigma_{s,i}^2).

Three configurations side-by-side:
  1. Uniform              : c_i = 3.0 for all i.
  2. Geometric gradient   : c_i geometric from 1.5 to 3.0.
  3. Two-group split      : c_i = 3.0 for i=1..10, c_i = 1.7 for i=11..20.

For each, runs the canonical diffusion, computes count(tau), reports the
plateau-rule estimate m_hat and tau*, and saves a 3-panel comparison figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Reuse infrastructure from the existing baseline script.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from simulate_planted_spikes import (  # noqa: E402
    _build_tau_grid, count_curve, auto_select_tau_star, random_rotation,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "noise_floor"))
from free_conv_utils import edges as fc_edges, edge_tau0  # noqa: E402

matplotlib.rcParams["figure.dpi"] = 120


# ============================================================================
# Per-axis-parameterized score and initial sampler.
#
# Each metastable axis i has its own (c_i, sigma_{s,i}); isotropic axes (i>=k)
# use sigma_iso. The score is the gradient of the log-density of a mixture of
# two Gaussians with means +/-c_i and standard deviation sigma_{s,i}.
# All operations are vectorized over (N, k) via broadcasting.
# ============================================================================

def _score_factorized_per_axis(X, c_arr, sigma_s_arr, sigma_iso, k):
    """
    X         : (N, d)
    c_arr     : (k,) per-axis well-center
    sigma_s_arr : (k,) per-axis well-width
    sigma_iso : scalar isotropic std for axes i >= k
    """
    Z = X[:, :k]                                     # (N, k)
    s2 = sigma_s_arr[None, :] ** 2                   # (1, k)
    c_b = c_arr[None, :]                             # (1, k)
    lp1 = -((Z - c_b) ** 2) / (2.0 * s2)
    lp2 = -((Z + c_b) ** 2) / (2.0 * s2)
    m = np.maximum(lp1, lp2)
    p1 = np.exp(lp1 - m)
    p2 = np.exp(lp2 - m)
    norm = p1 + p2
    p1 /= norm
    p2 /= norm
    s_meta = -((Z - c_b) * p1 + (Z + c_b) * p2) / s2
    s_iso = -X[:, k:] / (sigma_iso ** 2)
    out = np.empty_like(X)
    out[:, :k] = s_meta
    out[:, k:] = s_iso
    return out


def _sample_initial_per_axis(N, d, k, c_arr, sigma_s_arr, sigma_iso, rng):
    X = np.empty((N, d))
    signs = rng.choice([-1.0, 1.0], size=(N, k))
    X[:, :k] = signs * c_arr[None, :] + rng.normal(0.0, 1.0, size=(N, k)) * sigma_s_arr[None, :]
    X[:, k:] = rng.normal(0.0, sigma_iso, size=(N, d - k))
    return X


# ============================================================================
# Simulation with per-axis (c_i, sigma_{s,i}). The drift coefficient is
# 1/2 * sigma_f^2 with sigma_f^2 = V (matched variance), a single scalar.
# ============================================================================

def simulate_per_axis(
    d, k, c_arr, sigma_s_arr, V, N, dt, tau_steps, seed, rotation,
):
    assert c_arr.shape == (k,)
    assert sigma_s_arr.shape == (k,)
    sigma_iso = float(np.sqrt(V))            # axes i >= k: variance V
    sigma_f_sq = float(V)
    sigma_f = float(np.sqrt(sigma_f_sq))

    rng = np.random.default_rng(seed)
    X_fact = _sample_initial_per_axis(N, d, k, c_arr, sigma_s_arr, sigma_iso, rng)
    X = X_fact @ rotation.T

    snapshots: dict[int, np.ndarray] = {0: X.copy()}
    n_steps = max(tau_steps)
    sqrt_dt = float(np.sqrt(dt))
    drift_pref = 0.5 * sigma_f_sq

    for t in range(1, n_steps + 1):
        X_in_fact = X @ rotation
        grad_fact = _score_factorized_per_axis(X_in_fact, c_arr, sigma_s_arr, sigma_iso, k)
        grad = grad_fact @ rotation.T
        Z = rng.standard_normal((N, d))
        X = X + drift_pref * grad * dt + sigma_f * sqrt_dt * Z
        if t in tau_steps:
            snapshots[t] = X.copy()

    mu_f = np.zeros(d)
    X0c = snapshots[0] - mu_f
    eigs_by_tau: dict[float, np.ndarray] = {}
    for ts in tau_steps:
        Xtc = snapshots[ts] - mu_f
        C = (Xtc.T @ X0c) / N
        Csym = 0.5 * (C + C.T)
        eigs = np.sort(np.linalg.eigvalsh(Csym))[::-1]
        eigs_by_tau[round(ts * dt, 6)] = eigs

    return {
        "eigs_by_tau": eigs_by_tau,
        "sigma_f": sigma_f,
        "gamma": d / N,
        "d": d, "N": N, "k": k,
    }


# ============================================================================
# Configuration builders. All preserve V = c_i^2 + sigma_{s,i}^2 constant.
# ============================================================================

def config_uniform(k, V, c0=3.0):
    c_arr = np.full(k, c0)
    sigma_s_arr = np.sqrt(V - c_arr ** 2)
    return c_arr, sigma_s_arr


def config_geometric(k, V, c_min=1.5, c_max=3.0):
    # i = 1, ..., k. Ratio (c_max/c_min)^((i-1)/(k-1)). Index 0 -> c_min, index k-1 -> c_max.
    exponents = np.arange(k) / (k - 1)
    c_arr = c_min * (c_max / c_min) ** exponents
    sigma_s_arr = np.sqrt(V - c_arr ** 2)
    return c_arr, sigma_s_arr


def config_two_group(k, V, c_deep=3.0, c_shallow=1.7, n_deep=None):
    if n_deep is None:
        n_deep = k // 2
    c_arr = np.empty(k)
    c_arr[:n_deep] = c_deep
    c_arr[n_deep:] = c_shallow
    sigma_s_arr = np.sqrt(V - c_arr ** 2)
    return c_arr, sigma_s_arr


def barrier_heights(c_arr, sigma_s_arr):
    return c_arr ** 2 / (2.0 * sigma_s_arr ** 2)


# ============================================================================
# Reporting helpers
# ============================================================================

def _summarize(name, vals):
    return (f"  {name}: min={vals.min():.4f}, max={vals.max():.4f}, "
            f"mean={vals.mean():.4f}, std={vals.std():.4f}")


def plateau_width(taus, counts, m_hat):
    on_plateau = counts == m_hat
    if not on_plateau.any() or m_hat == 0:
        return 0.0, None, None
    tlo = float(taus[on_plateau].min())
    thi = float(taus[on_plateau].max())
    return thi - tlo, tlo, thi


def report_config(label, c_arr, sigma_s_arr, taus, counts, m_hat, tau_star, k):
    B = barrier_heights(c_arr, sigma_s_arr)
    width, tlo, thi = plateau_width(taus, counts, m_hat)
    print(f"\n--- {label} ---")
    print(_summarize("c_i        ", c_arr))
    print(_summarize("sigma_{s,i}", sigma_s_arr))
    print(_summarize("B_i = c^2/(2 sigma_s^2)", B))
    print(f"  m_hat (plateau-rule)      : {m_hat}   (planted k = {k})")
    print(f"  tau* (max tau in plateau) : {tau_star:.4f}")
    if tlo is not None:
        print(f"  plateau range             : [{tlo:.4f}, {thi:.4f}]   width = {width:.4f}")
    else:
        print(f"  plateau range             : (empty / m_hat=0)")
    print("  count(tau) curve:")
    print("       tau         count")
    for t, cnt in zip(taus, counts):
        print(f"    {t:9.4f}    {cnt:5d}")


# ============================================================================
# Main
# ============================================================================

def run_nonuniform(out_path: Path | None = None,
                   d: int = 200, k: int = 20, V: float = 9.09,
                   N: int = 2000, dt: float = 0.005, tau_max: float = 10.0,
                   seed_base: int = 4242):
    print("=" * 72)
    print("Non-uniform barrier-height planted-spike experiment")
    print("=" * 72)
    print(f"  d={d}, k={k}, V (per-axis variance)={V}, N={N}, dt={dt}, "
          f"tau_max={tau_max}")
    print(f"  Each configuration uses a fresh random rotation Q.")

    tau_steps = _build_tau_grid(dt, tau_max)
    grid_taus = np.array([round(ts * dt, 6) for ts in tau_steps])
    print(f"  tau-grid (n={len(grid_taus)}): "
          f"min={grid_taus[grid_taus > 0].min():.4f}, max={grid_taus.max():.4f}")

    configs = [
        ("Uniform (c=3.0)", *config_uniform(k, V, c0=3.0), 1001),
        ("Geometric gradient c in [1.5, 3.0]",
         *config_geometric(k, V, c_min=1.5, c_max=3.0), 1002),
        ("Two-group: 10 deep (c=3.0) / 10 shallow (c=1.7)",
         *config_two_group(k, V, c_deep=3.0, c_shallow=1.7, n_deep=k // 2), 1003),
    ]

    results = []
    for label, c_arr, sigma_s_arr, rot_seed in configs:
        Q = random_rotation(d, seed=rot_seed)
        sim = simulate_per_axis(d, k, c_arr, sigma_s_arr, V, N, dt, tau_steps,
                                seed=seed_base + rot_seed, rotation=Q)
        taus, counts = count_curve(sim["eigs_by_tau"], sim["sigma_f"], sim["gamma"])
        tau_star, m_hat = auto_select_tau_star(taus, counts)
        report_config(label, c_arr, sigma_s_arr, taus, counts, m_hat, tau_star, k)
        results.append({
            "label": label,
            "c_arr": c_arr,
            "sigma_s_arr": sigma_s_arr,
            "taus": taus,
            "counts": counts,
            "m_hat": m_hat,
            "tau_star": tau_star,
            "sigma_f": sim["sigma_f"],
            "gamma": sim["gamma"],
        })

    # ------------------------------------------------------------------------
    # Figure: 2 rows x 3 cols.
    # Row 1: count(tau) per configuration.
    # Row 2: barrier-height histogram per configuration.
    # ------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.5))

    for j, res in enumerate(results):
        taus = res["taus"]
        counts = res["counts"]
        m_hat = res["m_hat"]
        tau_star = res["tau_star"]
        width, tlo, thi = plateau_width(taus, counts, m_hat)

        ax = axes[0, j]
        ax.plot(taus, counts, "o-", color="C0", lw=1.6, markersize=5,
                label=r"$\#\{\lambda_j(\tau) > \lambda_+(\tau)\}$")
        ax.axhline(k, color="crimson", linestyle="--", lw=1.4,
                   label=fr"planted truth $k={k}$")
        ax.axvline(tau_star, color="green", linestyle=":", lw=1.4,
                   label=fr"$\tau^\star={tau_star:.2f}$")
        if tlo is not None and m_hat > 0:
            ax.axvspan(max(tlo, taus[taus > 0].min() if (taus > 0).any() else tlo),
                       thi, color="green", alpha=0.08,
                       label=fr"plateau (width$={width:.2f}$)")
        ax.set_xscale("symlog", linthresh=max(2.0 * dt, 0.05))
        ax.set_xlabel(r"$\tau$ (lag, symlog)")
        ax.set_ylabel(r"$\#\{\lambda_j > \lambda_+(\tau)\}$")
        ax.set_ylim(-1, max(k + 5, counts.max() + 3))
        ax.set_title(f"{res['label']}\n"
                     fr"$\hat m={m_hat}$, plateau width$={width:.2f}$")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        ax = axes[1, j]
        c_arr = res["c_arr"]
        sigma_s_arr = res["sigma_s_arr"]
        B = barrier_heights(c_arr, sigma_s_arr)
        idx = np.arange(1, k + 1)
        ax.bar(idx, B, color="steelblue", alpha=0.8, edgecolor="black", lw=0.5)
        ax.set_xlabel(r"axis index $i$")
        ax.set_ylabel(r"barrier height $B_i = c_i^2 / (2\sigma_{s,i}^2)$")
        ax.set_yscale("log")
        ax.set_title(fr"per-axis barriers ($V={V}$ matched)")
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        rf"Plateau rule under non-uniform barrier heights "
        rf"($d={d}$, $k={k}$, $\gamma=d/N={d/N:.2f}$, "
        rf"per-axis $c_i^2+\sigma_{{s,i}}^2 = V = {V}$ matched)",
        y=1.005, fontsize=11,
    )
    fig.tight_layout()

    if out_path is None:
        out_path = HERE / "fig_planted_spikes_nonuniform.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved figure to: {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Non-uniform barrier-height planted-spike experiment.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output figure path (default: "
                             "fig_planted_spikes_nonuniform.png in this directory).")
    args = parser.parse_args()
    run_nonuniform(out_path=Path(args.out) if args.out else None)
