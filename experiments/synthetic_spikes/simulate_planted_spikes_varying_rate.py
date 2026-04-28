"""
Varying-rate planted-spike experiment (no matched-variance constraint).

Addresses a follow-up concern after `simulate_planted_spikes_nonuniform.py`:
the matched-variance constraint c_i^2 + sigma_{s,i}^2 = V used there made
"shallow" axes effectively unimodal (sigma_s grew as c shrank), so the
non-uniform script ended up testing whether DA correctly *ignores*
non-metastable axes, not whether the plateau rule recovers k under a
*range of metastable rates*.

This script drops the matched-variance constraint:

  - All k metastable axes share c = 3.0 (same well separation, deep wells).
  - sigma_{s,i} is varied geometrically over [0.3, 0.5] for i = 1, ..., k.
    Smaller sigma_s -> deeper barrier -> smaller mu_meta_i.
    The Kramers exponent c^2/(2 sigma_s^2 sigma_f^2/2) ~ 1/sigma_s^2 spans
    [4, 11.1] over this range, giving mu_meta_i variation of ~exp(7) ~ 1000x.
  - Isotropic axes (i >= k) use sigma_iso^2 = V_avg = c^2 + mean(sigma_s^2),
    which we also use as the global sigma_f^2 for the bulk-edge formula.

Per-axis stationary variance V_i = c^2 + sigma_{s,i}^2 then varies by only
~1.8% across axes, so PCA at tau = 0 still has essentially no traction.

Output:
  - Two-panel figure `fig_planted_spikes_varying_rate.png`:
        Left  : count(tau) curve, planted truth k, auto-selected tau*.
        Right : histogram of per-axis Kramers exponents (proxy for
                -log mu_meta_i), showing the rate spread.
  - Stdout numerical readout: sigma_s_i values, Kramers exponents,
    per-axis variances, m_hat, tau*, plateau width, count(tau) table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from simulate_planted_spikes import (  # noqa: E402
    _build_tau_grid, count_curve, auto_select_tau_star, random_rotation,
)
from simulate_planted_spikes_nonuniform import (  # noqa: E402
    _score_factorized_per_axis, _sample_initial_per_axis,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "noise_floor"))
from free_conv_utils import edges as fc_edges, edge_tau0  # noqa: E402

matplotlib.rcParams["figure.dpi"] = 120


# ============================================================================
# Per-axis simulation. Identical to simulate_per_axis() in the nonuniform
# script, except:
#   - sigma_iso (variance of the d-k isotropic axes) is decoupled from V,
#     and may differ slightly from per-axis V_i of the metastable axes.
#   - The "global" sigma_f passed to the free-conv bulk formula is set to
#     sigma_iso, since that is the variance of the (d-k)-dimensional
#     isotropic background that dominates the bulk.
# ============================================================================

def simulate_per_axis_varying_rate(
    d, k, c_arr, sigma_s_arr, sigma_iso_sq, N, dt, tau_steps, seed, rotation,
):
    assert c_arr.shape == (k,)
    assert sigma_s_arr.shape == (k,)
    sigma_iso = float(np.sqrt(sigma_iso_sq))
    sigma_f_sq = float(sigma_iso_sq)
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
    topvecs_by_tau: dict[float, np.ndarray] = {}
    for ts in tau_steps:
        Xtc = snapshots[ts] - mu_f
        C = (Xtc.T @ X0c) / N
        Csym = 0.5 * (C + C.T)
        evals, evecs = np.linalg.eigh(Csym)
        order = np.argsort(evals)[::-1]
        eigs_by_tau[round(ts * dt, 6)] = evals[order]
        topvecs_by_tau[round(ts * dt, 6)] = evecs[:, order[:k]]

    # PCA: top-k eigenvectors of the sample covariance Chat(0) = (1/N) X0 X0^T.
    # This is the standard PCA estimator; equivalent to top-k of Csym(0) up to
    # sign/order. We compute it directly for clarity.
    sample_cov = (X0c.T @ X0c) / N
    evals_pca, evecs_pca = np.linalg.eigh(sample_cov)
    order_pca = np.argsort(evals_pca)[::-1]
    pca_topk = evecs_pca[:, order_pca[:k]]

    return {
        "eigs_by_tau": eigs_by_tau,
        "topvecs_by_tau": topvecs_by_tau,
        "pca_topk": pca_topk,
        "sigma_f": sigma_f,
        "gamma": d / N,
        "d": d, "N": N, "k": k,
    }


def subspace_alignment(V, Q_planted):
    """||V^T Q_planted||_F^2 / k. Equals 1 for perfect recovery, k/d for random."""
    return float(np.linalg.norm(V.T @ Q_planted, ord="fro") ** 2 / Q_planted.shape[1])


def plateau_width(taus, counts, m_hat):
    on_plateau = counts == m_hat
    if not on_plateau.any() or m_hat == 0:
        return 0.0, None, None
    tlo = float(taus[on_plateau].min())
    thi = float(taus[on_plateau].max())
    return thi - tlo, tlo, thi


def _build_tau_grid_dense(dt, tau_max, n_log=48):
    """Denser variant of _build_tau_grid for capturing narrow plateaus."""
    raw = np.geomspace(max(2.0 * dt, 0.05), tau_max, n_log)
    steps = sorted({int(round(t / dt)) for t in raw})
    return [0] + steps


# ============================================================================
# Main
# ============================================================================

def run_varying_rate(
    out_path: Path | None = None,
    d: int = 200, k: int = 20,
    c_uniform: float = 3.0,
    sigma_s_min: float = 0.3, sigma_s_max: float = 0.5,
    N: int = 2000, dt: float = 0.005, tau_max: float = 10.0,
    seed_base: int = 7777, rotation_seed: int = 2026,
    use_dense_grid: bool = False,
):
    print("=" * 72)
    print("Varying-rate planted-spike experiment (no matched-variance constraint)")
    print("=" * 72)

    # ---- Per-axis parameters ------------------------------------------------
    c_arr = np.full(k, float(c_uniform))
    exponents = np.arange(k) / (k - 1)
    sigma_s_arr = sigma_s_min * (sigma_s_max / sigma_s_min) ** exponents
    V_i = c_arr ** 2 + sigma_s_arr ** 2          # per-axis stationary variance
    V_avg = float(np.mean(V_i))                  # used as sigma_iso^2 = sigma_f^2

    # Kramers exponent (proxy for -log mu_meta_i):
    #   B_i_eff = c^2 / (sigma_{s,i}^2 * sigma_f^2)
    # As specified in the design note. (sigma_f^2 = V_avg.)
    kramers = c_arr ** 2 / (sigma_s_arr ** 2 * V_avg)

    # ---- Print configuration ------------------------------------------------
    print(f"  d={d}, k={k}, N={N}, dt={dt}, tau_max={tau_max}")
    print(f"  c (uniform)            = {c_uniform}")
    print(f"  sigma_s range          = [{sigma_s_arr.min():.4f}, "
          f"{sigma_s_arr.max():.4f}]   geom ratio = "
          f"{sigma_s_arr.max()/sigma_s_arr.min():.4f}")
    print(f"  sigma_s mean           = {sigma_s_arr.mean():.4f}")
    print(f"  V_i = c^2 + sigma_s^2  : range = [{V_i.min():.4f}, {V_i.max():.4f}], "
          f"mean = {V_avg:.4f}, "
          f"rel-spread = {(V_i.max()-V_i.min())/V_i.mean()*100:.2f}%")
    print(f"  sigma_iso^2 = V_avg    = {V_avg:.4f}  (also used as sigma_f^2)")
    print(f"  Kramers exponents      : range = [{kramers.min():.4f}, "
          f"{kramers.max():.4f}]")
    print(f"     -> ~ exp(spread)   = exp({kramers.max()-kramers.min():.3f}) "
          f"= {np.exp(kramers.max()-kramers.min()):.1f}x dynamic range in mu_meta")

    print("\n  Per-axis table (i, sigma_s_i, V_i, Kramers_i):")
    for i in range(k):
        print(f"    i={i+1:>2d}  sigma_s={sigma_s_arr[i]:.4f}   "
              f"V_i={V_i[i]:.4f}   Kramers={kramers[i]:.4f}")

    # ---- tau-grid -----------------------------------------------------------
    grid_builder = _build_tau_grid_dense if use_dense_grid else _build_tau_grid
    tau_steps = grid_builder(dt, tau_max)
    grid_taus = np.array([round(ts * dt, 6) for ts in tau_steps])
    print(f"\n  tau-grid (n={len(grid_taus)}): "
          f"min(>0)={grid_taus[grid_taus > 0].min():.4f}, "
          f"max={grid_taus.max():.4f}, "
          f"dense={use_dense_grid}")

    # ---- Run simulation -----------------------------------------------------
    Q = random_rotation(d, seed=rotation_seed)
    sim = simulate_per_axis_varying_rate(
        d=d, k=k, c_arr=c_arr, sigma_s_arr=sigma_s_arr,
        sigma_iso_sq=V_avg, N=N, dt=dt,
        tau_steps=tau_steps, seed=seed_base, rotation=Q,
    )

    taus, counts = count_curve(sim["eigs_by_tau"], sim["sigma_f"], sim["gamma"])
    tau_star, m_hat = auto_select_tau_star(taus, counts)
    width, tlo, thi = plateau_width(taus, counts, m_hat)

    # PCA-vs-DA subspace alignment with the planted basis Q[:, :k].
    Q_planted = Q[:, :k]
    pca_align = subspace_alignment(sim["pca_topk"], Q_planted)
    da_align_at_tau_star = subspace_alignment(sim["topvecs_by_tau"][tau_star], Q_planted)
    chance = k / d
    da_align_by_tau = np.array([
        subspace_alignment(sim["topvecs_by_tau"][float(t)], Q_planted) for t in taus
    ])

    # ---- Numerical readout --------------------------------------------------
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"  m_hat (plateau-rule)      : {m_hat}   (planted k = {k})")
    print(f"  tau* (max tau in plateau) : {tau_star:.4f}")
    if tlo is not None and m_hat > 0:
        print(f"  plateau range             : [{tlo:.4f}, {thi:.4f}]   "
              f"width = {width:.4f}")
    else:
        print("  plateau range             : (empty / m_hat = 0)")
    print(f"\n  Subspace alignment with planted basis Q[:, :k] "
          f"(1 = perfect, {chance:.3f} = chance k/d):")
    print(f"    PCA  (top-k of sample covariance Chat(0)) : {pca_align:.4f}")
    print(f"    DA   (top-k of Csym(tau*)  at tau*={tau_star:.2f})  : "
          f"{da_align_at_tau_star:.4f}")
    print(f"    chance (k/d)                              : {chance:.4f}")

    print("\n  count(tau) curve:")
    print("       tau         count")
    for t, cnt in zip(taus, counts):
        marker = "  <-- plateau" if (m_hat > 0 and cnt == m_hat) else ""
        print(f"    {t:9.4f}    {cnt:5d}{marker}")

    # ---- Figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.0))

    # (a): count(tau) with truth & tau*
    ax = axes[0]
    ax.plot(taus, counts, "o-", color="C0", lw=1.6, markersize=5,
            label=r"$\#\{\lambda_j(\tau) > \lambda_+(\tau)\}$")
    ax.axhline(k, color="crimson", linestyle="--", lw=1.4,
               label=fr"planted truth $k={k}$")
    ax.axvline(tau_star, color="green", linestyle=":", lw=1.4,
               label=fr"auto $\tau^\star={tau_star:.2f}$")
    if tlo is not None and m_hat > 0:
        tlo_show = max(tlo, taus[taus > 0].min() if (taus > 0).any() else tlo)
        ax.axvspan(tlo_show, thi, color="green", alpha=0.10,
                   label=fr"plateau (width$={width:.2f}$)")
    ax.set_xscale("symlog", linthresh=max(2.0 * dt, 0.05))
    ax.set_xlabel(r"$\tau$ (lag, symlog)")
    ax.set_ylabel(r"$\#\{\lambda_j > \lambda_+(\tau)\}$")
    ax.set_ylim(-1, max(k + 5, counts.max() + 3))
    ax.set_title(rf"(a) plateau rule: $\hat m={m_hat}$ (planted $k={k}$)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # (b): histogram of Kramers exponent (proxy for -log mu_meta_i)
    ax = axes[1]
    n_bins = max(6, k // 3)
    ax.hist(kramers, bins=n_bins, color="steelblue", alpha=0.85,
            edgecolor="black", lw=0.5)
    ax.set_xlabel(r"Kramers exponent $\propto c^2 / (\sigma_{s,i}^2\,\sigma_f^2)$")
    ax.set_ylabel("count of axes")
    ax.set_title(rf"(b) per-axis rate spread "
                 rf"(${kramers.min():.1f}$ to ${kramers.max():.1f}$)")
    ax.grid(True, alpha=0.3, axis="y")

    # (c): subspace alignment vs tau, DA versus PCA
    ax = axes[2]
    ax.plot(taus, da_align_by_tau, "o-", color="C0", lw=1.6, markersize=5,
            label=r"DA (top-$k$ of $\hat C^{\rm sym}(\tau)$)")
    ax.axhline(pca_align, color="C1", linestyle="-", lw=1.6,
               label=fr"PCA (top-$k$ of $\hat C(0)$) $={pca_align:.3f}$")
    ax.axhline(chance, color="gray", linestyle=":", lw=1.2,
               label=fr"chance $=k/d={chance:.2f}$")
    ax.axhline(1.0, color="crimson", linestyle="--", lw=1.0, alpha=0.5,
               label="perfect $=1$")
    ax.set_xscale("symlog", linthresh=max(2.0 * dt, 0.05))
    ax.set_xlabel(r"$\tau$ (lag, symlog)")
    ax.set_ylabel(r"subspace alignment with $Q[:,\,:k]$")
    ax.set_title(rf"(c) DA recovers planted subspace; PCA at chance "
                 rf"(DA@$\tau^\star$ $={da_align_at_tau_star:.3f}$)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf"Varying-rate planted spikes (no matched-V constraint): "
        rf"$d={d}$, $k={k}$, $\gamma=d/N={d/N:.2f}$, $c={c_uniform}$, "
        rf"$\sigma_s\in[{sigma_s_min},{sigma_s_max}]$ geometric",
        y=1.01, fontsize=11,
    )
    fig.tight_layout()

    if out_path is None:
        out_path = HERE / "fig_planted_spikes_varying_rate.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved figure to: {out_path}")

    return {
        "c_arr": c_arr, "sigma_s_arr": sigma_s_arr, "V_i": V_i,
        "V_avg": V_avg, "kramers": kramers,
        "taus": taus, "counts": counts,
        "m_hat": m_hat, "tau_star": tau_star,
        "plateau_width": width, "plateau_lo": tlo, "plateau_hi": thi,
        "pca_align": pca_align, "da_align_at_tau_star": da_align_at_tau_star,
        "da_align_by_tau": da_align_by_tau, "chance": chance,
        "sigma_f": sim["sigma_f"], "gamma": sim["gamma"],
        "out_path": out_path,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Varying-rate planted-spike experiment.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output figure path (default: "
                             "fig_planted_spikes_varying_rate.png in this directory).")
    parser.add_argument("--dense", action="store_true",
                        help="Use a denser tau grid (n_log=48 instead of 24) "
                             "for narrow-plateau detection.")
    args = parser.parse_args()
    run_varying_rate(
        out_path=Path(args.out) if args.out else None,
        use_dense_grid=args.dense,
    )
