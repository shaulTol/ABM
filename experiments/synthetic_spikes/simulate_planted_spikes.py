"""
Synthetic baseline (Design A): factorized double-well metastable axes embedded
in a higher-dimensional isotropic Gaussian background, with matched per-axis
variance so PCA at tau=0 sees no preferred direction.

Six panels:
    (a) tau = 0 spectrum: empirical vs analytic free-conv bulk; no spikes.
    (b) tau = tau* (auto-selected) spectrum: 20 spikes outside the bulk.
    (c) count(tau) vs tau, plateau detection, auto-selected tau*.
    (d) detected count vs N (sample-complexity / detection-threshold curve).
    (e) detected count vs k_planted (linearity, identity line).
    (f) rotated frame: count(tau) overlap with unrotated, subspace alignment.

Run:
    python3 simulate_planted_spikes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "noise_floor"))
from free_conv_utils import edges as fc_edges, density as fc_density, edge_tau0  # noqa: E402

matplotlib.rcParams["figure.dpi"] = 120


# ============================================================================
# Score function for the factorized double-well + isotropic Gaussian energy.
#   U(x) = sum_{i=1}^{k} V_meta(x_i; c, sigma_small)
#        + sum_{i=k+1}^{d} x_i^2 / (2 sigma_iso^2)
# ============================================================================

def _score_factorized(X, c, sigma_small, sigma_iso, k):
    """Score of the factorized double-well + iso-Gaussian energy on (N, d)."""
    s2 = sigma_small ** 2
    Z = X[:, :k]
    lp1 = -((Z - c) ** 2) / (2 * s2)
    lp2 = -((Z + c) ** 2) / (2 * s2)
    m = np.maximum(lp1, lp2)
    p1 = np.exp(lp1 - m)
    p2 = np.exp(lp2 - m)
    norm = p1 + p2
    p1 /= norm
    p2 /= norm
    s_meta = -((Z - c) * p1 + (Z + c) * p2) / s2
    s_iso = -X[:, k:] / (sigma_iso ** 2)
    out = np.empty_like(X)
    out[:, :k] = s_meta
    out[:, k:] = s_iso
    return out


def _sample_initial(N, d, k, c, sigma_small, sigma_iso, rng):
    X = np.empty((N, d))
    signs = rng.choice([-1.0, 1.0], size=(N, k))
    X[:, :k] = signs * c + rng.normal(0.0, sigma_small, size=(N, k))
    X[:, k:] = rng.normal(0.0, sigma_iso, size=(N, d - k))
    return X


def _build_tau_grid(dt, tau_max, n_log=24):
    raw = np.geomspace(max(2.0 * dt, 0.05), tau_max, n_log)
    steps = sorted({int(round(t / dt)) for t in raw})
    return [0] + steps


# ============================================================================
# Single simulation. Returns eigenvalues of the symmetrized lag-tau
# autocovariance at each lag in tau_steps, and the top-k eigenvectors at the
# largest lag (used for the rotation-invariance subspace check).
# ============================================================================

def simulate(
    d, k, c, sigma_small, N, dt, tau_steps, seed,
    rotation: np.ndarray | None = None,
    return_top_eigvecs: bool = False,
):
    """Run the canonical diffusion and return eigenvalues at each lag.

    rotation : optional orthogonal d x d matrix Q. The score and initial
    samples are evaluated in the rotated frame: planted directions are
    Q e_1, ..., Q e_k. If None, planted directions are coordinate axes.
    """
    sigma_iso = float(np.sqrt(c ** 2 + sigma_small ** 2))
    sigma_f_sq = c ** 2 + sigma_small ** 2
    sigma_f = float(np.sqrt(sigma_f_sq))

    rng = np.random.default_rng(seed)
    X_fact = _sample_initial(N, d, k, c, sigma_small, sigma_iso, rng)
    if rotation is not None:
        X = X_fact @ rotation.T  # rows are samples; rotate each sample
    else:
        X = X_fact

    snapshots: dict[int, np.ndarray] = {0: X.copy()}
    n_steps = max(tau_steps)
    sqrt_dt = float(np.sqrt(dt))
    drift_pref = 0.5 * sigma_f_sq

    for t in range(1, n_steps + 1):
        if rotation is not None:
            X_in_fact = X @ rotation  # = Q^T y
            grad_fact = _score_factorized(X_in_fact, c, sigma_small, sigma_iso, k)
            grad = grad_fact @ rotation.T
        else:
            grad = _score_factorized(X, c, sigma_small, sigma_iso, k)
        Z = rng.standard_normal((N, d))
        X = X + drift_pref * grad * dt + sigma_f * sqrt_dt * Z
        if t in tau_steps:
            snapshots[t] = X.copy()

    mu_f = np.zeros(d)
    X0c = snapshots[0] - mu_f
    eigs_by_tau: dict[float, np.ndarray] = {}
    top_eigvecs = None
    last_step = max(tau_steps)
    for ts in tau_steps:
        Xtc = snapshots[ts] - mu_f
        C = (Xtc.T @ X0c) / N
        Csym = 0.5 * (C + C.T)
        if return_top_eigvecs and ts == last_step:
            evals, evecs = np.linalg.eigh(Csym)
            order = np.argsort(evals)[::-1]
            eigs_by_tau[round(ts * dt, 6)] = evals[order]
            top_eigvecs = evecs[:, order[:k]]
        else:
            eigs = np.sort(np.linalg.eigvalsh(Csym))[::-1]
            eigs_by_tau[round(ts * dt, 6)] = eigs

    return {
        "eigs_by_tau": eigs_by_tau,
        "sigma_f": sigma_f,
        "gamma": d / N,
        "d": d,
        "N": N,
        "k": k,
        "top_eigvecs": top_eigvecs,
    }


def count_curve(eigs_by_tau, sigma_f, gamma):
    taus = []
    counts = []
    edges_lp = []
    for tau, eigs in eigs_by_tau.items():
        if tau == 0.0:
            _, lp = edge_tau0(sigma_f, gamma)
        else:
            _, lp = fc_edges(tau, sigma=sigma_f, gamma=gamma)
        taus.append(tau)
        edges_lp.append(lp)
        counts.append(int(np.sum(eigs > lp)))
    return np.array(taus), np.array(counts), np.array(edges_lp)


def auto_select_tau_star(taus, counts):
    """Plateau rule: pick the largest count value, take the largest tau where
    count attains it. Matches the paper's 'largest tau where top-m exceeds
    lambda_+' criterion with m made an output rather than an input."""
    plateau_count = int(counts.max())
    if plateau_count == 0:
        return float(taus[-1]), 0
    on_plateau = counts == plateau_count
    return float(taus[on_plateau].max()), plateau_count


# ============================================================================
# Sweeps
# ============================================================================

def sweep_N(d, k, c, sigma_small, dt, tau_max, N_values, seed=0):
    """For each N, run the simulation and return the auto-detected count."""
    out = []
    for i, N in enumerate(N_values):
        tau_steps = _build_tau_grid(dt, tau_max)
        res = simulate(d, k, c, sigma_small, N, dt, tau_steps, seed=seed + i)
        taus, counts, _ = count_curve(res["eigs_by_tau"], res["sigma_f"], res["gamma"])
        tau_star, m_hat = auto_select_tau_star(taus, counts)
        print(f"  N={N:>5d}  gamma={res['gamma']:.3f}  tau*={tau_star:.2f}  "
              f"detected count = {m_hat:>3d}  (truth k={k})")
        out.append((N, m_hat, tau_star))
    return out


def sweep_k(d, c, sigma_small, N, dt, tau_max, k_values, seed=0):
    out = []
    for i, k in enumerate(k_values):
        tau_steps = _build_tau_grid(dt, tau_max)
        res = simulate(d, k, c, sigma_small, N, dt, tau_steps, seed=seed + i)
        taus, counts, _ = count_curve(res["eigs_by_tau"], res["sigma_f"], res["gamma"])
        tau_star, m_hat = auto_select_tau_star(taus, counts)
        print(f"  k_planted={k:>3d}  tau*={tau_star:.2f}  "
              f"detected count = {m_hat:>3d}")
        out.append((k, m_hat, tau_star))
    return out


def run_rotated(d, k, c, sigma_small, N, dt, tau_max, seed=0):
    """Random rotation of the planted basis; check that count(tau) and the
    top-k subspace match the unrotated result."""
    rng = np.random.default_rng(1234 + seed)
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    tau_steps = _build_tau_grid(dt, tau_max)
    res = simulate(d, k, c, sigma_small, N, dt, tau_steps, seed=seed,
                   rotation=Q, return_top_eigvecs=True)
    taus, counts, _ = count_curve(res["eigs_by_tau"], res["sigma_f"], res["gamma"])
    tau_star, m_hat = auto_select_tau_star(taus, counts)

    # Subspace alignment: Frobenius norm of projection of planted basis onto
    # recovered top-k subspace, divided by k. =1 iff perfect recovery.
    planted = Q[:, :k]  # d x k
    V = res["top_eigvecs"]  # d x k
    align = float(np.linalg.norm(V.T @ planted, ord="fro") ** 2 / k)
    print(f"  rotated: tau*={tau_star:.2f}  detected count={m_hat}  "
          f"subspace alignment = {align:.4f}")
    return {
        "taus": taus, "counts": counts, "tau_star": tau_star,
        "m_hat": m_hat, "alignment": align,
    }


# ============================================================================
# Main: 6-panel figure
# ============================================================================

def main(out_path: Path | None = None):
    # Baseline config
    d = 200
    k = 20
    c = 3.0
    sigma_small = 0.3
    N = 2000
    dt = 0.005
    tau_max = 10.0

    print("=" * 70)
    print("baseline run")
    print("=" * 70)
    tau_steps = _build_tau_grid(dt, tau_max)
    res = simulate(d, k, c, sigma_small, N, dt, tau_steps, seed=0)
    taus, counts, edges_lp = count_curve(res["eigs_by_tau"], res["sigma_f"], res["gamma"])
    tau_star, m_hat = auto_select_tau_star(taus, counts)
    sigma_f = res["sigma_f"]
    gamma = res["gamma"]
    print(f"baseline: tau*={tau_star:.2f}, detected count={m_hat}, planted k={k}")

    print("\n" + "=" * 70)
    print("sweep over N")
    print("=" * 70)
    N_values = [200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
    sweep_N_out = sweep_N(d, k, c, sigma_small, dt, tau_max, N_values, seed=10)

    print("\n" + "=" * 70)
    print("sweep over k_planted")
    print("=" * 70)
    k_values = [5, 10, 20, 40, 80]
    sweep_k_out = sweep_k(d, c, sigma_small, N, dt, tau_max, k_values, seed=20)

    print("\n" + "=" * 70)
    print("rotated frame")
    print("=" * 70)
    rot = run_rotated(d, k, c, sigma_small, N, dt, tau_max, seed=30)

    # ------------------------------------------------------------------------
    # Assemble 6-panel figure
    # ------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.5))

    # (a) tau = 0 spectrum
    ax = axes[0, 0]
    eigs0 = res["eigs_by_tau"][0.0]
    lm0, lp0 = edge_tau0(sigma_f, gamma)
    bins = np.linspace(min(eigs0.min(), lm0) - 0.2,
                       max(eigs0.max(), lp0) + 0.2, 60)
    ax.hist(eigs0, bins=bins, density=True, color="steelblue", alpha=0.55,
            label=f"empirical ($d={d}$, $N={N}$)")
    lam_grid = np.linspace(lm0, lp0, 500)
    ax.plot(lam_grid, fc_density(lam_grid, 0.0, sigma=sigma_f, gamma=gamma),
            "g--", lw=2, label="free-conv (= MP at $\\tau=0$)")
    ax.axvline(lp0, color="g", linestyle=":", lw=1.2,
               label=fr"$\lambda_+={lp0:.2f}$")
    ax.axvline(lm0, color="g", linestyle=":", lw=1.2)
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("density")
    ax.set_title(r"(a) $\tau = 0$: PCA-blind, no spikes visible")
    ax.legend(fontsize=8)

    # (b) tau* spectrum
    ax = axes[0, 1]
    eigs_t = res["eigs_by_tau"][tau_star]
    lm_r, lp_r = fc_edges(tau_star, sigma=sigma_f, gamma=gamma)
    bins = np.linspace(min(eigs_t.min(), lm_r) - 0.2,
                       max(eigs_t.max(), lp_r) + 0.5, 60)
    ax.hist(eigs_t, bins=bins, density=True, color="steelblue", alpha=0.55,
            label="empirical")
    lam_grid = np.linspace(lm_r, lp_r, 500)
    ax.plot(lam_grid, fc_density(lam_grid, tau_star, sigma=sigma_f, gamma=gamma),
            "g--", lw=2, label="free-conv density (Thm 3)")
    ax.axvline(lp_r, color="g", linestyle=":", lw=1.2,
               label=fr"$\lambda_+(\tau^*)={lp_r:.2f}$")
    ax.axvline(lm_r, color="g", linestyle=":", lw=1.2)
    yloc = ax.get_ylim()[1] * 0.05 if ax.get_ylim()[1] > 0 else 0.05
    ax.scatter(eigs_t[:m_hat], np.full(m_hat, yloc),
               marker="|", s=180, color="crimson", zorder=5,
               label=fr"top-{m_hat} eigenvalues")
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("density")
    ax.set_title(rf"(b) $\tau^*={tau_star:.1f}$: {m_hat} spikes "
                 rf"(planted $k={k}$)")
    ax.legend(fontsize=8)

    # (c) count(tau) curve with auto-selection
    ax = axes[0, 2]
    ax.plot(taus, counts, "o-", color="C0", lw=1.6, markersize=5,
            label=r"$\#\{\lambda_j(\tau) > \lambda_+(\tau)\}$")
    ax.axhline(k, color="crimson", linestyle="--", lw=1.4,
               label=fr"planted truth $k={k}$")
    ax.axvline(tau_star, color="green", linestyle=":", lw=1.4,
               label=fr"auto $\tau^*={tau_star:.1f}$")
    ax.set_xscale("symlog", linthresh=max(2.0 * dt, 0.05))
    ax.set_xlabel(r"$\tau$ (lag, symlog)")
    ax.set_ylabel(r"$\#\{\lambda_j > \lambda_+(\tau)\}$")
    ax.set_title(r"(c) auto-selection of $\tau^*$ via plateau")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) detected count vs N
    ax = axes[1, 0]
    Ns = np.array([t[0] for t in sweep_N_out])
    ms = np.array([t[1] for t in sweep_N_out])
    ax.plot(Ns, ms, "o-", color="C0", lw=1.6, markersize=6,
            label="detected count")
    ax.axhline(k, color="crimson", linestyle="--", lw=1.4,
               label=fr"planted truth $k={k}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N$ (number of trajectories)")
    ax.set_ylabel("detected count")
    ax.set_title(rf"(d) sample-complexity sweep ($d={d}$, $k={k}$ fixed)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (e) detected count vs k_planted
    ax = axes[1, 1]
    ks = np.array([t[0] for t in sweep_k_out])
    ms = np.array([t[1] for t in sweep_k_out])
    ax.plot(ks, ms, "o-", color="C0", lw=1.6, markersize=6,
            label="detected count")
    line_max = max(ks.max(), ms.max()) + 5
    ax.plot([0, line_max], [0, line_max], "k:", lw=1, alpha=0.6,
            label="identity (perfect recovery)")
    ax.set_xlabel(r"planted $k$ (number of metastable axes)")
    ax.set_ylabel("detected count")
    ax.set_title(rf"(e) recovery vs $k_{{\rm planted}}$ "
                 rf"($d={d}$, $N={N}$ fixed)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (f) rotated frame: count overlap + subspace alignment annotation
    ax = axes[1, 2]
    ax.plot(taus, counts, "o-", color="C0", lw=1.6, markersize=4,
            label="unrotated (axis-aligned)", alpha=0.85)
    ax.plot(rot["taus"], rot["counts"], "s--", color="C1", lw=1.6,
            markersize=4, label="random rotation $Q$", alpha=0.85)
    ax.axhline(k, color="crimson", linestyle="--", lw=1.4,
               label=fr"planted truth $k={k}$")
    ax.set_xscale("symlog", linthresh=max(2.0 * dt, 0.05))
    ax.set_xlabel(r"$\tau$ (lag, symlog)")
    ax.set_ylabel(r"$\#\{\lambda_j > \lambda_+(\tau)\}$")
    ax.set_title(rf"(f) rotation invariance "
                 rf"(subspace alignment $={rot['alignment']:.3f}$)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf"Synthetic planted-spike validation: $d={d}$, $k={k}$, "
        rf"$\gamma=d/N={gamma:.2f}$, $c={c}$, $\sigma_s={sigma_small}$, "
        rf"matched per-axis variance",
        y=1.005, fontsize=11,
    )
    fig.tight_layout()

    if out_path is None:
        out_path = Path(__file__).parent / "fig_planted_spikes.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved figure to: {out_path}")


if __name__ == "__main__":
    main()
