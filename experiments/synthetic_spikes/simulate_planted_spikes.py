"""
Synthetic baseline (Design A): factorized double-well metastable axes embedded
in a higher-dimensional isotropic Gaussian background, with matched per-axis
variance so PCA at tau=0 sees no preferred direction.

A single random orthogonal Q is drawn at the start: planted directions are
q_1 = Q e_1, ..., q_k = Q e_k. Under similarity invariance the eigenvalue
distributions of Csym(tau) are identical regardless of Q, but Q removes any
suggestion of an axis-aligned (sparse) signal assumption.

Six panels:
    (a) tau = 0 spectrum: empirical vs analytic free-conv bulk; no spikes.
    (b) tau = tau* (auto-selected) spectrum: 20 spikes outside the bulk.
    (c) count(tau) vs tau, plateau detection, auto-selected tau*.
    (d) detected count vs N (sample-complexity / detection-threshold curve).
    (e) detected count vs k_planted (linearity, identity line).
    (f) subspace alignment of recovered top-k vs planted basis Q[:, :k]:
        DA-at-lag-tau (rises with tau) vs PCA-equivalent (constant at chance).

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
# Planted directions are rotated by Q applied externally.
# ============================================================================

def _score_factorized(X, c, sigma_small, sigma_iso, k):
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


def random_rotation(d, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    return Q


# ============================================================================
# Single simulation. Returns eigenvalues at each lag; optionally returns the
# top-k eigenvectors of Csym(tau) at every lag (for subspace alignment).
# ============================================================================

def simulate(
    d, k, c, sigma_small, N, dt, tau_steps, seed,
    rotation: np.ndarray,
    return_top_eigvecs_all_lags: bool = False,
):
    sigma_iso = float(np.sqrt(c ** 2 + sigma_small ** 2))
    sigma_f_sq = c ** 2 + sigma_small ** 2
    sigma_f = float(np.sqrt(sigma_f_sq))

    rng = np.random.default_rng(seed)
    X_fact = _sample_initial(N, d, k, c, sigma_small, sigma_iso, rng)
    X = X_fact @ rotation.T   # samples in rotated frame

    snapshots: dict[int, np.ndarray] = {0: X.copy()}
    n_steps = max(tau_steps)
    sqrt_dt = float(np.sqrt(dt))
    drift_pref = 0.5 * sigma_f_sq

    for t in range(1, n_steps + 1):
        X_in_fact = X @ rotation        # = Q^T y, into factorized frame
        grad_fact = _score_factorized(X_in_fact, c, sigma_small, sigma_iso, k)
        grad = grad_fact @ rotation.T   # back to rotated frame
        Z = rng.standard_normal((N, d))
        X = X + drift_pref * grad * dt + sigma_f * sqrt_dt * Z
        if t in tau_steps:
            snapshots[t] = X.copy()

    mu_f = np.zeros(d)
    X0c = snapshots[0] - mu_f
    C0 = (X0c.T @ X0c) / N
    tr_C0 = np.trace(C0)

    eigs_by_tau: dict[float, np.ndarray] = {}
    topvecs_by_tau: dict[float, np.ndarray] = {}
    rho_by_tau: dict[float, float] = {}
    for ts in tau_steps:
        tau_key = round(ts * dt, 6)
        Xtc = snapshots[ts] - mu_f
        C = (Xtc.T @ X0c) / N
        Csym = 0.5 * (C + C.T)
        if return_top_eigvecs_all_lags:
            evals, evecs = np.linalg.eigh(Csym)
            order = np.argsort(evals)[::-1]
            eigs_by_tau[tau_key] = evals[order]
            topvecs_by_tau[tau_key] = evecs[:, order[:k]]
        else:
            eigs = np.sort(np.linalg.eigvalsh(Csym))[::-1]
            eigs_by_tau[tau_key] = eigs
        # rho(tau) = tr(Csym) / tr(C0)
        rho_by_tau[tau_key] = float(np.trace(Csym) / tr_C0)

    return {
        "eigs_by_tau": eigs_by_tau,
        "topvecs_by_tau": topvecs_by_tau,
        "rho_by_tau": rho_by_tau,
        "sigma_f": sigma_f,
        "sigma_f_sq": sigma_f_sq,
        "tr_C0": float(tr_C0),
        "gamma": d / N,
        "d": d, "N": N, "k": k,
    }


def count_curve(eigs_by_tau, sigma_f, gamma):
    taus, counts = [], []
    for tau, eigs in eigs_by_tau.items():
        if tau == 0.0:
            _, lp = edge_tau0(sigma_f, gamma)
        else:
            _, lp = fc_edges(tau, sigma=sigma_f, gamma=gamma)
        taus.append(tau)
        counts.append(int(np.sum(eigs > lp)))
    return np.array(taus), np.array(counts)


def auto_select_tau_star(taus, counts):
    plateau_count = int(counts.max())
    if plateau_count == 0:
        return float(taus[-1]), 0
    on_plateau = counts == plateau_count
    return float(taus[on_plateau].max()), plateau_count


def subspace_alignment(V_topk: np.ndarray, planted: np.ndarray) -> float:
    """Squared Frobenius norm of V^T planted, divided by k."""
    return float(np.linalg.norm(V_topk.T @ planted, ord="fro") ** 2 / planted.shape[1])


# ============================================================================
# Sweeps (always rotated; eigenvalue distributions invariant under Q)
# ============================================================================

def sweep_N(d, k, c, sigma_small, dt, tau_max, N_values, rotation, seed=0):
    out = []
    for i, N in enumerate(N_values):
        tau_steps = _build_tau_grid(dt, tau_max)
        res = simulate(d, k, c, sigma_small, N, dt, tau_steps,
                       seed=seed + i, rotation=rotation)
        taus, counts = count_curve(res["eigs_by_tau"], res["sigma_f"], res["gamma"])
        tau_star, m_hat = auto_select_tau_star(taus, counts)
        print(f"  N={N:>5d}  gamma={res['gamma']:.3f}  tau*={tau_star:.2f}  "
              f"detected count = {m_hat:>3d}  (truth k={k})")
        out.append((N, m_hat, tau_star))
    return out


def sweep_k(d, c, sigma_small, N, dt, tau_max, k_values, rotation_seed=999, seed=0):
    """Note: each k uses a fresh random rotation (different planted subspace)."""
    out = []
    d_eff = d
    for i, k in enumerate(k_values):
        Q = random_rotation(d_eff, seed=rotation_seed + i)
        tau_steps = _build_tau_grid(dt, tau_max)
        res = simulate(d_eff, k, c, sigma_small, N, dt, tau_steps,
                       seed=seed + i, rotation=Q)
        taus, counts = count_curve(res["eigs_by_tau"], res["sigma_f"], res["gamma"])
        tau_star, m_hat = auto_select_tau_star(taus, counts)
        print(f"  k_planted={k:>3d}  tau*={tau_star:.2f}  "
              f"detected count = {m_hat:>3d}")
        out.append((k, m_hat, tau_star))
    return out


# ============================================================================
# Main: 6-panel figure
# ============================================================================

def main(out_path: Path | None = None):
    d = 200
    k = 20
    c = 3.0
    sigma_small = 0.3
    N = 2000
    dt = 0.005
    tau_max = 10.0

    # Single random rotation Q for the baseline + sample-complexity sweep.
    # The k_planted sweep uses fresh Q's per k (since the planted subspace
    # dimension changes with k anyway).
    Q = random_rotation(d, seed=1234)
    Q_planted = Q[:, :k]   # d x k planted basis

    print("=" * 70)
    print("baseline run (random rotation Q)")
    print("=" * 70)
    tau_steps = _build_tau_grid(dt, tau_max)
    res = simulate(d, k, c, sigma_small, N, dt, tau_steps, seed=0,
                   rotation=Q, return_top_eigvecs_all_lags=True)
    taus, counts = count_curve(res["eigs_by_tau"], res["sigma_f"], res["gamma"])
    tau_star, m_hat = auto_select_tau_star(taus, counts)
    sigma_f = res["sigma_f"]
    gamma = res["gamma"]
    print(f"  tau*={tau_star:.2f}, detected count={m_hat}, planted k={k}")

    # Subspace alignment vs tau, for both DA (top-k of Csym(tau)) and PCA
    # (top-k of Csym(0), which is tau=0 sample covariance, i.e. constant in tau).
    pca_topk = res["topvecs_by_tau"][0.0]
    pca_align = subspace_alignment(pca_topk, Q_planted)
    da_align_by_tau = []
    for tau in taus:
        V = res["topvecs_by_tau"][float(tau)]
        da_align_by_tau.append(subspace_alignment(V, Q_planted))
    da_align_by_tau = np.array(da_align_by_tau)
    print(f"  subspace alignment: PCA={pca_align:.3f}, "
          f"DA@tau*={da_align_by_tau[-1]:.3f}, "
          f"random-baseline = k/d = {k/d:.3f}")

    print("\n" + "=" * 70)
    print("sweep over N")
    print("=" * 70)
    N_values = [200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
    sweep_N_out = sweep_N(d, k, c, sigma_small, dt, tau_max, N_values, Q, seed=10)

    print("\n" + "=" * 70)
    print("sweep over k_planted (fresh random rotation per k)")
    print("=" * 70)
    k_values = [5, 10, 20, 40, 80]
    sweep_k_out = sweep_k(d, c, sigma_small, N, dt, tau_max, k_values, seed=20)

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

    # (f) DA vs PCA subspace alignment with the planted basis Q[:, :k]
    ax = axes[1, 2]
    ax.plot(taus, da_align_by_tau, "o-", color="C0", lw=1.6, markersize=5,
            label="DA (top-k of $\\hat C_{\\rm sym}(\\tau)$)")
    ax.axhline(pca_align, color="C1", linestyle="-", lw=1.6,
               label=fr"PCA (top-k of $\hat C(0)$) $= {pca_align:.2f}$")
    ax.axhline(k / d, color="gray", linestyle=":", lw=1.2,
               label=fr"chance $= k/d = {k/d:.2f}$")
    ax.axhline(1.0, color="crimson", linestyle="--", lw=1.0, alpha=0.5,
               label="perfect $= 1$")
    ax.set_xscale("symlog", linthresh=max(2.0 * dt, 0.05))
    ax.set_xlabel(r"$\tau$ (lag, symlog)")
    ax.set_ylabel(r"subspace alignment with $Q[:,:k]$")
    ax.set_title("(f) DA recovers planted subspace; PCA stays at chance")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf"Synthetic planted-spike validation: $d={d}$, $k={k}$, "
        rf"$\gamma=d/N={gamma:.2f}$, $c={c}$, $\sigma_s={sigma_small}$, "
        rf"matched per-axis variance, planted basis $Q[:,:k]$ (random rotation)",
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

    # Save rho(tau) and auxiliary data
    rho_taus = np.array(sorted(res["rho_by_tau"].keys()))
    rho_vals = np.array([res["rho_by_tau"][t] for t in rho_taus])
    npz_path = out_path.with_suffix(".npz")
    np.savez(npz_path,
             taus=rho_taus,
             rho=rho_vals,
             d=d, N=N, k=k, c=c, sigma_small=sigma_small,
             dt=dt, tau_max=tau_max,
             sigma_f_sq=res["sigma_f_sq"],
             tr_C0=res["tr_C0"],
             gamma=gamma,
             tau_star=tau_star, m_hat=m_hat,
             counts=counts,
             da_align=da_align_by_tau,
             pca_align=pca_align)
    print(f"saved data to: {npz_path}")
    print(f"  rho range: [{rho_vals[1:].min():.4e}, {rho_vals[1:].max():.4e}] "
          f"(excluding tau=0)")


if __name__ == "__main__":
    main()
