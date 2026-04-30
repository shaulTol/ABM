"""
Variants of the planted-spike experiment with mixed axis types:
  - "sep" axes: double-well (separation c, width sigma_small) — metastable
  - "var" axes: wide unimodal Gaussian (variance sigma_big^2) — slow but not metastable
  - background axes: isotropic Gaussian (variance sigma_iso^2 = c^2 + sigma_small^2)

Three configs:
  (1) k_sep=20, k_var=0   — already done by simulate_planted_spikes.py
  (2) k_sep=0,  k_var=20  — pure variance, no metastability
  (3) k_sep=10, k_var=10  — mixed

Run:
    python3 simulate_planted_spikes_variants.py
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
# Generalized score / sampling for k_sep double-well + k_var wide-Gaussian
# Axis layout: [0:k_sep] = double-well, [k_sep:k_sep+k_var] = wide, [rest] = iso
# ============================================================================

def _score_mixed(X, c, sigma_small, sigma_iso, sigma_big, k_sep, k_var):
    out = np.empty_like(X)
    s2 = sigma_small ** 2

    # Double-well axes
    if k_sep > 0:
        Z = X[:, :k_sep]
        lp1 = -((Z - c) ** 2) / (2 * s2)
        lp2 = -((Z + c) ** 2) / (2 * s2)
        m = np.maximum(lp1, lp2)
        p1 = np.exp(lp1 - m)
        p2 = np.exp(lp2 - m)
        norm = p1 + p2
        p1 /= norm; p2 /= norm
        out[:, :k_sep] = -((Z - c) * p1 + (Z + c) * p2) / s2

    # Wide unimodal axes
    if k_var > 0:
        i0, i1 = k_sep, k_sep + k_var
        out[:, i0:i1] = -X[:, i0:i1] / (sigma_big ** 2)

    # Isotropic background
    out[:, k_sep + k_var:] = -X[:, k_sep + k_var:] / (sigma_iso ** 2)
    return out


def _sample_mixed(N, d, k_sep, k_var, c, sigma_small, sigma_iso, sigma_big, rng):
    X = np.empty((N, d))
    # Double-well axes
    if k_sep > 0:
        signs = rng.choice([-1.0, 1.0], size=(N, k_sep))
        X[:, :k_sep] = signs * c + rng.normal(0.0, sigma_small, size=(N, k_sep))
    # Wide unimodal axes
    if k_var > 0:
        i0, i1 = k_sep, k_sep + k_var
        X[:, i0:i1] = rng.normal(0.0, sigma_big, size=(N, k_var))
    # Isotropic background
    X[:, k_sep + k_var:] = rng.normal(0.0, sigma_iso, size=(N, d - k_sep - k_var))
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


def simulate_mixed(
    d, k_sep, k_var, c, sigma_small, sigma_big, N, dt, tau_steps, seed,
    rotation: np.ndarray,
    return_top_eigvecs_all_lags: bool = False,
    empirical_sigma: bool = False,
):
    k_total = k_sep + k_var
    sigma_iso = float(np.sqrt(c ** 2 + sigma_small ** 2))
    sigma_iso_sq = c ** 2 + sigma_small ** 2

    rng = np.random.default_rng(seed)
    X_fact = _sample_mixed(N, d, k_sep, k_var, c, sigma_small, sigma_iso, sigma_big, rng)
    X = X_fact @ rotation.T

    if empirical_sigma:
        # Estimate sigma_f from finite sample, like SDXL/alanine
        C0_emp = (X.T @ X) / N  # X already zero-mean by construction
        sigma_f_sq = float(np.trace(C0_emp) / d)
    else:
        # Analytic sigma_f from known distribution
        sigma_f_sq = (k_sep * sigma_iso_sq + k_var * sigma_big**2
                      + (d - k_total) * sigma_iso_sq) / d
    sigma_f = float(np.sqrt(sigma_f_sq))

    snapshots: dict[int, np.ndarray] = {0: X.copy()}
    n_steps = max(tau_steps)
    sqrt_dt = float(np.sqrt(dt))
    drift_pref = 0.5 * sigma_f_sq

    for t in range(1, n_steps + 1):
        X_in_fact = X @ rotation
        grad_fact = _score_mixed(X_in_fact, c, sigma_small, sigma_iso, sigma_big, k_sep, k_var)
        grad = grad_fact @ rotation.T
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
            topvecs_by_tau[tau_key] = evecs[:, order[:k_total]]
        else:
            eigs = np.sort(np.linalg.eigvalsh(Csym))[::-1]
            eigs_by_tau[tau_key] = eigs
        rho_by_tau[tau_key] = float(np.trace(Csym) / tr_C0)

    return {
        "eigs_by_tau": eigs_by_tau,
        "topvecs_by_tau": topvecs_by_tau,
        "rho_by_tau": rho_by_tau,
        "sigma_f": sigma_f,
        "sigma_f_sq": sigma_f_sq,
        "tr_C0": float(tr_C0),
        "gamma": d / N,
        "d": d, "N": N, "k_sep": k_sep, "k_var": k_var,
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


def subspace_alignment(V_topk, planted):
    return float(np.linalg.norm(V_topk.T @ planted, ord="fro") ** 2 / planted.shape[1])


def run_config(label, d, k_sep, k_var, c, sigma_small, sigma_big, N, dt, tau_max,
               seed=0, empirical_sigma=False):
    """Run one config, return results dict."""
    k_total = k_sep + k_var
    Q = random_rotation(d, seed=1234)
    Q_planted = Q[:, :k_total]

    print(f"\n{'='*70}")
    print(f"Config: {label}  (k_sep={k_sep}, k_var={k_var}, sigma_big={sigma_big:.1f})")
    print(f"{'='*70}")

    tau_steps = _build_tau_grid(dt, tau_max)
    res = simulate_mixed(d, k_sep, k_var, c, sigma_small, sigma_big, N, dt, tau_steps,
                         seed=seed, rotation=Q, return_top_eigvecs_all_lags=True,
                         empirical_sigma=empirical_sigma)
    taus, counts = count_curve(res["eigs_by_tau"], res["sigma_f"], res["gamma"])

    # Subspace alignment
    pca_topk = res["topvecs_by_tau"][0.0]
    pca_align = subspace_alignment(pca_topk, Q_planted)
    da_align = []
    for tau in taus:
        V = res["topvecs_by_tau"][float(tau)]
        da_align.append(subspace_alignment(V, Q_planted))
    da_align = np.array(da_align)

    # rho and eigenvalues
    rho_taus = np.array(sorted(res["rho_by_tau"].keys()))
    rho_vals = np.array([res["rho_by_tau"][t] for t in rho_taus])
    eigs_by_tau = res["eigs_by_tau"]
    lam1_vals = np.array([eigs_by_tau[t][0] for t in rho_taus])

    # Save top-m eigenvalues at each lag (m = k_total = k_sep + k_var)
    topk_eigs = np.array([eigs_by_tau[t][:k_total] for t in rho_taus])  # (n_lags, k_total)

    # Auto tau*
    plateau_count = int(counts.max())
    on_plateau = counts == plateau_count
    tau_star = float(taus[on_plateau].max()) if plateau_count > 0 else float(taus[-1])

    print(f"  tau*={tau_star:.2f}, detected={plateau_count}, "
          f"PCA align={pca_align:.3f}, DA@last={da_align[-1]:.3f}")
    print(f"  rho range (excl tau=0): [{rho_vals[1:].min():.4e}, {rho_vals[1:].max():.4e}]")

    return {
        "label": label,
        "taus": taus, "counts": counts,
        "rho_taus": rho_taus, "rho_vals": rho_vals, "lam1": lam1_vals,
        "topk_eigs": topk_eigs,  # (n_lags, k_total)
        "da_align": da_align, "pca_align": pca_align,
        "res": res, "tau_star": tau_star,
        "k_sep": k_sep, "k_var": k_var, "k_total": k_total,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau_max", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--out_suffix", type=str, default="")
    parser.add_argument("--var_factors", type=str, default="2.0",
                        help="Comma-separated variance factors (sigma_big^2 / sigma_iso^2)")
    parser.add_argument("--empirical_sigma", action="store_true",
                        help="Estimate sigma_f from sample (like SDXL/alanine)")
    args = parser.parse_args()

    d = 200
    c = 3.0
    sigma_small = 0.3
    sigma_iso = float(np.sqrt(c**2 + sigma_small**2))
    N = 2000
    dt = args.dt
    tau_max = args.tau_max
    var_factors = [float(x) for x in args.var_factors.split(",")]

    suffix = args.out_suffix if args.out_suffix else ""
    npz_path = Path(__file__).parent / f"fig_planted_spikes_variants{suffix}.npz"

    all_results = []  # list of (var_factor, results_list)
    for vf in var_factors:
        sigma_big = np.sqrt(vf) * sigma_iso  # vf × variance
        print(f"\n{'#'*70}")
        print(f"# Variance factor = {vf}x  (sigma_big = {sigma_big:.2f})")
        print(f"{'#'*70}")

        configs = [
            ("20 sep",          20, 0,  0),
            ("20 var",          0,  20, 1),
            ("10 sep + 10 var", 10, 10, 2),
        ]

        results = []
        for label, k_sep, k_var, seed_offset in configs:
            r = run_config(label, d, k_sep, k_var, c, sigma_small, sigma_big,
                           N, dt, tau_max, seed=seed_offset,
                           empirical_sigma=args.empirical_sigma)
            results.append(r)
        all_results.append((vf, results))

        # Incremental save after each variance factor
        save_dict = {"d": d, "N": N, "sigma_iso": sigma_iso,
                     "var_factors": np.array(var_factors)}
        for vi, (vf2, res_list) in enumerate(all_results):
            for ci, r in enumerate(res_list):
                pfx = f"vf{vi}_cfg{ci}_"
                save_dict[pfx + "label"] = r["label"]
                save_dict[pfx + "taus"] = r["taus"]
                save_dict[pfx + "counts"] = r["counts"]
                save_dict[pfx + "rho_taus"] = r["rho_taus"]
                save_dict[pfx + "rho"] = r["rho_vals"]
                save_dict[pfx + "lam1"] = r["lam1"]
                save_dict[pfx + "topk_eigs"] = r["topk_eigs"]
                save_dict[pfx + "da_align"] = r["da_align"]
                save_dict[pfx + "pca_align"] = r["pca_align"]
                save_dict[pfx + "k_sep"] = r["k_sep"]
                save_dict[pfx + "k_var"] = r["k_var"]
        save_dict["n_completed_factors"] = len(all_results)
        np.savez(npz_path, **save_dict)
        print(f"\n  [incremental save] {len(all_results)}/{len(var_factors)} factors → {npz_path}")

    print(f"\nAll done. Final save: {npz_path}")


if __name__ == "__main__":
    main()
