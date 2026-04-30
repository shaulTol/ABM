#!/usr/bin/env python3
"""
Exp 10 variants: Planted Spikes with mixed axis types — whitened.

Three configs (matching simulate_planted_spikes_variants.py):
  (1) k_sep=20, k_var=0   — isotropic Sigma_f → whitening = no-op
  (2) k_sep=0,  k_var=20  — anisotropic (var axes have sigma_big > sigma_iso)
  (3) k_sep=10, k_var=10  — mixed anisotropic

For each config:
  - Compute analytic Sigma_f (diagonal in factorized basis, rotated)
  - WhiteningTransform → W, W_inv
  - Run Langevin in whitened coordinates
  - Compute eigenvalue spectra + spike counts + rho + alignment

Also runs original (unwhitened) for comparison.
"""
import sys, os, json, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "noise_floor"))
sys.path.insert(0, str(ROOT / "experiments" / "synthetic_spikes"))

from free_conv_utils import edges as fc_edges, edge_tau0
from experiments_whitening.shared.whitening import WhiteningTransform


# ── Score (factorized frame, mixed axes) ──
def _score_mixed(X, c, sigma_small, sigma_iso, sigma_big, k_sep, k_var):
    out = np.empty_like(X)
    s2 = sigma_small ** 2
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
    if k_var > 0:
        i0, i1 = k_sep, k_sep + k_var
        out[:, i0:i1] = -X[:, i0:i1] / (sigma_big ** 2)
    out[:, k_sep + k_var:] = -X[:, k_sep + k_var:] / (sigma_iso ** 2)
    return out


def _sample_mixed(N, d, k_sep, k_var, c, sigma_small, sigma_iso, sigma_big, rng):
    X = np.empty((N, d))
    if k_sep > 0:
        signs = rng.choice([-1.0, 1.0], size=(N, k_sep))
        X[:, :k_sep] = signs * c + rng.normal(0.0, sigma_small, size=(N, k_sep))
    if k_var > 0:
        i0, i1 = k_sep, k_sep + k_var
        X[:, i0:i1] = rng.normal(0.0, sigma_big, size=(N, k_var))
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


def _build_analytic_cov(d, k_sep, k_var, c, sigma_small, sigma_big, sigma_iso, Q):
    """Build analytic Sigma_f = Q diag(variances) Q^T."""
    variances = np.full(d, sigma_iso ** 2)
    variances[:k_sep] = c ** 2 + sigma_small ** 2  # = sigma_iso^2
    if k_var > 0:
        variances[k_sep:k_sep + k_var] = sigma_big ** 2
    # In rotated frame: Sigma_f = Q @ diag(variances) @ Q^T
    Sigma_f = Q @ np.diag(variances) @ Q.T
    return Sigma_f


def simulate_config(d, k_sep, k_var, c, sigma_small, sigma_big, N, dt, tau_steps,
                    seed, rotation, whitened=False):
    """Run one config, optionally in whitened coordinates."""
    k_total = k_sep + k_var
    sigma_iso = float(np.sqrt(c ** 2 + sigma_small ** 2))

    if whitened:
        Sigma_f = _build_analytic_cov(d, k_sep, k_var, c, sigma_small,
                                       sigma_big, sigma_iso, rotation)
        mu_f = np.zeros(d)
        wt = WhiteningTransform(analytic_cov=Sigma_f, mu=mu_f)
    else:
        wt = None

    rng = np.random.default_rng(seed)
    X_fact = _sample_mixed(N, d, k_sep, k_var, c, sigma_small, sigma_iso, sigma_big, rng)
    X = X_fact @ rotation.T

    if whitened:
        X = wt.whiten(X)
        var_f_tilde = wt.sigma_tilde_sq
    else:
        # Analytic sigma_f_sq
        sigma_f_sq = (k_sep * (c**2 + sigma_small**2) + k_var * sigma_big**2
                      + (d - k_total) * sigma_iso**2) / d
        var_f_tilde = sigma_f_sq

    sigma_f_tilde = float(np.sqrt(var_f_tilde))
    drift_pref = 0.5 * var_f_tilde

    snapshots = {0: X.copy()}
    n_steps = max(tau_steps)
    sqrt_dt = float(np.sqrt(dt))

    for t in range(1, n_steps + 1):
        if whitened:
            X_orig = wt.unwhiten(X)
            X_in_fact = X_orig @ rotation
            grad_fact = _score_mixed(X_in_fact, c, sigma_small, sigma_iso,
                                     sigma_big, k_sep, k_var)
            grad_orig = grad_fact @ rotation.T
            grad = grad_orig @ wt.W_inv
        else:
            X_in_fact = X @ rotation
            grad_fact = _score_mixed(X_in_fact, c, sigma_small, sigma_iso,
                                     sigma_big, k_sep, k_var)
            grad = grad_fact @ rotation.T

        Z = rng.standard_normal((N, d))
        X = X + drift_pref * grad * dt + sigma_f_tilde * sqrt_dt * Z
        if t in tau_steps:
            snapshots[t] = X.copy()

    # Eigenvalue spectra
    mu = np.zeros(d)
    X0c = snapshots[0] - mu
    C0 = (X0c.T @ X0c) / N
    tr_C0 = np.trace(C0)

    eigs_by_tau = {}
    topvecs_by_tau = {}
    rho_by_tau = {}
    for ts in tau_steps:
        tau_key = round(ts * dt, 6)
        Xtc = snapshots[ts] - mu
        C = (Xtc.T @ X0c) / N
        Csym = 0.5 * (C + C.T)
        evals, evecs = np.linalg.eigh(Csym)
        order = np.argsort(evals)[::-1].copy()
        eigs_by_tau[tau_key] = evals[order]
        topvecs_by_tau[tau_key] = evecs[:, order[:k_total]]
        rho_by_tau[tau_key] = float(np.trace(Csym) / tr_C0)

    return {
        "eigs_by_tau": eigs_by_tau,
        "topvecs_by_tau": topvecs_by_tau,
        "rho_by_tau": rho_by_tau,
        "sigma_f_tilde": sigma_f_tilde,
        "tr_C0": float(tr_C0),
        "gamma": d / N,
        "d": d, "N": N,
        "whitening_diagnostics": wt.diagnostics() if wt else None,
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


def run_one_config(label, d, k_sep, k_var, c, sigma_small, sigma_big,
                   N, dt, tau_max, seed, Q):
    """Run both whitened and original, return comparison dict."""
    k_total = k_sep + k_var
    Q_planted = Q[:, :k_total]
    sigma_iso = float(np.sqrt(c ** 2 + sigma_small ** 2))
    tau_steps = _build_tau_grid(dt, tau_max)

    print(f"\n{'='*70}")
    print(f"Config: {label}  (k_sep={k_sep}, k_var={k_var}, "
          f"sigma_big={sigma_big:.2f}, sigma_iso={sigma_iso:.2f})")
    print(f"{'='*70}")

    results = {}
    for mode in ['original', 'whitened']:
        is_w = (mode == 'whitened')
        res = simulate_config(d, k_sep, k_var, c, sigma_small, sigma_big,
                              N, dt, tau_steps, seed, Q, whitened=is_w)

        taus, counts = count_curve(res["eigs_by_tau"], res["sigma_f_tilde"], res["gamma"])

        pca_align = subspace_alignment(res["topvecs_by_tau"][0.0], Q_planted)
        da_align = np.array([subspace_alignment(res["topvecs_by_tau"][float(t)], Q_planted)
                             for t in taus])

        rho_taus = np.array(sorted(res["rho_by_tau"].keys()))
        rho_vals = np.array([res["rho_by_tau"][t] for t in rho_taus])

        plateau_count = int(counts.max())
        on_plateau = counts == plateau_count
        tau_star = float(taus[on_plateau].max()) if plateau_count > 0 else float(taus[-1])

        print(f"  [{mode}] tau*={tau_star:.2f}, detected={plateau_count}, "
              f"PCA={pca_align:.3f}, DA@last={da_align[-1]:.3f}, "
              f"sigma_f_tilde={res['sigma_f_tilde']:.3f}")

        results[mode] = {
            'taus': taus, 'counts': counts,
            'rho_taus': rho_taus, 'rho_vals': rho_vals,
            'da_align': da_align, 'pca_align': pca_align,
            'tau_star': tau_star, 'detected': plateau_count,
            'diagnostics': res.get('whitening_diagnostics'),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau_max", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--var_factor", type=float, default=2.0)
    args = parser.parse_args()

    d = 200
    c = 3.0
    sigma_small = 0.3
    sigma_iso = float(np.sqrt(c**2 + sigma_small**2))
    sigma_big = np.sqrt(args.var_factor) * sigma_iso
    N = 2000
    dt = args.dt
    tau_max = args.tau_max

    Q = random_rotation(d, seed=1234)

    configs = [
        ("20 sep",          20, 0,  0),
        ("20 var",          0,  20, 1),
        ("10 sep + 10 var", 10, 10, 2),
    ]

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)

    save_dict = {"d": d, "N": N, "sigma_iso": sigma_iso, "sigma_big": sigma_big,
                 "var_factor": args.var_factor, "dt": dt, "tau_max": tau_max}
    summary = {}

    for ci, (label, k_sep, k_var, seed_offset) in enumerate(configs):
        results = run_one_config(label, d, k_sep, k_var, c, sigma_small, sigma_big,
                                 N, dt, tau_max, seed_offset, Q)

        for mode in ['original', 'whitened']:
            r = results[mode]
            pfx = f"cfg{ci}_{mode}_"
            save_dict[pfx + "taus"] = r['taus']
            save_dict[pfx + "counts"] = r['counts']
            save_dict[pfx + "rho_taus"] = r['rho_taus']
            save_dict[pfx + "rho_vals"] = r['rho_vals']
            save_dict[pfx + "da_align"] = r['da_align']
            save_dict[pfx + "pca_align"] = r['pca_align']

        save_dict[f"cfg{ci}_label"] = label
        save_dict[f"cfg{ci}_k_sep"] = k_sep
        save_dict[f"cfg{ci}_k_var"] = k_var

        summary[label] = {
            'orig_detected': results['original']['detected'],
            'whit_detected': results['whitened']['detected'],
            'orig_da_last': float(results['original']['da_align'][-1]),
            'whit_da_last': float(results['whitened']['da_align'][-1]),
        }

    npz_path = os.path.join(out_dir, 'planted_spikes_variants_whitened.npz')
    np.savez(npz_path, **save_dict)
    print(f"\nSaved: {npz_path}")

    json_path = os.path.join(out_dir, 'planted_spikes_variants_whitened.json')
    with open(json_path, 'w') as f:
        json.dump({
            'params': {'d': d, 'N': N, 'c': c, 'sigma_small': sigma_small,
                       'sigma_big': float(sigma_big), 'sigma_iso': float(sigma_iso),
                       'var_factor': args.var_factor, 'dt': dt, 'tau_max': tau_max},
            'summary': summary,
        }, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == '__main__':
    main()
