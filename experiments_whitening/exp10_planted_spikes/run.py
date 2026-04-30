#!/usr/bin/env python3
"""
Exp 10: Planted Spikes — whitened variant.

Replicates experiments/synthetic_spikes/simulate_planted_spikes.py but with
whitening applied before Langevin simulation.

For the planted-spike model:
  - k metastable axes: variance = c^2 + sigma_small^2  (double-well)
  - (d-k) isotropic axes: variance = sigma_iso^2 = c^2 + sigma_small^2
  (matched by design: all axes have equal marginal variance)

Since sigma_iso = sqrt(c^2 + sigma_small^2) = sigma_f, the covariance is
Sigma_f = sigma_f^2 * I_d (isotropic). So whitening is just W = (1/sigma_f)*I
and the whitened diffusion has var_f = 1.

The score in whitened coordinates:
  score_tilde(x_tilde) = W_inv^T @ score(W_inv @ x_tilde)
  = sigma_f * score(sigma_f * x_tilde)  (since W = (1/sigma_f)*I, mu=0)
"""
import sys, os, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "noise_floor"))

from free_conv_utils import edges as fc_edges, density as fc_density, edge_tau0
from experiments_whitening.shared.whitening import WhiteningTransform


# ── Score function (factorized frame, same as original) ──
def _score_factorized(X, c, sigma_small, sigma_iso, k):
    s2 = sigma_small ** 2
    Z = X[:, :k]
    lp1 = -((Z - c) ** 2) / (2 * s2)
    lp2 = -((Z + c) ** 2) / (2 * s2)
    m = np.maximum(lp1, lp2)
    p1 = np.exp(lp1 - m)
    p2 = np.exp(lp2 - m)
    norm = p1 + p2
    p1 /= norm; p2 /= norm
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


def simulate_whitened(
    d, k, c, sigma_small, N, dt, tau_steps, seed,
    rotation,
    return_top_eigvecs_all_lags=False,
):
    """Run planted-spike simulation with whitening."""
    sigma_iso = float(np.sqrt(c**2 + sigma_small**2))
    sigma_f_sq = c**2 + sigma_small**2
    sigma_f = float(np.sqrt(sigma_f_sq))

    # Analytic Sigma_f = sigma_f^2 * I (isotropic by matched-variance design)
    Sigma_f = sigma_f_sq * np.eye(d)
    mu_f = np.zeros(d)
    wt = WhiteningTransform(analytic_cov=Sigma_f, mu=mu_f)
    # W = (1/sigma_f) * I, so whitening just scales by 1/sigma_f

    rng = np.random.default_rng(seed)
    X_fact = _sample_initial(N, d, k, c, sigma_small, sigma_iso, rng)
    X = X_fact @ rotation.T  # rotate to hide axes

    # Whiten: X_tilde = X / sigma_f (since W = I/sigma_f and mu=0)
    X_tilde = wt.whiten(X)

    snapshots = {0: X_tilde.copy()}
    n_steps = max(tau_steps)
    sqrt_dt = float(np.sqrt(dt))

    # In whitened coords: var_f_tilde ≈ 1
    # drift = 0.5 * var_f_tilde * score_tilde(x_tilde) * dt
    # score_tilde(x_tilde) = W_inv^T @ score(W_inv @ x_tilde)
    # For isotropic whitening: score_tilde(x_tilde) = sigma_f * score(sigma_f * x_tilde)
    var_f_tilde = wt.sigma_tilde_sq  # should be 1.0
    drift_pref = 0.5 * var_f_tilde

    for t in range(1, n_steps + 1):
        # Unwhiten to original frame
        X_orig = wt.unwhiten(X_tilde)
        # Rotate to factorized frame for score
        X_in_fact = X_orig @ rotation
        grad_fact = _score_factorized(X_in_fact, c, sigma_small, sigma_iso, k)
        grad_orig = grad_fact @ rotation.T
        # Transform gradient to whitened space: grad_tilde = grad_orig @ W_inv
        grad_tilde = grad_orig @ wt.W_inv

        Z = rng.standard_normal((N, d))
        sigma_tilde = float(np.sqrt(var_f_tilde))
        X_tilde = X_tilde + drift_pref * grad_tilde * dt + sigma_tilde * sqrt_dt * Z

        if t in tau_steps:
            snapshots[t] = X_tilde.copy()

    # Compute eigenvalue spectra (in whitened coordinates)
    mu_tilde = np.zeros(d)
    X0c = snapshots[0] - mu_tilde
    C0 = (X0c.T @ X0c) / N
    tr_C0 = np.trace(C0)

    eigs_by_tau = {}
    topvecs_by_tau = {}
    rho_by_tau = {}
    for ts in tau_steps:
        tau_key = round(ts * dt, 6)
        Xtc = snapshots[ts] - mu_tilde
        C = (Xtc.T @ X0c) / N
        Csym = 0.5 * (C + C.T)
        if return_top_eigvecs_all_lags:
            evals, evecs = np.linalg.eigh(Csym)
            order = np.argsort(evals)[::-1].copy()
            eigs_by_tau[tau_key] = evals[order]
            topvecs_by_tau[tau_key] = evecs[:, order[:k]]
        else:
            eigs = np.sort(np.linalg.eigvalsh(Csym))[::-1].copy()
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
        "d": d, "N": N, "k": k,
        "whitening_diagnostics": wt.diagnostics(),
    }


def count_curve(eigs_by_tau, sigma_f, gamma):
    """Count eigenvalues above free-conv upper edge at each lag."""
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


def subspace_alignment(V_topk, planted):
    return float(np.linalg.norm(V_topk.T @ planted, ord="fro")**2 / planted.shape[1])


def main():
    d = 200
    k = 20
    c = 3.0
    sigma_small = 0.3
    N = 2000
    dt = 0.005
    tau_max = 10.0

    Q = random_rotation(d, seed=1234)
    Q_planted = Q[:, :k]

    print("=" * 70)
    print("WHITENED planted-spike baseline (random rotation Q)")
    print("=" * 70)

    tau_steps = _build_tau_grid(dt, tau_max)
    res = simulate_whitened(d, k, c, sigma_small, N, dt, tau_steps, seed=0,
                            rotation=Q, return_top_eigvecs_all_lags=True)
    taus, counts = count_curve(res["eigs_by_tau"], res["sigma_f"], res["gamma"])
    tau_star, m_hat = auto_select_tau_star(taus, counts)
    print(f"  tau*={tau_star:.2f}, detected count={m_hat}, planted k={k}")

    # Subspace alignment
    pca_topk = res["topvecs_by_tau"][0.0]
    pca_align = subspace_alignment(pca_topk, Q_planted)
    da_align_by_tau = []
    for tau in taus:
        V = res["topvecs_by_tau"][float(tau)]
        da_align_by_tau.append(subspace_alignment(V, Q_planted))
    da_align_by_tau = np.array(da_align_by_tau)
    print(f"  PCA alignment={pca_align:.3f}, DA@tau*={da_align_by_tau[-1]:.3f}, "
          f"chance=k/d={k/d:.3f}")

    # Also run original for comparison
    print("\n" + "=" * 70)
    print("ORIGINAL planted-spike baseline (for comparison)")
    print("=" * 70)
    sys.path.insert(0, str(ROOT / "experiments" / "synthetic_spikes"))
    from simulate_planted_spikes import simulate as simulate_orig, count_curve as cc_orig, \
        auto_select_tau_star as ast_orig, subspace_alignment as sa_orig

    res_orig = simulate_orig(d, k, c, sigma_small, N, dt, tau_steps, seed=0,
                             rotation=Q, return_top_eigvecs_all_lags=True)
    taus_orig, counts_orig = cc_orig(res_orig["eigs_by_tau"], res_orig["sigma_f"], res_orig["gamma"])
    tau_star_orig, m_hat_orig = ast_orig(taus_orig, counts_orig)
    print(f"  tau*={tau_star_orig:.2f}, detected count={m_hat_orig}, planted k={k}")

    pca_align_orig = sa_orig(res_orig["topvecs_by_tau"][0.0], Q_planted)
    da_align_orig = []
    for tau in taus_orig:
        da_align_orig.append(sa_orig(res_orig["topvecs_by_tau"][float(tau)], Q_planted))
    da_align_orig = np.array(da_align_orig)
    print(f"  PCA alignment={pca_align_orig:.3f}, DA@tau*={da_align_orig[-1]:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Whitened: m_hat={m_hat}, DA@tau* alignment={da_align_by_tau[-1]:.3f}")
    print(f"Original: m_hat={m_hat_orig}, DA@tau* alignment={da_align_orig[-1]:.3f}")

    # Save results
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)

    results = {
        'whitened': {
            'tau_star': tau_star,
            'm_hat': m_hat,
            'pca_align': pca_align,
            'da_align_tau_star': float(da_align_by_tau[-1]),
            'diagnostics': res['whitening_diagnostics'],
        },
        'original': {
            'tau_star': tau_star_orig,
            'm_hat': m_hat_orig,
            'pca_align': pca_align_orig,
            'da_align_tau_star': float(da_align_orig[-1]),
        },
        'params': {
            'd': d, 'k': k, 'c': c, 'sigma_small': sigma_small,
            'N': N, 'dt': dt, 'tau_max': tau_max,
        },
    }
    out_json = os.path.join(out_dir, 'planted_spikes_whitened.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Save rho and alignment curves
    # Extract rho(tau) arrays
    rho_taus_w = np.array(sorted(res["rho_by_tau"].keys()))
    rho_vals_w = np.array([res["rho_by_tau"][t] for t in rho_taus_w])
    rho_taus_o = np.array(sorted(res_orig["rho_by_tau"].keys()))
    rho_vals_o = np.array([res_orig["rho_by_tau"][t] for t in rho_taus_o])

    # Extract lambda1(tau) / lambda1(0) for both
    lam1_0_w = res["eigs_by_tau"][0.0][0]
    lam1_w = np.array([res["eigs_by_tau"][t][0] / lam1_0_w for t in rho_taus_w])
    lam1_0_o = res_orig["eigs_by_tau"][0.0][0]
    lam1_o = np.array([res_orig["eigs_by_tau"][t][0] / lam1_0_o for t in rho_taus_o])

    np.savez(
        os.path.join(out_dir, 'planted_spikes_whitened.npz'),
        taus_whitened=taus, counts_whitened=counts,
        da_align_whitened=da_align_by_tau,
        taus_original=taus_orig, counts_original=counts_orig,
        da_align_original=da_align_orig,
        pca_align_whitened=pca_align,
        pca_align_original=pca_align_orig,
        rho_taus_whitened=rho_taus_w, rho_vals_whitened=rho_vals_w,
        rho_taus_original=rho_taus_o, rho_vals_original=rho_vals_o,
        lam1_ratio_whitened=lam1_w, lam1_ratio_original=lam1_o,
    )


if __name__ == '__main__':
    main()
