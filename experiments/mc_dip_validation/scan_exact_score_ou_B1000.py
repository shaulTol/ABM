#!/usr/bin/env python3
"""
B=1000 exact-score OU null for tighter Bonferroni p-values.

Same pipeline as scan_exact_score_ou.py Part B, but 1000 reps instead of 100.
Skips Part A (decay scan already validated).

Reports per-direction p-values and m_hat = max{m : DA1..DAm all pass at alpha/m}.
"""
import numpy as np
import torch
import sys, os, time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.simulation.generic_langevin import GenericLangevinSimulator
from src.metrics.da_directions import extract_da_directions


class ExactGaussianScore:
    """Exact score for N(0, sigma_f^2 * I_d): grad log f(x) = -x / sigma_f^2."""
    def __init__(self, sigma_f_sq):
        self.sigma_f_sq = sigma_f_sq

    def __call__(self, x):
        return -x / self.sigma_f_sq


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load sigma_f^2 from canonical file
    ctau_path = os.path.join(project_root,
        'experiments/save_autocovariance/results/alanine_C_tau_full_h_half_10k.npz')
    ctau_data = np.load(ctau_path)
    C0_data = np.array(ctau_data['C0'], dtype=np.float64)
    d = int(ctau_data['d'])
    sigma_f_sq = np.trace(C0_data) / d
    sigma_f = np.sqrt(sigma_f_sq)

    # Load observed eigenvalues from canonical file
    C_tau = ctau_data['C_tau'][-1]  # last lag = 10000ps
    C_sym = 0.5 * (C_tau + C_tau.T).astype(np.float64)
    obs_evals_all, _ = np.linalg.eigh(C_sym)
    obs_idx = np.argsort(obs_evals_all)[::-1].copy()
    obs_evals = obs_evals_all[obs_idx]

    N = 2000
    dt = 0.1
    B = 1000
    n_eigvals = 10
    lag = 100000

    print(f"d={d}, N={N}, dt={dt}, B={B}, lag={lag} (phys={lag*dt:.0f} ps)")
    print(f"sigma_f_sq = {sigma_f_sq:.6e}, sigma_f = {sigma_f:.6e}")
    print(f"\nObserved eigenvalues:")
    for i in range(5):
        print(f"  DA{i+1}: lambda = {obs_evals[i]:.6e}")

    # Build exact score
    exact_score = ExactGaussianScore(sigma_f_sq)

    # --- MC null: B=1000 reps ---
    print(f"\n=== MC null with exact score, B={B}, lag={lag} ===")

    null_evals_all = np.zeros((B, n_eigvals))
    seed_offset = 70000  # different from B=100 run (which used 60000)

    t_start = time.time()
    for rep in range(B):
        seed = seed_offset + rep
        t0 = time.time()

        # Fresh isotropic samples each rep
        rng = np.random.RandomState(seed)
        x_null_np = sigma_f * rng.randn(N, d)
        x_null = torch.from_numpy(x_null_np).float().to(device)

        # New simulator with exact score, fresh samples
        sim_null = GenericLangevinSimulator(
            score_fn=exact_score, samples=x_null, dt=dt, device=device,
            var_f=sigma_f_sq)

        evals_b, _, _ = extract_da_directions(
            simulator=sim_null, lag=lag, n_trajectories=N,
            dt=dt, device=device, seed=seed + 200000)

        evals_np = evals_b.cpu().numpy()
        null_evals_all[rep, :] = evals_np[:n_eigvals]

        elapsed = time.time() - t0
        if rep < 5 or rep % 100 == 99:
            print(f"  Rep {rep+1:>4d}/{B}: lam1={evals_np[0]:.6e}, lam2={evals_np[1]:.6e}, "
                  f"gap={evals_np[0]/max(evals_np[1],1e-15):.2f}, time={elapsed:.0f}s")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # --- Results: null eigenvalue distribution ---
    print(f"\n=== Null eigenvalue distribution (B={B}) ===")
    print(f"  {'k':>4} {'median':>12} {'mean':>12} {'2.5th':>12} {'97.5th':>12} {'min':>12} {'max':>12}")
    print("  " + "-" * 80)

    for k in range(min(n_eigvals, 5)):
        vals = null_evals_all[:, k]
        print(f"  {k+1:>4d} {np.median(vals):>12.4e} {np.mean(vals):>12.4e} "
              f"{np.percentile(vals, 2.5):>12.4e} {np.percentile(vals, 97.5):>12.4e} "
              f"{np.min(vals):>12.4e} {np.max(vals):>12.4e}")

    # --- Per-direction p-values and Bonferroni ---
    print(f"\n=== Per-direction Bonferroni table (B={B}) ===")
    print(f"  {'i':>3} {'lam_obs':>12} {'null_med':>12} {'null_max':>12} {'p_emp':>10} {'alpha/i':>8} {'pass?':>6}")
    print("  " + "-" * 65)

    m_hat = 0
    for i in range(5):
        null_i = null_evals_all[:, i]
        p_emp = (1 + np.sum(null_i >= obs_evals[i])) / (1 + B)
        alpha_i = 0.05 / (i + 1)
        passes = p_emp <= alpha_i
        tag = "YES" if passes else "NO"
        print(f"  {i+1:>3d} {obs_evals[i]:>12.4e} {np.median(null_i):>12.4e} "
              f"{np.max(null_i):>12.4e} {p_emp:>10.6f} {alpha_i:>8.4f} {tag:>6}")
        if passes and i == m_hat:
            m_hat = i + 1

    print(f"\n  m_hat = {m_hat}")
    print(f"  (max m such that DA1..DA_m all pass Bonferroni at alpha/m, alpha=0.05)")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'results', 'exact_score_ou_null_B1000.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path,
        null_eigenvalues=null_evals_all,
        obs_eigenvalues=obs_evals[:n_eigvals],
        n_reps=B, N=N, d=d, dt=dt, lag=lag,
        sigma_f_sq=sigma_f_sq, n_eigvals=n_eigvals,
        null_type='exact_score_isotropic')
    print(f"\nSaved {out_path}")
    print("DONE")


if __name__ == '__main__':
    main()
