#!/usr/bin/env python3
"""
Diagnostic: canonical Langevin with EXACT Gaussian score (no KDE).

For f = N(0, sigma_f^2 * I_d), the score is:
    grad_log_f(x) = -x / sigma_f^2

The canonical Langevin is:
    dX = (sigma_f^2 / 2) * (-x/sigma_f^2) dt + sigma_f dW
       = -x/2 dt + sigma_f dW

This is an OU process with rate 1/2. The autocovariance is:
    C(tau) = sigma_f^2 * exp(-tau/2) * I_d

We verify:
(a) Eigenvalues decay as exp(-tau/2) with exact score
(b) Compare to KDE score (which returns 0 in 45D)
(c) Run MC null: 100 reps at lag=100000 steps with exact score to get null eigenvalue distribution
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

    N = 2000
    dt = 0.1
    n_eigvals = 10

    print(f"d={d}, N={N}, dt={dt}")
    print(f"sigma_f_sq = {sigma_f_sq:.6e}, sigma_f = {sigma_f:.6e}")

    # Build exact score
    exact_score = ExactGaussianScore(sigma_f_sq)

    # Generate isotropic samples for initial conditions
    rng_np = np.random.RandomState(55555)
    x_init_np = sigma_f * rng_np.randn(N, d)
    x_init = torch.from_numpy(x_init_np).float().to(device)

    # Build simulator with exact score
    sim = GenericLangevinSimulator(
        score_fn=exact_score, samples=x_init, dt=dt, device=device,
        var_f=sigma_f_sq)

    print(f"Simulator var_f={sim.var_f:.6e}, sigma_f={sim.sigma_f:.6e}")

    # --- Verify drift is NOT zero ---
    torch.manual_seed(55555)
    x_test = sim.sample_initial(100)
    grad = exact_score(x_test)
    drift = 0.5 * sim.var_f * dt * grad
    drift_norms = drift.norm(dim=1)
    noise_scale = sim.sigma_f * np.sqrt(dt)
    noise_norm = noise_scale * np.sqrt(d)
    print(f"\n=== Drift check (exact score) ===")
    print(f"  ||drift||: mean={drift_norms.mean().item():.6e}, max={drift_norms.max().item():.6e}")
    print(f"  ||noise||: expected={noise_norm:.6e}")
    print(f"  drift/noise ratio: {drift_norms.mean().item()/noise_norm:.6f}")

    # --- Part A: Eigenvalue decay scan ---
    print(f"\n=== Part A: Eigenvalue decay scan (exact score) ===")

    # C(0)
    torch.manual_seed(55555)
    x0_c0 = sim.sample_initial(N)
    mu_f = sim.mu_f
    C0_emp = ((x0_c0 - mu_f).t() @ (x0_c0 - mu_f)) / N
    trace_C0 = torch.trace(C0_emp).item()

    lag_list = [1, 10, 100, 1000, 10000, 100000]
    print(f"\n  {'lag':>10} {'phys_ps':>10} {'lam1':>12} {'lam2':>12} {'lam3':>12} "
          f"{'OU_pred':>12} {'rho_obs':>10} {'rho_OU':>10} {'time':>8}")
    print("  " + "-" * 100)

    for lag_steps in lag_list:
        phys_ps = lag_steps * dt
        t0 = time.time()
        evals, _, C_tau = extract_da_directions(
            simulator=sim, lag=lag_steps, n_trajectories=N,
            dt=dt, device=device, seed=55555 + lag_steps)
        elapsed = time.time() - t0
        evals_np = evals.cpu().numpy()
        tr_Ctau = torch.trace(C_tau).item()
        rho_obs = tr_Ctau / trace_C0
        rho_ou = np.exp(-phys_ps / 2.0)
        ou_pred = sigma_f_sq * np.exp(-phys_ps / 2.0)
        print(f"  {lag_steps:>10d} {phys_ps:>10.1f} {evals_np[0]:>12.4e} {evals_np[1]:>12.4e} "
              f"{evals_np[2]:>12.4e} {ou_pred:>12.4e} {rho_obs:>10.6f} {rho_ou:>10.4e} {elapsed:>8.1f}")

    # --- Part B: MC null with exact score (100 reps at lag=100000) ---
    print(f"\n=== Part B: MC null with exact score, lag=100000 (phys_lag=10000 ps) ===")
    print(f"  100 reps, N={N}, d={d}")

    null_evals_all = np.zeros((100, n_eigvals))
    seed_offset = 60000

    for rep in range(100):
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
            simulator=sim_null, lag=100000, n_trajectories=N,
            dt=dt, device=device, seed=seed + 100000)

        evals_np = evals_b.cpu().numpy()
        null_evals_all[rep, :] = evals_np[:n_eigvals]

        elapsed = time.time() - t0
        if rep < 5 or rep % 20 == 19:
            print(f"  Rep {rep+1:>3d}/100: lam1={evals_np[0]:.6e}, lam2={evals_np[1]:.6e}, "
                  f"gap={evals_np[0]/max(evals_np[1],1e-15):.2f}, time={elapsed:.0f}s")

    # --- Part B results ---
    print(f"\n=== Part B results: exact-score null eigenvalue distribution ===")
    print(f"  {'k':>4} {'median':>12} {'mean':>12} {'2.5th':>12} {'97.5th':>12} {'min':>12} {'max':>12}")
    print("  " + "-" * 80)

    for k in range(min(n_eigvals, 5)):
        vals = null_evals_all[:, k]
        print(f"  {k+1:>4d} {np.median(vals):>12.4e} {np.mean(vals):>12.4e} "
              f"{np.percentile(vals, 2.5):>12.4e} {np.percentile(vals, 97.5):>12.4e} "
              f"{np.min(vals):>12.4e} {np.max(vals):>12.4e}")

    # --- Part C: Compare observed vs exact-score null ---
    obs_evals = np.array([1.2099e-2, 6.0956e-4, 6.024e-5, 4.820e-5, 2.140e-5])
    print(f"\n=== Part C: Observed vs exact-score null at phys_lag=10000 ps ===")
    print(f"  {'k':>4} {'obs':>12} {'null_med':>12} {'null_97.5':>12} {'obs/med':>10} {'obs>97.5?':>10}")
    print("  " + "-" * 60)

    for k in range(min(5, n_eigvals)):
        vals = null_evals_all[:, k]
        med = np.median(vals)
        p975 = np.percentile(vals, 97.5)
        ratio = obs_evals[k] / med if med > 0 else float('inf')
        above = "YES" if obs_evals[k] > p975 else "NO"
        print(f"  {k+1:>4d} {obs_evals[k]:>12.4e} {med:>12.4e} {p975:>12.4e} {ratio:>10.2f} {above:>10}")

    # Per-direction p-values
    print(f"\n=== Per-direction p-values (exact-score null, 100 reps) ===")
    for k in range(min(5, n_eigvals)):
        vals = null_evals_all[:, k]
        p = (1 + np.sum(vals >= obs_evals[k])) / (1 + len(vals))
        bonf = min(1.0, p * 2)  # Bonferroni m=2
        print(f"  k={k+1}: p={p:.4f}, Bonferroni(m=2)={bonf:.4f}")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'results', 'exact_score_ou_null.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path,
        null_eigenvalues=null_evals_all,
        n_reps=100, N=N, d=d, dt=dt, lag=100000,
        sigma_f_sq=sigma_f_sq, n_eigvals=n_eigvals,
        null_type='exact_score_isotropic')
    print(f"\nSaved {out_path}")
    print("DONE")


if __name__ == '__main__':
    main()
