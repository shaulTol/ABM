#!/usr/bin/env python3
"""
Diagnostic: eigenvalue decay of isotropic null vs lag.
Single-process scan across multiple lags on GPU.
"""
import numpy as np
import torch
import sys, os, time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.kde.kde_score import KDEScore
from src.simulation.generic_langevin import GenericLangevinSimulator
from src.metrics.da_directions import extract_da_directions


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load C0 for sigma_f^2
    ctau_path = os.path.join(project_root,
        'experiments/save_autocovariance/results/alanine_C_tau_full_h_half_10k.npz')
    ctau_data = np.load(ctau_path)
    C0_data = np.array(ctau_data['C0'], dtype=np.float64)
    d = int(ctau_data['d'])
    sigma_f_sq = np.trace(C0_data) / d
    sigma_f = np.sqrt(sigma_f_sq)

    N = 2000
    h = 0.0091
    dt = 0.1
    seed = 77777
    n_eigvals = 5

    print(f"d={d}, N={N}, h={h}, dt={dt}")
    print(f"sigma_f_sq = tr(C0)/d = {sigma_f_sq:.6e}")
    print(f"sigma_f = {sigma_f:.6e}")

    # Generate isotropic null samples
    rng = np.random.RandomState(seed)
    x_null_np = sigma_f * rng.randn(N, d)
    x_null = torch.from_numpy(x_null_np).float().to(device)

    # Build KDE and simulator
    kde = KDEScore(x_null, bandwidth=h, device=device, batch_size=500)
    sim = GenericLangevinSimulator(score_fn=kde.score, samples=x_null, dt=dt, device=device)

    print(f"Simulator var_f={sim.var_f:.6e}, sigma_f={sim.sigma_f:.6e}")
    print(f"||mu_f|| = {sim.mu_f.norm().item():.6e}")

    # --- Probe drift magnitude at t=0 ---
    print("\n=== Drift magnitude at t=0 ===")
    torch.manual_seed(seed + 100)
    x_test = sim.sample_initial(N)
    grad = kde.score(x_test)  # (N, d)
    drift = 0.5 * sim.var_f * dt * grad  # one-step drift
    drift_norms = drift.norm(dim=1)
    noise_scale = sim.sigma_f * np.sqrt(dt)  # noise std per coordinate
    noise_norm_expected = noise_scale * np.sqrt(d)  # expected ||noise||

    print(f"  drift = 0.5 * var_f * dt * grad_log_f(x)")
    print(f"  ||drift||: mean={drift_norms.mean().item():.6e}, "
          f"std={drift_norms.std().item():.6e}, "
          f"max={drift_norms.max().item():.6e}, "
          f"min={drift_norms.min().item():.6e}")
    print(f"  ||grad_log_f||: mean={grad.norm(dim=1).mean().item():.6e}, "
          f"max={grad.norm(dim=1).max().item():.6e}")
    print(f"  noise per step: sigma_f * sqrt(dt) * ||z|| ~ {noise_scale:.6e} * sqrt({d}) = {noise_norm_expected:.6e}")
    print(f"  drift/noise ratio (norms): {drift_norms.mean().item() / noise_norm_expected:.4f}")

    # --- Compute C(0) for trace normalization ---
    torch.manual_seed(seed + 200)
    x0_c0 = sim.sample_initial(N)
    mu_f = sim.mu_f
    x0_c0_c = x0_c0 - mu_f.unsqueeze(0)
    C0_emp = (x0_c0_c.t() @ x0_c0_c) / N
    trace_C0 = torch.trace(C0_emp).item()
    evals_C0 = torch.linalg.eigvalsh(C0_emp).flip(0).cpu().numpy()
    print(f"\nEmpirical C(0): trace={trace_C0:.6e}")
    print(f"C(0) top 5 eigenvalues: {evals_C0[:5]}")

    # --- Scan lags ---
    lag_steps_list = [1, 10, 100, 1000, 10000, 100000]

    print(f"\n{'lag_steps':>10} {'phys_ps':>10} "
          f"{'lam1':>12} {'lam2':>12} {'lam3':>12} {'lam4':>12} {'lam5':>12} "
          f"{'OU_pred':>12} {'tr_ratio':>10} {'OU_rho':>10} {'time_s':>8}")
    print("-" * 140)

    results = []
    for lag_steps in lag_steps_list:
        phys_ps = lag_steps * dt
        t0 = time.time()

        evals_b, _, C_tau_sym = extract_da_directions(
            simulator=sim, lag=lag_steps, n_trajectories=N,
            dt=dt, device=device, seed=seed + 300000 + lag_steps)

        elapsed = time.time() - t0
        evals_np = evals_b.cpu().numpy()

        trace_Ctau = torch.trace(C_tau_sym).item()
        tr_ratio = trace_Ctau / trace_C0 if trace_C0 != 0 else float('nan')

        ou_pred = sigma_f_sq * np.exp(-phys_ps / 2.0)
        ou_rho = np.exp(-phys_ps / 2.0)

        print(f"{lag_steps:>10d} {phys_ps:>10.1f} "
              f"{evals_np[0]:>12.4e} {evals_np[1]:>12.4e} {evals_np[2]:>12.4e} "
              f"{evals_np[3]:>12.4e} {evals_np[4]:>12.4e} "
              f"{ou_pred:>12.4e} {tr_ratio:>10.6f} {ou_rho:>10.4e} {elapsed:>8.1f}")

        results.append({
            'lag_steps': lag_steps, 'phys_ps': phys_ps,
            'evals': evals_np[:n_eigvals], 'tr_ratio': tr_ratio,
            'ou_pred': ou_pred, 'ou_rho': ou_rho, 'elapsed': elapsed
        })

    # --- Fit exponential decay ---
    print("\n=== Exponential decay fit ===")
    # Fit log(lambda_1) vs phys_lag to get decay constant
    phys_lags = np.array([r['phys_ps'] for r in results])
    lam1s = np.array([r['evals'][0] for r in results])
    tr_ratios = np.array([r['tr_ratio'] for r in results])

    # Only fit where lambda_1 > 0 and is decaying
    mask = lam1s > 0
    if mask.sum() >= 2:
        log_lam1 = np.log(lam1s[mask])
        phys_fit = phys_lags[mask]
        # Linear fit: log(lam1) = a - c * phys_lag
        if len(phys_fit) >= 2:
            coeffs = np.polyfit(phys_fit, log_lam1, 1)
            c_fit = -coeffs[0]
            a_fit = coeffs[1]
            print(f"  Fit: log(lam1) = {a_fit:.4f} - {c_fit:.6f} * phys_lag_ps")
            print(f"  Fitted decay constant c = {c_fit:.6f} per ps")
            print(f"  OU prediction: c = 1/2 = 0.500000 per ps")
            print(f"  Ratio: fitted_c / OU_c = {c_fit / 0.5:.6f}")

    # tr_ratio fit
    mask_tr = tr_ratios > 0
    if mask_tr.sum() >= 2:
        log_tr = np.log(tr_ratios[mask_tr])
        phys_fit_tr = phys_lags[mask_tr]
        coeffs_tr = np.polyfit(phys_fit_tr, log_tr, 1)
        c_tr = -coeffs_tr[0]
        print(f"\n  Fit tr(C(tau))/tr(C(0)): log(rho) = {coeffs_tr[1]:.4f} - {c_tr:.6f} * phys_lag_ps")
        print(f"  Fitted rho decay constant = {c_tr:.6f} per ps")
        print(f"  OU prediction: 0.500000 per ps")

    print("\nDONE")


if __name__ == '__main__':
    main()
