"""Compute slow-mode eigenvalue weights for SSA ordering justification.

For each cell (prompt × guidance scale), computes:
  w_2     = lambda_1(C_sym(tau)) / tr(C(tau))
  w_2+w_3 = (lambda_1 + lambda_2) / tr(C(tau))
  rho_T_N = rho_hat(T)^2 * N   (how far above stopping rule 1/N)

Uses trajectory-averaged C_sym(tau) at the last fixed-20 lag before tau_star.
Reads from z_clip_traj_*.npz (CLIP trajectories) + traj_avg_*.json (metadata).

Output: weight_table.json with all results.
"""
import json, numpy as np, os, sys, glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAJ_DIR = os.path.join(SCRIPT_DIR, "results")


def compute_C_sym_traj_averaged(z_traj, fixed20_lags, fixed20_with_zero, target_lag):
    """Compute trajectory-averaged symmetrized autocovariance C_sym(target_lag).

    Uses per-starting-time centering (consistent with SSA computation).
    Returns C_sym (D, D), tr_C (scalar).
    """
    N, n_times, D = z_traj.shape
    target_idx_offset = target_lag // 20

    C_sum = np.zeros((D, D), dtype=np.float64)
    n_pairs = 0

    for i in range(n_times):
        j = i + target_idx_offset
        if j < n_times:
            if fixed20_with_zero[j] - fixed20_with_zero[i] == target_lag:
                mu_s = z_traj[:, i, :].mean(axis=0)  # (D,)
                z_s_c = z_traj[:, i, :] - mu_s  # (N, D)
                z_t_c = z_traj[:, j, :] - mu_s  # (N, D)
                # C(s, s+tau) = (1/N) Z_s^T Z_t
                C_pair = (z_s_c.T @ z_t_c) / N  # (D, D)
                C_sum += C_pair
                n_pairs += 1

    if n_pairs == 0:
        return None, 0, 0

    C_avg = C_sum / n_pairs
    C_sym = (C_avg + C_avg.T) / 2
    tr_C = np.trace(C_avg)

    return C_sym, tr_C, n_pairs


def process_cell(npz_path, json_path):
    """Compute eigenvalue weights for one cell."""
    data = np.load(npz_path)
    z0 = data['z0_clip']  # (N, D)
    snapshot_lags = data['snapshot_lags']
    N, D = z0.shape

    jdata = json.load(open(json_path))
    dt = jdata.get('dt', 0.005)
    tau_star = jdata.get('tau_star_steps', 2000)

    # Build trajectory array for fixed-20 grid
    fixed20_lags = sorted([int(l) for l in snapshot_lags if l % 20 == 0 and l > 0])
    fixed20_with_zero = [0] + fixed20_lags
    n_times = len(fixed20_with_zero)

    z_traj = np.zeros((N, n_times, D), dtype=np.float32)
    z_traj[:, 0, :] = z0
    for i, lag in enumerate(fixed20_lags):
        z_traj[:, i + 1, :] = data[f'z_clip_{lag}']

    # Pick the last fixed-20 lag before or at tau_star for eigendecomposition
    valid_lags = [l for l in fixed20_lags if l <= tau_star]
    if not valid_lags:
        return None
    eval_lag = valid_lags[-1]

    # Compute trajectory-averaged C_sym at eval_lag
    C_sym, tr_C, n_pairs = compute_C_sym_traj_averaged(
        z_traj, fixed20_lags, fixed20_with_zero, eval_lag)

    if C_sym is None or tr_C < 1e-12:
        return None

    # Eigendecompose (top few eigenvalues only)
    eigenvalues = np.linalg.eigvalsh(C_sym)[::-1]  # descending
    lambda_1 = float(eigenvalues[0])
    lambda_2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0

    w_2 = lambda_1 / tr_C
    w_2_w_3 = (lambda_1 + lambda_2) / tr_C

    # rho_hat(T)^2 * N — from the last lag in rho_sq_combined
    rho_sq_combined = jdata.get('rho_sq_combined', {})
    T_lag = max(int(k) for k in rho_sq_combined) if rho_sq_combined else 0
    rho_T_sq = rho_sq_combined.get(str(T_lag), 0)
    rho_T_N = rho_T_sq * N

    return {
        'eval_lag': eval_lag,
        'tau_star': tau_star,
        'lambda_1': lambda_1,
        'lambda_2': lambda_2,
        'tr_C': float(tr_C),
        'w_2': w_2,
        'w_2_w_3': w_2_w_3,
        'rho_T_sq': float(rho_T_sq),
        'rho_T_N': float(rho_T_N),
        'n_pairs': n_pairs,
        'N': N,
        'D': D,
        'dt': dt,
    }


def main():
    os.chdir(TRAJ_DIR)

    # Collect all cells
    results = {}

    # 8 original prompts
    for npz_path in sorted(glob.glob("z_clip_traj_traj_*.npz")):
        name = npz_path.replace("z_clip_traj_", "").replace(".npz", "")
        json_path = f"traj_avg_{name}.json"
        if not os.path.exists(json_path):
            print(f"SKIP {name}: no JSON")
            continue
        r = process_cell(npz_path, json_path)
        if r:
            results[name] = r
            print(f"  {name}: w2={r['w_2']:.4f}, w2+w3={r['w_2_w_3']:.4f}, "
                  f"rhoT*N={r['rho_T_N']:.2f}, lag={r['eval_lag']}")

    # 6 extended prompts
    for npz_path in sorted(glob.glob("z_clip_traj_ext_*.npz")):
        name = npz_path.replace("z_clip_traj_", "").replace(".npz", "")
        json_path = f"traj_avg_{name}.json"
        if not os.path.exists(json_path):
            print(f"SKIP {name}: no JSON")
            continue
        r = process_cell(npz_path, json_path)
        if r:
            results[name] = r
            print(f"  {name}: w2={r['w_2']:.4f}, w2+w3={r['w_2_w_3']:.4f}, "
                  f"rhoT*N={r['rho_T_N']:.2f}, lag={r['eval_lag']}")

    # Save results
    out_path = os.path.join(SCRIPT_DIR, "weight_table.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} cells to {out_path}")

    # Print summary table
    print(f"\n{'Cell':<45} {'w2':>6} {'w2+w3':>7} {'rhoT*N':>7} {'tau*':>5}")
    print("-" * 75)
    for name in sorted(results):
        r = results[name]
        print(f"{name:<45} {r['w_2']:6.4f} {r['w_2_w_3']:7.4f} "
              f"{r['rho_T_N']:7.2f} {r['tau_star']:5d}")


if __name__ == "__main__":
    main()
