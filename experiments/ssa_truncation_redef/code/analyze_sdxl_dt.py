#!/usr/bin/env python3
"""
Analyze SDXL dt experiment results.

Reads JSON outputs from the dt experiment (dt=0.002, 0.005) and the
original dt=0.0005 results, computes S_old and S_hat, and checks
whether rankings are preserved.
"""
import sys
import os
import json
import glob
import numpy as np
from scipy.stats import spearmanr

CODEDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.join(CODEDIR, '..')
REPODIR = os.path.join(BASEDIR, '..', '..')
sys.path.insert(0, REPODIR)
sys.path.insert(0, CODEDIR)

from ssa_redef import compute_s_old, compute_s_hat

RESDIR = os.path.join(BASEDIR, 'results')
DT_DIR = os.path.join(RESDIR, 'sdxl_dt_experiment')

PROMPTS = ['person', 'cat', 'mountains', 'flowers', 'street', 'food', 'elderly', 'car']
GS_KEYS = ['gs1p0', 'gs7p5']
GS_VALS = {'gs1p0': 1.0, 'gs7p5': 7.5}
DT_NEW = ['0p002', '0p005']
DT_ORIG = 0.0005


def load_dt_results():
    """Load all dt experiment JSON files."""
    results = {}

    for prompt in PROMPTS:
        for gs_key in GS_KEYS:
            for dt_label in DT_NEW:
                name = f"redef_{prompt}_{gs_key}_dt{dt_label}"
                json_path = os.path.join(DT_DIR, f"clip_da_{name}.json")
                if not os.path.exists(json_path):
                    # Try alternate naming
                    candidates = glob.glob(os.path.join(DT_DIR, f"*{prompt}*{gs_key}*dt{dt_label}*.json"))
                    if candidates:
                        json_path = candidates[0]
                    else:
                        print(f"  MISSING: {name}")
                        continue

                with open(json_path) as f:
                    data = json.load(f)

                dt_val = float(dt_label.replace('p', '.'))
                key = (prompt, gs_key, dt_val)
                results[key] = data

    return results


def load_original_results():
    """Load original dt=0.0005 results."""
    orig = json.load(open(os.path.join(RESDIR, 'sdxl_guidance_redef.json')))
    results = {}
    for prompt in PROMPTS:
        for gs_key in GS_KEYS:
            gs_json_key = f"gs_{gs_key.replace('gs', '')}"
            if gs_json_key in orig.get(prompt, {}):
                r = orig[prompt][gs_json_key]
                key = (prompt, gs_key, DT_ORIG)
                results[key] = r
    return results


def compute_ssa_from_json(data, dt_val):
    """Extract ρ² and compute S_old, S_hat from a run_ssa_da_clip.py JSON."""
    rho_sq = np.array(data.get('rho_sq_all', []))
    snapshot_lags = np.array(data.get('snapshot_lags_all', []))
    var_f = data.get('var_f', 0.0)
    N = data.get('n_traj', 200)
    d = data.get('embed_dim', 768)

    if len(rho_sq) == 0:
        return None

    lags_phys = snapshot_lags * dt_val
    tau_max = lags_phys[-1] if len(lags_phys) > 0 else 0

    # S_old: mean of all ρ²
    s_old = float(rho_sq.mean())

    # Find τ* from 1/N floor
    floor = 1.0 / N
    tau_star = tau_max
    for i in range(len(lags_phys)):
        if rho_sq[i] < floor:
            tau_star = lags_phys[max(0, i - 1)]
            break

    # S_hat: Riemann sum with actual spacing
    lag_spacings = np.zeros(len(lags_phys))
    for i in range(len(lags_phys)):
        if i == 0:
            lag_spacings[i] = (lags_phys[1] - lags_phys[0]) if len(lags_phys) > 1 else dt_val
        elif i == len(lags_phys) - 1:
            lag_spacings[i] = lags_phys[i] - lags_phys[i - 1]
        else:
            lag_spacings[i] = (lags_phys[i + 1] - lags_phys[i - 1]) / 2

    s_hat = 0.0
    L_star = 0
    for i in range(len(lags_phys)):
        if lags_phys[i] > tau_star:
            break
        s_hat += rho_sq[i] * lag_spacings[i]
        L_star += 1

    return {
        's_old': s_old,
        's_hat': float(s_hat),
        'tau_star': tau_star,
        'tau_max': tau_max,
        'L_star': L_star,
        'n_lags': len(rho_sq),
        'rho_sq_last': float(rho_sq[-1]) if len(rho_sq) > 0 else None,
        'ssa_original': data.get('ssa_clip', data.get('ssa_fixed20_original', None)),
    }


def main():
    print("Loading results...")
    dt_results = load_dt_results()
    orig_results = load_original_results()

    if not dt_results:
        print("No dt experiment results found. Run cluster job first.")
        return

    # Print comparison table
    print(f"\n{'='*100}")
    print(f"{'Prompt':>10s} {'gs':>5s} {'dt':>7s} {'τ_max':>7s} {'S_old':>8s} {'S_hat':>8s} "
          f"{'τ*':>7s} {'L*':>5s} {'ρ²_last':>8s}")
    print("-" * 100)

    # Collect for rank correlation
    all_data = {}  # dt_val -> list of (prompt, gs, s_old, s_hat)

    for dt_val in [DT_ORIG, 0.002, 0.005]:
        dt_label = f"{dt_val}"
        all_data[dt_val] = []

        for prompt in PROMPTS:
            for gs_key in GS_KEYS:
                key = (prompt, gs_key, dt_val)

                if dt_val == DT_ORIG:
                    if key in orig_results:
                        r = orig_results[key]
                        print(f"{prompt:>10s} {GS_VALS[gs_key]:5.1f} {dt_val:7.4f} "
                              f"{'1.00':>7s} {r['s_old_fixed20']:8.4f} {r['s_hat']:8.4f} "
                              f"{r['tau_star']:7.2f} {r['L_star']:5d} {'—':>8s}")
                        all_data[dt_val].append((prompt, gs_key, r['s_old_fixed20'], r['s_hat']))
                else:
                    if key in dt_results:
                        ssa = compute_ssa_from_json(dt_results[key], dt_val)
                        if ssa:
                            print(f"{prompt:>10s} {GS_VALS[gs_key]:5.1f} {dt_val:7.4f} "
                                  f"{ssa['tau_max']:7.2f} {ssa['s_old']:8.4f} {ssa['s_hat']:8.4f} "
                                  f"{ssa['tau_star']:7.2f} {ssa['L_star']:5d} {ssa['rho_sq_last']:8.4f}")
                            all_data[dt_val].append((prompt, gs_key, ssa['s_old'], ssa['s_hat']))
                    else:
                        print(f"{prompt:>10s} {GS_VALS[gs_key]:5.1f} {dt_val:7.4f} {'—':>7s} "
                              f"{'—':>8s} {'—':>8s} {'—':>7s} {'—':>5s} {'—':>8s}")

    # Rank correlations across dt values
    print(f"\n{'='*60}")
    print("RANK CORRELATIONS")
    print("-" * 60)

    dt_vals = sorted(all_data.keys())
    for i in range(len(dt_vals)):
        for j in range(i + 1, len(dt_vals)):
            dt_a, dt_b = dt_vals[i], dt_vals[j]
            # Match on (prompt, gs)
            pairs_a = {(p, g): (s_old, s_hat) for p, g, s_old, s_hat in all_data[dt_a]}
            pairs_b = {(p, g): (s_old, s_hat) for p, g, s_old, s_hat in all_data[dt_b]}
            common = sorted(set(pairs_a.keys()) & set(pairs_b.keys()))

            if len(common) < 3:
                continue

            s_old_a = [pairs_a[k][0] for k in common]
            s_old_b = [pairs_b[k][0] for k in common]
            s_hat_a = [pairs_a[k][1] for k in common]
            s_hat_b = [pairs_b[k][1] for k in common]

            rho_old, _ = spearmanr(s_old_a, s_old_b)
            rho_hat, _ = spearmanr(s_hat_a, s_hat_b)

            print(f"  dt={dt_a} vs dt={dt_b} (n={len(common)}):")
            print(f"    Spearman ρ(S_old): {rho_old:.4f}")
            print(f"    Spearman ρ(S_hat): {rho_hat:.4f}")

    # Check gs=7.5 > gs=1.0 ordering per prompt
    print(f"\n{'='*60}")
    print("ORDERING CHECK: gs=7.5 > gs=1.0 per prompt")
    print("-" * 60)
    for dt_val in dt_vals:
        pairs = {(p, g): (s_old, s_hat) for p, g, s_old, s_hat in all_data[dt_val]}
        n_correct_old = 0
        n_correct_hat = 0
        n_total = 0
        for prompt in PROMPTS:
            if (prompt, 'gs1p0') in pairs and (prompt, 'gs7p5') in pairs:
                n_total += 1
                if pairs[(prompt, 'gs7p5')][0] > pairs[(prompt, 'gs1p0')][0]:
                    n_correct_old += 1
                if pairs[(prompt, 'gs7p5')][1] > pairs[(prompt, 'gs1p0')][1]:
                    n_correct_hat += 1
        print(f"  dt={dt_val}: S_old {n_correct_old}/{n_total}, S_hat {n_correct_hat}/{n_total}")

    # Save
    out = {
        'comparison': {},
        'rank_correlations': {},
    }
    for dt_val in dt_vals:
        out['comparison'][str(dt_val)] = [
            {'prompt': p, 'gs': g, 's_old': s_old, 's_hat': s_hat}
            for p, g, s_old, s_hat in all_data[dt_val]
        ]
    out_path = os.path.join(RESDIR, 'sdxl_dt_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
