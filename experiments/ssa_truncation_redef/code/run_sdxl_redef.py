#!/usr/bin/env python3
"""
SDXL Guidance Sweep: Recompute S_old and S_hat from cached CLIP embeddings.

Reads pre-computed rho_sq_all and snapshot_lags_all from existing JSON files
(8 prompts × 4 guidance scales). No GPU or trajectory regeneration needed.

Also computes τ* via the noise-floor criterion for each configuration.
"""
import sys
import os
import json
import numpy as np

# Path setup
CODEDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.join(CODEDIR, '..')
REPODIR = os.path.join(BASEDIR, '..', '..')
sys.path.insert(0, REPODIR)

from ssa_redef import compute_s_old, compute_s_hat, compute_both_ssa
from experiments.noise_floor.noise_floor_utils import analytic_floor_upper

# ── JSON paths ──
GUIDANCE_DIR = os.path.join(REPODIR, 'experiments', 'sdxl', 'results_guidance_sweep')
MULTIPROMPT_DIR = os.path.join(REPODIR, 'experiments', 'sdxl', 'results_multiprompt_sweep')

# Prompt mapping: person is in guidance_sweep/, others in multiprompt_sweep/
PROMPTS = [
    ('person', 'clip_da_sweep_gs{gs}.json', GUIDANCE_DIR),
    ('cat', 'clip_da_mp_p0_cat_gs{gs}.json', MULTIPROMPT_DIR),
    ('mountains', 'clip_da_mp_p1_mountains_gs{gs}.json', MULTIPROMPT_DIR),
    ('flowers', 'clip_da_mp_p2_flowers_gs{gs}.json', MULTIPROMPT_DIR),
    ('street', 'clip_da_mp_p3_street_gs{gs}.json', MULTIPROMPT_DIR),
    ('food', 'clip_da_mp_p4_food_gs{gs}.json', MULTIPROMPT_DIR),
    ('elderly', 'clip_da_mp_p5_elderly_gs{gs}.json', MULTIPROMPT_DIR),
    ('car', 'clip_da_mp_p6_car_gs{gs}.json', MULTIPROMPT_DIR),
]

GUIDANCE_SCALES = ['1p0', '3p0', '7p5', '15p0']
GS_VALUES = [1.0, 3.0, 7.5, 15.0]


def gs_label(gs_str):
    """Convert '7p5' → 7.5."""
    return float(gs_str.replace('p', '.'))


def find_tau_star_from_floor(lags, rho_sq, var_f, d, N):
    """
    Find τ* where ρ²(τ) drops below the analytic noise floor.

    For scalar ρ², the noise floor is approximately:
      floor(τ) ≈ analytic_floor_upper(τ, var_f, d, N) / (d * var_f)
    since ρ = tr(C(τ))/tr(C(0)) and the floor is on eigenvalues.

    Alternatively, use 1/N as a simple CLT floor.
    We use both and take the more conservative (larger τ*).
    """
    lags = np.asarray(lags, dtype=float)
    rho_sq = np.asarray(rho_sq)

    # Method 1: 1/N floor (CLT)
    floor_clt = 1.0 / N
    tau_star_clt = lags[-1]
    for i in range(len(lags)):
        if lags[i] <= 0:
            continue
        if rho_sq[i] < floor_clt:
            tau_star_clt = lags[max(0, i - 1)]
            break

    # Method 2: MP-based floor adapted for scalar ρ²
    # ρ² ≈ (λ₁/tr(C(0)))². The floor on λ₁ is analytic_floor_upper.
    # tr(C(0)) ≈ d * var_f, so floor on ρ² ≈ (floor_upper / (d * var_f))²
    floor_mp = analytic_floor_upper(lags, var_f, d, N)
    floor_rho_sq = (floor_mp / (d * var_f)) ** 2
    tau_star_mp = lags[-1]
    for i in range(len(lags)):
        if lags[i] <= 0:
            continue
        if rho_sq[i] < floor_rho_sq[i]:
            tau_star_mp = lags[max(0, i - 1)]
            break

    # Take the more conservative (larger) τ*
    tau_star = max(tau_star_clt, tau_star_mp)
    return float(tau_star)


def process_one_json(filepath):
    """Load a JSON and compute S_old, S_hat."""
    with open(filepath) as f:
        data = json.load(f)

    rho_sq_all = np.array(data['rho_sq_all'])
    snapshot_lags = np.array(data['snapshot_lags_all'])
    var_f = data['var_f']
    dt = data['dt']
    T = data['T']
    N = data['n_traj']
    d = data['embed_dim']

    # Convert snapshot lags to physical time
    lags_phys = snapshot_lags * dt

    # Find τ*
    tau_star = find_tau_star_from_floor(lags_phys, rho_sq_all, var_f, d, N)

    # Compute lag spacing (non-uniform for log+fixed schedule)
    # For non-uniform lags, dt in the Riemann sum should be the actual spacing
    # between consecutive lags, not the simulation dt.
    # Use trapezoidal-like approach: dt_i = (lag_{i+1} - lag_{i-1}) / 2
    lag_spacings = np.zeros(len(lags_phys))
    for i in range(len(lags_phys)):
        if i == 0:
            lag_spacings[i] = (lags_phys[1] - lags_phys[0]) if len(lags_phys) > 1 else dt
        elif i == len(lags_phys) - 1:
            lag_spacings[i] = lags_phys[i] - lags_phys[i - 1]
        else:
            lag_spacings[i] = (lags_phys[i + 1] - lags_phys[i - 1]) / 2

    # S_old: mean of all ρ² (matching ssa_fixed20 style — lags >= 20*dt)
    s_old_all = float(rho_sq_all.mean())
    # Also compute matching the stored ssa_fixed20 (lags >= 20)
    mask_fixed20 = snapshot_lags >= 20
    s_old_fixed20 = float(rho_sq_all[mask_fixed20].mean()) if mask_fixed20.any() else s_old_all

    # S_hat: Riemann sum with actual lag spacings up to τ*
    s_hat = 0.0
    L_star = 0
    for i in range(len(lags_phys)):
        if lags_phys[i] <= 0:
            continue
        if lags_phys[i] > tau_star:
            break
        s_hat += rho_sq_all[i] * lag_spacings[i]
        L_star += 1
    s_hat = float(s_hat)

    return {
        's_old_all': s_old_all,
        's_old_fixed20': s_old_fixed20,
        's_hat': s_hat,
        'tau_star': tau_star,
        'L_star': L_star,
        'ssa_clip_original': data.get('ssa_clip', None),
        'ssa_fixed20_original': data.get('ssa_fixed20', None),
        'var_f': var_f,
        'dt': dt,
        'T': T,
        'N': N,
        'd': d,
        'n_lags_total': len(rho_sq_all),
        'rho_sq_first5': rho_sq_all[:5].tolist(),
        'rho_sq_last5': rho_sq_all[-5:].tolist(),
    }


def main():
    results_dir = os.path.join(BASEDIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}

    for prompt_name, template, basedir in PROMPTS:
        prompt_results = {}
        for gs_str, gs_val in zip(GUIDANCE_SCALES, GS_VALUES):
            filepath = os.path.join(basedir, template.format(gs=gs_str))
            if not os.path.exists(filepath):
                print(f"  SKIP {prompt_name} gs={gs_val}: {filepath} not found")
                continue

            res = process_one_json(filepath)
            prompt_results[f'gs_{gs_str}'] = res
            print(f"  {prompt_name:10s} gs={gs_val:5.1f}: "
                  f"S_old={res['s_old_fixed20']:.4f} (orig={res['ssa_fixed20_original']:.4f}), "
                  f"S_hat={res['s_hat']:.4f}, τ*={res['tau_star']:.2f}, L*={res['L_star']}")

        all_results[prompt_name] = prompt_results

    # Summary table
    print("\n" + "="*80)
    print(f"{'Prompt':>12s} {'gs':>5s} {'S_old':>8s} {'S_hat':>8s} {'τ*':>8s} {'L*':>5s}")
    print("-"*80)
    for prompt_name in [p[0] for p in PROMPTS]:
        for gs_str, gs_val in zip(GUIDANCE_SCALES, GS_VALUES):
            key = f'gs_{gs_str}'
            if key in all_results.get(prompt_name, {}):
                r = all_results[prompt_name][key]
                print(f"{prompt_name:>12s} {gs_val:5.1f} {r['s_old_fixed20']:8.4f} "
                      f"{r['s_hat']:8.4f} {r['tau_star']:8.2f} {r['L_star']:5d}")

    # Save
    out_path = os.path.join(results_dir, 'sdxl_guidance_redef.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
