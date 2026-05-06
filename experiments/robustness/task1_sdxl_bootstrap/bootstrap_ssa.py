#!/usr/bin/env python3
"""
Bootstrap uncertainty analysis for SDXL guidance sweep SSA (Table 2).

Data situation:
- We have 111 per-lag ρ²(τ) values per cell (8 prompts × 4 scales = 32 cells)
- SSA_fixed20 = mean of ρ² at 100 evenly-spaced lags (τ=20,40,...,2000)
- Per-trajectory data was NOT saved (only the mean ρ̂(τ) per lag)
- CLIP embeddings for entropy only saved for gs=7.5

Bootstrap strategy:
1. **Lag bootstrap** for SSA: resample which of the 100 fixed20 lags enter
   the SSA mean. This quantifies sensitivity to lag selection, which is the
   main source of methodological uncertainty (trajectory count N=200 is fixed).
2. **Paired lag bootstrap** for peak probability: for each bootstrap replicate,
   compute SSA at all 4 scales using the SAME resampled lag set, then check
   whether gs=7.5 still wins. This respects the pairing.
3. Entropy: report point estimates only (no per-image embeddings saved for
   most cells; flag this limitation).

Output:
  results_sdxl_bootstrap.csv
  results_sdxl_peak_probabilities.csv
  SUMMARY.md
"""
import json
import glob
import os
import numpy as np
import csv

EXPDIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(os.path.dirname(EXPDIR), '..', 'sdxl', 'results_guidance_sweep')
MP_DIR = os.path.join(os.path.dirname(EXPDIR), '..', 'sdxl', 'results_multiprompt_sweep')
SWEEP_DIR = os.path.abspath(SWEEP_DIR)
MP_DIR = os.path.abspath(MP_DIR)

N_BOOT = 10000
RNG = np.random.RandomState(42)

# ── Prompt/file mapping ──
PROMPTS = [
    {
        'label': 'person',
        'ssa_pattern': os.path.join(SWEEP_DIR, 'clip_da_sweep_gs{gs}.json'),
        'ent_pattern': os.path.join(SWEEP_DIR, 'gen_sweep_gs{gs}_entropy.json'),
    },
]
for i, (plabel, _) in enumerate([
    ('cat', 0), ('mountains', 1), ('flowers', 2),
    ('street', 3), ('food', 4), ('elderly', 5), ('car', 6),
]):
    PROMPTS.append({
        'label': plabel,
        'ssa_pattern': os.path.join(MP_DIR, f'clip_da_mp_p{i}_{plabel}_gs{{gs}}.json'),
        'ent_pattern': os.path.join(MP_DIR, f'gen_mp_p{i}_{plabel}_gs{{gs}}_entropy.json'),
    })

GS_LABELS = ['1p0', '3p0', '7p5', '15p0']
GS_VALUES = [1.0, 3.0, 7.5, 15.0]


def load_rho_sq_fixed20(filepath):
    """Load ρ²(τ) values at the fixed20 lags from a clip_da JSON."""
    with open(filepath) as f:
        d = json.load(f)
    rho_sq = np.array(d['rho_sq_all'])
    lags = np.array(d['snapshot_lags_all'])
    # fixed20 lags: 20, 40, 60, ..., 2000
    fixed20 = set(range(20, 2001, 20))
    mask = np.array([l in fixed20 for l in lags])
    return rho_sq[mask]


def load_entropy(filepath):
    """Load kNN entropy (k=5, d=50) from an entropy JSON."""
    with open(filepath) as f:
        d = json.load(f)
    return d['entropy']['knn']['k5']['50']


def bootstrap_ssa(rho_sq_fixed20, n_boot=N_BOOT, rng=RNG):
    """Bootstrap SSA by resampling which lags enter the mean.

    Each replicate draws 100 lags WITH replacement from the 100 fixed20 lags.
    """
    n_lags = len(rho_sq_fixed20)
    boot_ssas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n_lags, size=n_lags)
        boot_ssas[b] = rho_sq_fixed20[idx].mean()
    return boot_ssas


def main():
    print(f'SDXL Guidance Sweep Bootstrap Analysis')
    print(f'N_boot = {N_BOOT}')
    print(f'=' * 70)

    # ── Load all data ──
    all_data = {}  # (prompt_label, gs_label) -> rho_sq_fixed20
    all_entropy = {}
    all_ssa_point = {}

    for pinfo in PROMPTS:
        label = pinfo['label']
        for gs_label, gs_val in zip(GS_LABELS, GS_VALUES):
            ssa_file = pinfo['ssa_pattern'].format(gs=gs_label)
            ent_file = pinfo['ent_pattern'].format(gs=gs_label)

            rho_sq = load_rho_sq_fixed20(ssa_file)
            all_data[(label, gs_label)] = rho_sq
            all_ssa_point[(label, gs_label)] = rho_sq.mean()

            if os.path.exists(ent_file):
                all_entropy[(label, gs_label)] = load_entropy(ent_file)
            else:
                all_entropy[(label, gs_label)] = np.nan

            print(f'  {label:>10} gs={gs_val:>4}: {len(rho_sq)} lags, '
                  f'SSA={rho_sq.mean():.4f}, H={all_entropy[(label, gs_label)]:.1f}')

    # ── Bootstrap SSA for each cell ──
    print(f'\nBootstrapping SSA ({N_BOOT} replicates per cell)...')
    boot_results = {}
    for key, rho_sq in all_data.items():
        boot_ssas = bootstrap_ssa(rho_sq)
        boot_results[key] = boot_ssas

    # ── Write results_sdxl_bootstrap.csv ──
    csv_path = os.path.join(EXPDIR, 'results_sdxl_bootstrap.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt', 'w', 'ssa_mean', 'ssa_se', 'ssa_ci_lo',
                          'ssa_ci_hi', 'entropy_mean', 'entropy_se',
                          'entropy_ci_lo', 'entropy_ci_hi'])
        for pinfo in PROMPTS:
            label = pinfo['label']
            for gs_label, gs_val in zip(GS_LABELS, GS_VALUES):
                key = (label, gs_label)
                boot = boot_results[key]
                ssa_mean = boot.mean()
                ssa_se = boot.std()
                ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
                ent = all_entropy[key]
                # No entropy bootstrap — point estimate only
                writer.writerow([
                    label, gs_val,
                    f'{ssa_mean:.6f}', f'{ssa_se:.6f}',
                    f'{ci_lo:.6f}', f'{ci_hi:.6f}',
                    f'{ent:.4f}' if not np.isnan(ent) else '',
                    '', '', ''  # no entropy SE/CI
                ])
    print(f'Saved: {csv_path}')

    # ── Peak probability: Pr[SSA(7.5) > SSA(w)] via paired bootstrap ──
    print(f'\nComputing peak probabilities (paired lag bootstrap)...')
    peak_rows = []

    for pinfo in PROMPTS:
        label = pinfo['label']
        # For paired bootstrap, use same lag indices across all 4 scales
        rho_75 = all_data[(label, '7p5')]
        n_lags = len(rho_75)

        for gs_label, gs_val in zip(GS_LABELS, GS_VALUES):
            if gs_label == '7p5':
                continue
            rho_w = all_data[(label, gs_label)]

            # Paired bootstrap: same resampled lag indices for both
            count_75_wins = 0
            for b in range(N_BOOT):
                idx = RNG.randint(0, n_lags, size=n_lags)
                ssa_75_b = rho_75[idx].mean()
                ssa_w_b = rho_w[idx].mean()
                if ssa_75_b > ssa_w_b:
                    count_75_wins += 1

            prob = count_75_wins / N_BOOT
            delta = all_ssa_point[(label, '7p5')] - all_ssa_point[(label, gs_label)]
            peak_rows.append({
                'prompt': label,
                'w_compare': gs_val,
                'prob_w75_greater': prob,
                'delta_ssa': delta,
            })
            flag = '' if prob >= 0.95 else ' *** < 0.95'
            print(f'  {label:>10} SSA(7.5) > SSA({gs_val}): '
                  f'Pr={prob:.3f}, Δ={delta:+.4f}{flag}')

    peak_csv = os.path.join(EXPDIR, 'results_sdxl_peak_probabilities.csv')
    with open(peak_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['prompt', 'w_compare',
                                                'prob_w75_greater', 'delta_ssa'])
        writer.writeheader()
        writer.writerows(peak_rows)
    print(f'Saved: {peak_csv}')

    # ── Summary table for paper ──
    print(f'\n{"=" * 70}')
    print(f'SUMMARY TABLE (for paper)')
    print(f'{"=" * 70}')
    print(f'{"Prompt":>12} {"w":>5} {"SSA":>8} {"SE":>7} {"95% CI":>18} {"H(kNN)":>10}')
    print(f'{"-" * 70}')
    for pinfo in PROMPTS:
        label = pinfo['label']
        for gs_label, gs_val in zip(GS_LABELS, GS_VALUES):
            key = (label, gs_label)
            boot = boot_results[key]
            ssa_mean = boot.mean()
            ssa_se = boot.std()
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            ent = all_entropy[key]
            ent_str = f'{ent:.1f}' if not np.isnan(ent) else '—'
            peak = ' *' if gs_label == '7p5' else ''
            print(f'{label:>12} {gs_val:>5} {ssa_mean:>8.4f} {ssa_se:>7.4f} '
                  f'[{ci_lo:.4f}, {ci_hi:.4f}] {ent_str:>10}{peak}')

    # ── Check for weak peaks ──
    print(f'\n{"=" * 70}')
    print(f'PEAK ROBUSTNESS CHECK')
    print(f'{"=" * 70}')
    weak_peaks = []
    for row in peak_rows:
        if row['prob_w75_greater'] < 0.95:
            weak_peaks.append(row)
            print(f'  WARNING: {row["prompt"]}: Pr[SSA(7.5)>SSA({row["w_compare"]})] = '
                  f'{row["prob_w75_greater"]:.3f} < 0.95')

    if not weak_peaks:
        print(f'  All 8 prompts: Pr[SSA(7.5) > SSA(w)] >= 0.95 for all w != 7.5')
        print(f'  The "SSA peaks at w=7.5" claim is robust under lag bootstrap.')
    else:
        print(f'\n  {len(weak_peaks)} comparison(s) have Pr < 0.95.')

    # ── Write SUMMARY.md ──
    write_summary(boot_results, peak_rows, all_entropy, weak_peaks)


def write_summary(boot_results, peak_rows, all_entropy, weak_peaks):
    lines = []
    lines.append('# Task 1: Bootstrap SEs for SDXL Guidance Sweep SSA')
    lines.append('')
    lines.append('## Method')
    lines.append('')
    lines.append('**Lag bootstrap** (N_boot=10,000): For each of the 32 cells ')
    lines.append('(8 prompts × 4 guidance scales), SSA_fixed20 is the mean of ')
    lines.append('ρ²(τ) over 100 evenly-spaced lags (τ = 20, 40, ..., 2000). ')
    lines.append('We resample these 100 lag-level ρ² values with replacement ')
    lines.append('to get bootstrap distributions of SSA.')
    lines.append('')
    lines.append('**Paired lag bootstrap** for peak probabilities: For each ')
    lines.append('prompt, we resample the SAME lag indices across all 4 scales, ')
    lines.append('then compute Pr[SSA(w=7.5) > SSA(w)] over 10,000 replicates.')
    lines.append('')
    lines.append('**Limitation**: Per-trajectory data (N=200 Langevin trajectories) ')
    lines.append('was not saved — only the mean ρ̂(τ) at each lag. The bootstrap ')
    lines.append('therefore quantifies **lag selection sensitivity**, not trajectory ')
    lines.append('sampling uncertainty. Since trajectories are iid given z₀, the ')
    lines.append('trajectory-level variance contributes O(1/√200) ≈ 7% relative ')
    lines.append('noise to each ρ̂(τ), which is smaller than the lag-to-lag ')
    lines.append('variation visible in the ρ²(τ) curves.')
    lines.append('')
    lines.append('**Entropy**: Point estimates only (kNN, k=5, d=50). CLIP ')
    lines.append('embeddings for most cells were not saved; entropy bootstrap ')
    lines.append('would require regenerating 2,000 images per cell.')
    lines.append('')
    lines.append('## Results')
    lines.append('')
    lines.append('### SSA with Bootstrap SEs')
    lines.append('')
    lines.append('| Prompt | w=1.0 | w=3.0 | w=7.5 | w=15.0 |')
    lines.append('|--------|-------|-------|-------|--------|')

    for pinfo in PROMPTS:
        label = pinfo['label']
        cells = []
        for gs_label in GS_LABELS:
            key = (label, gs_label)
            boot = boot_results[key]
            m = boot.mean()
            se = boot.std()
            bold = '**' if gs_label == '7p5' else ''
            cells.append(f'{bold}{m:.3f}±{se:.3f}{bold}')
        lines.append(f'| {label} | {" | ".join(cells)} |')

    lines.append('')
    lines.append('### Peak Probabilities: Pr[SSA(w=7.5) > SSA(w)]')
    lines.append('')
    lines.append('| Prompt | vs w=1.0 | vs w=3.0 | vs w=15.0 |')
    lines.append('|--------|----------|----------|-----------|')

    for pinfo in PROMPTS:
        label = pinfo['label']
        cells = []
        for gs_val in [1.0, 3.0, 15.0]:
            row = next((r for r in peak_rows
                        if r['prompt'] == label and r['w_compare'] == gs_val), None)
            if row:
                p = row['prob_w75_greater']
                flag = '' if p >= 0.95 else ' ⚠'
                cells.append(f'{p:.3f}{flag}')
            else:
                cells.append('—')
        lines.append(f'| {label} | {" | ".join(cells)} |')

    lines.append('')
    lines.append('### Assessment')
    lines.append('')
    if not weak_peaks:
        lines.append('All 8 prompts show Pr[SSA(w=7.5) > SSA(w)] ≥ 0.95 for every ')
        lines.append('comparison scale w ∈ {1.0, 3.0, 15.0}. The claim "SSA peaks ')
        lines.append('at w=7.5 for all 8 prompts" is robust under lag bootstrap.')
    else:
        lines.append('The following comparisons have Pr < 0.95:')
        lines.append('')
        for wp in weak_peaks:
            lines.append(f'- **{wp["prompt"]}**: SSA(7.5) vs SSA({wp["w_compare"]}), '
                         f'Pr = {wp["prob_w75_greater"]:.3f}, '
                         f'Δ = {wp["delta_ssa"]:+.4f}')
        lines.append('')
        lines.append('For these prompt(s), the paper should soften language from ')
        lines.append('"peaks at w=7.5" to "is highest at w=7.5, though the ')
        lines.append('difference is not statistically significant at the 95% level."')

    summary_path = os.path.join(EXPDIR, 'SUMMARY.md')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nSaved: {summary_path}')


if __name__ == '__main__':
    main()
