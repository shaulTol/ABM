"""Tasks 3-5: SSA SE bootstrap, paired peak prob bootstrap, rho^2(T).

For each of the 12 cells in {cat, elderly, person} x {1, 3, 7.5, 15}:
  - extract rho^2 at fixed20_lags from rho_sq_all (index-mapping via snapshot_lags_all)
  - lag bootstrap: resample 100 lag indices with replacement; recompute
    Shat = delta * sum(rho^2 at sampled indices); B=10000 reps; SE = std
  - paired bootstrap (per prompt): same lag indices applied to all 4 w-traces,
    compute Pr[Shat_w7.5 > Shat_w]
  - rho^2(T=2000) = rho_sq_all[-1]

Sanity:
  delta * sum(rho^2 at fixed20) ~= ssa_fixed20 from JSON. Tolerance 1e-6.
"""
import json
import os
import sys
import numpy as np

REPO = '/Users/stolk/github/ssa-paper'
OUT_DIR = f'{REPO}/experiments/sdxl_3prompt'
CHECK_PATH = f'{OUT_DIR}/CHECK.md'

PROMPTS = [
    ('cat',     f'{REPO}/experiments/sdxl/results_multiprompt_sweep/clip_da_mp_p0_cat_gs{{}}_dt0p005.json'),
    ('elderly', f'{REPO}/experiments/sdxl/results_multiprompt_sweep/clip_da_mp_p5_elderly_gs{{}}_dt0p005.json'),
    ('person',  f'{REPO}/experiments/sdxl/results_guidance_sweep/clip_da_sweep_gs{{}}_dt0p005.json'),
]
GS_TAGS = [('1p0', 'w1'), ('3p0', 'w3'), ('7p5', 'w7p5'), ('15p0', 'w15')]
B = 10000
SEED = 20260427


def append_check(msg):
    os.makedirs(OUT_DIR, exist_ok=True)
    header = '# CHECK.md\n\n' if not os.path.exists(CHECK_PATH) else ''
    with open(CHECK_PATH, 'a') as f:
        f.write(header + msg + '\n')


def load_cell(path):
    """Return dict with rho2_fixed20, delta, ssa_fixed20, rho2_T."""
    if not os.path.exists(path):
        append_check(f'MISSING: {path}')
        return None
    j = json.load(open(path))
    snap = list(j['snapshot_lags_all'])
    fixed20 = list(j['fixed20_lags'])
    rho_all = list(j['rho_sq_all'])
    snap_to_idx = {l: i for i, l in enumerate(snap)}
    rho_fixed20 = np.array([rho_all[snap_to_idx[l]] for l in fixed20], dtype=np.float64)
    dt = float(j['dt'])
    delta = 20 * dt
    ssa_json = float(j['ssa_fixed20'])
    ssa_recomputed = delta * rho_fixed20.sum()
    if abs(ssa_recomputed - ssa_json) > 1e-6:
        append_check(f'SSA mismatch in {os.path.basename(path)}: '
                     f'json={ssa_json:.6f}, recomputed={ssa_recomputed:.6f}')
    return dict(
        rho2_fixed20=rho_fixed20,
        delta=delta,
        ssa_fixed20=ssa_json,
        rho2_T=float(rho_all[-1]),
        T=int(j['T']),
        N=int(j['n_samples']),
    )


def main():
    cells = {}
    for prompt, path_tmpl in PROMPTS:
        for gs_tag, key_tag in GS_TAGS:
            cells[f'{prompt}_{key_tag}'] = load_cell(path_tmpl.format(gs_tag))
    if any(v is None for v in cells.values()):
        print('Some cells missing — see CHECK.md', file=sys.stderr)

    # Task 5: rho^2(T)
    rho_sq_T = {k: v['rho2_T'] for k, v in cells.items() if v is not None}
    floor = 1.0 / next(v['N'] for v in cells.values() if v is not None)
    above = [k for k, v in rho_sq_T.items() if v > floor]
    expected = {'cat_w7p5', 'elderly_w7p5', 'person_w7p5', 'person_w15'}
    actual = set(above)
    if actual != expected:
        append_check(f'Task 5 sanity: rho^2(T) > 1/N pattern mismatch.\n'
                     f'  expected: {sorted(expected)}\n'
                     f'  actual:   {sorted(actual)}\n'
                     f'  floor: {floor}')

    with open(f'{OUT_DIR}/rho_sq_T.json', 'w') as f:
        json.dump(rho_sq_T, f, indent=2)
    print(f'saved rho_sq_T.json ({len(rho_sq_T)} cells); above-floor: {sorted(above)}')

    # Task 3: SSA SE via lag bootstrap
    rng = np.random.default_rng(SEED)
    n_lags = next(len(v['rho2_fixed20']) for v in cells.values() if v is not None)
    indices = rng.integers(0, n_lags, size=(B, n_lags))

    se = {}
    for k, v in cells.items():
        if v is None:
            continue
        rho = v['rho2_fixed20']
        delta = v['delta']
        boots = delta * rho[indices].sum(axis=1)  # shape (B,)
        se[k] = float(boots.std(ddof=1))
    with open(f'{OUT_DIR}/ssa_se.json', 'w') as f:
        json.dump(se, f, indent=2)
    print(f'saved ssa_se.json (B={B})')

    # Task 4: paired bootstrap. Same lag indices applied to all 4 w per prompt.
    peak_probs = {}
    for prompt, _ in PROMPTS:
        traces = {tag: cells.get(f'{prompt}_{key_tag}') for tag, (_, key_tag) in zip(['1p0','3p0','7p5','15p0'], GS_TAGS)}
        cell_w7p5 = cells.get(f'{prompt}_w7p5')
        if cell_w7p5 is None:
            continue
        delta = cell_w7p5['delta']
        rho_w7p5 = cell_w7p5['rho2_fixed20']
        for w_tag in ('w1', 'w3', 'w15'):
            cmp_cell = cells.get(f'{prompt}_{w_tag}')
            if cmp_cell is None:
                continue
            rho_other = cmp_cell['rho2_fixed20']
            ssa_w7p5_b = delta * rho_w7p5[indices].sum(axis=1)
            ssa_other_b = delta * rho_other[indices].sum(axis=1)
            prob = float((ssa_w7p5_b > ssa_other_b).mean())
            peak_probs[f'{prompt}_vs_{w_tag}'] = prob
    with open(f'{OUT_DIR}/peak_probs.json', 'w') as f:
        json.dump(peak_probs, f, indent=2)
    print(f'saved peak_probs.json (B={B}); 9 entries: {sorted(peak_probs.keys())}')

    # Task 6: verify all >= 0.95
    below = {k: v for k, v in peak_probs.items() if v < 0.95}
    if below:
        append_check(f'Task 6 sanity: paired peak probs below 0.95: {below}')
    else:
        print('Task 6 sanity: all 9 paired peak probs >= 0.95')


if __name__ == '__main__':
    main()
