"""Sparse K-means baseline for alanine (Witten and Tibshirani 2010).

Steps:
  1. Load 45-d heavy-atom-distance features and dihedrals.
  2. Subsample N_fit frames for tractable repeated fitting.
  3. Select s by gap statistic on a 10-point grid in [1.1, sqrt(45)].
  4. Final fit at s_star; project all 750k frames; compute |r(psi)|, |r(phi)|.
  5. Bootstrap B=5000 frame-level resamples (refit each time) for SE.
  6. ARI sanity check across 3 random inits at s_star.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import load_alanine
from sparse_kmeans import sparse_kmeans, gap_statistic_select_s

EXPDIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(EXPDIR, 'results')
os.makedirs(RESDIR, exist_ok=True)


def project(X, w, mean):
    direction = w / max(np.linalg.norm(w), 1e-12)
    return (X - mean) @ direction, direction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_fit', type=int, default=5000,
                        help='subsample size for sparse K-means fitting')
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--n_grid', type=int, default=10)
    parser.add_argument('--max_nnz', type=int, default=15,
                        help='hard sparsity ceiling; if 1-SE pick exceeds it, '
                             'fall back to smallest grid point with nnz <= max_nnz')
    parser.add_argument('--gap_B', type=int, default=20)
    parser.add_argument('--boot_B', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_npz', type=str,
                        default=os.path.join(RESDIR, 'sparse_kmeans.npz'))
    parser.add_argument('--out_json', type=str,
                        default=os.path.join(RESDIR, 'sparse_kmeans.json'))
    args = parser.parse_args()

    print('=' * 70)
    print('Sparse K-means baseline (Witten and Tibshirani 2010)')
    print('=' * 70)

    print('\n[1/6] Loading data...')
    X, phi, psi = load_alanine()
    N, p = X.shape
    print(f'  X={X.shape}, phi/psi range [-pi, pi]')

    print(f'\n[2/6] Subsampling N_fit={args.n_fit}...')
    rng = np.random.RandomState(args.seed)
    sub = rng.choice(N, args.n_fit, replace=False)
    Xs = X[sub]
    Xs_mean = Xs.mean(axis=0)
    Xs_centered = Xs - Xs_mean

    print(f'\n[3/6] Selecting s via gap statistic (grid={args.n_grid}, B={args.gap_B})...')
    s_grid = np.geomspace(1.1, np.sqrt(p), args.n_grid)
    t0 = time.time()
    sel = gap_statistic_select_s(
        Xs_centered, K=args.K, s_grid=s_grid, B=args.gap_B,
        seed=args.seed + 1, verbose=True, rule='1se')
    gap = sel['gap']
    sds = sel['sds']
    obj_obs = sel['obj_obs']
    obj_ref = sel['obj_ref_mean']
    s_1se = sel['s_star']
    print(f'  gap profile: ' + ', '.join(f'{g:.3f}' for g in gap))
    print(f'  sds:         ' + ', '.join(f'{s:.3f}' for s in sds))
    print(f'  argmax gap = {sel["i_max"]} (s={s_grid[sel["i_max"]]:.3f}), '
          f'1-SE choice = {sel["i_star"]} (s={s_1se:.3f})')

    # Sparsity ceiling: count nnz at each grid point; if 1-SE pick exceeds
    # max_nnz, fall back to the smallest s with nnz <= max_nnz, per task brief.
    print(f'  Counting nnz across grid (sparsity-ceiling check, max_nnz={args.max_nnz})...')
    nnz_grid = np.zeros(len(s_grid), dtype=int)
    rng_chk = np.random.RandomState(args.seed + 7)
    for i, s in enumerate(s_grid):
        w_chk, _, _ = sparse_kmeans(Xs_centered, K=args.K, s=s,
                                    max_iter=15, n_init=3,
                                    seed=int(rng_chk.randint(2**31 - 1)))
        nnz_grid[i] = int((w_chk > 1e-8).sum())
    print(f'  nnz at each grid s: {list(nnz_grid)}')

    i_1se = sel['i_star']
    if nnz_grid[i_1se] <= args.max_nnz:
        i_star_final = i_1se
        sparsity_rule = '1se'
    else:
        eligible = np.where(nnz_grid <= args.max_nnz)[0]
        if len(eligible) == 0:
            i_star_final = i_1se
            sparsity_rule = '1se_fallback_no_eligible'
            print(f'  WARN: no s with nnz<={args.max_nnz}; using 1-SE pick')
        else:
            i_star_final = int(eligible.max())  # densest sparse pick
            sparsity_rule = '1se_then_max_eligible_under_ceiling'
            print(f'  1-SE pick has nnz={nnz_grid[i_1se]} > {args.max_nnz}; '
                  f'falling back to s={s_grid[i_star_final]:.3f} '
                  f'(largest s with nnz<={args.max_nnz}, nnz={nnz_grid[i_star_final]})')
    s_star = float(s_grid[i_star_final])
    print(f'  s_star = {s_star:.3f} (rule: {sparsity_rule})')
    print(f'  [{time.time() - t0:.1f}s]')

    print(f'\n[4/6] Final fit at s_star={s_star:.3f}...')
    w, labels, hist = sparse_kmeans(Xs_centered, K=args.K, s=s_star,
                                    max_iter=20, n_init=5,
                                    seed=args.seed + 100)
    nnz = int((w > 1e-8).sum())
    print(f'  nnz(w) = {nnz}, top weights: '
          f'{sorted([float(x) for x in w], reverse=True)[:10]}')

    # ARI sanity check across inits
    aris = []
    labels_runs = [labels]
    for r in range(3):
        _, lab_r, _ = sparse_kmeans(Xs_centered, K=args.K, s=s_star,
                                    max_iter=20, n_init=5,
                                    seed=args.seed + 200 + r)
        labels_runs.append(lab_r)
    for i in range(len(labels_runs)):
        for j in range(i + 1, len(labels_runs)):
            aris.append(adjusted_rand_score(labels_runs[i], labels_runs[j]))
    print(f'  ARI across 4 inits (pairwise mean): {np.mean(aris):.3f}, min: {np.min(aris):.3f}')

    # Project all 750k onto w-direction (unit-L2 normalized)
    direction = w / max(np.linalg.norm(w), 1e-12)
    score_all = (X - Xs_mean) @ direction
    r_psi = float(abs(pearsonr(score_all, psi)[0]))
    r_phi = float(abs(pearsonr(score_all, phi)[0]))
    print(f'\n  Point estimate: |r(psi)|={r_psi:.4f}, |r(phi)|={r_phi:.4f}')

    print(f'\n[5/6] Bootstrap (B={args.boot_B}, refit each rep)...')
    rs_psi = np.zeros(args.boot_B)
    rs_phi = np.zeros(args.boot_B)
    rng_b = np.random.RandomState(args.seed + 9999)
    t0 = time.time()
    for b in range(args.boot_B):
        idx_b = rng_b.choice(N, args.n_fit, replace=True)
        Xb = X[idx_b]
        Xb_centered = Xb - Xb.mean(axis=0)
        w_b, _, _ = sparse_kmeans(Xb_centered, K=args.K, s=s_star,
                                  max_iter=15, n_init=1,
                                  seed=int(rng_b.randint(2**31 - 1)))
        d_b = w_b / max(np.linalg.norm(w_b), 1e-12)
        score_b = (X[idx_b] - X[idx_b].mean(axis=0)) @ d_b
        rs_psi[b] = abs(pearsonr(score_b, psi[idx_b])[0])
        rs_phi[b] = abs(pearsonr(score_b, phi[idx_b])[0])
        if (b + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (b + 1) * (args.boot_B - b - 1)
            print(f'  {b+1}/{args.boot_B}  '
                  f'r(psi) mean={rs_psi[:b+1].mean():.3f}  '
                  f'r(phi) mean={rs_phi[:b+1].mean():.3f}  '
                  f'ETA {eta/60:.1f} min')
    print(f'  [{(time.time() - t0)/60:.1f} min]')

    se_psi = float(rs_psi.std(ddof=1))
    se_phi = float(rs_phi.std(ddof=1))
    print(f'\n  |r(psi)| = {r_psi:.4f} +/- {se_psi:.4f}')
    print(f'  |r(phi)| = {r_phi:.4f} +/- {se_phi:.4f}')

    print(f'\n[6/6] Saving results...')
    np.savez_compressed(
        args.out_npz,
        w=w, labels=labels, direction=direction,
        s_grid=s_grid, gap=gap, obj_obs=obj_obs, obj_ref=obj_ref,
        rs_psi=rs_psi, rs_phi=rs_phi,
        score_all=score_all.astype(np.float32),
        Xs_mean=Xs_mean,
    )
    payload = {
        'method': 'sparse_kmeans',
        'reference': 'Witten and Tibshirani 2010, JASA 105(490):713-726',
        'K': args.K,
        'n_fit': args.n_fit,
        's_grid': [float(x) for x in s_grid],
        'gap': [float(x) for x in gap],
        'gap_sds': [float(x) for x in sds],
        'nnz_per_grid_s': [int(x) for x in nnz_grid],
        'sparsity_rule': sparsity_rule,
        's_1se': float(s_1se),
        's_selected': float(s_star),
        'n_nonzero_weights': nnz,
        'ari_across_inits_mean': float(np.mean(aris)),
        'ari_across_inits_min': float(np.min(aris)),
        'r_psi': r_psi, 'r_psi_se': se_psi,
        'r_phi': r_phi, 'r_phi_se': se_phi,
        'weight_vector': [float(x) for x in w],
        'direction_unit_l2': [float(x) for x in direction],
        'bootstrap_B': args.boot_B,
        'bootstrap_note': 'Each replicate refits sparse K-means on a fresh '
                          'frame-level resample of size n_fit; correlation '
                          'computed on the resampled frames.',
    }
    with open(args.out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'  saved {args.out_npz}')
    print(f'  saved {args.out_json}')
    print('\nDONE')


if __name__ == '__main__':
    main()
