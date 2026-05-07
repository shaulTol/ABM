"""Spectral Map baseline for alanine.

Reference implementation: PLUMED-NEST plumID:24.005, file `spectral-map.py`,
Zenodo record 10678142 (plumID:24.005).
This script imports and uses that reference implementation directly; it is
saved alongside this file as `spectral_map_reference.py`.

Default architecture and hyperparameters are taken verbatim from the reference
script's __main__ block (alanine has 45 input features):
    layers = [45, 200, 100, 50, 2]
    activation = ELU
    optimizer = Adam, lr=1e-3, maximize=True (maximizes spectral gap)
    scheduler = LinearLR(start=1.0, end=0.1, iters=50)
    n_eigen=50, n_gap=2, alpha=0.5, n_neigh=0.5, batch_size=1000
    n_epochs=100

Bootstrap protocol: model fixed (refitting a NN 5000 times is intractable).
Frame-level bootstrap on the (SM_k, dihedral) correlation only.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import load_alanine
from spectral_map_reference import SpectralMap

EXPDIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(EXPDIR, 'results')
os.makedirs(RESDIR, exist_ok=True)

REF_VERSION = 'PLUMED-NEST plumID:24.005 (Zenodo 10678142, deposited 2024-02-19)'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_fit', type=int, default=50000,
                        help='subsample size for spectral map training')
    parser.add_argument('--valid_frac', type=float, default=0.2)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--n_neigh', type=float, default=0.5)
    parser.add_argument('--n_eigen', type=int, default=50)
    parser.add_argument('--n_gap', type=int, default=2)
    parser.add_argument('--boot_B', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--out_npz', type=str,
                        default=os.path.join(RESDIR, 'spectral_map.npz'))
    parser.add_argument('--out_json', type=str,
                        default=os.path.join(RESDIR, 'spectral_map.json'))
    args = parser.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f'device: {device}')

    print('=' * 70)
    print('Spectral Map baseline (spectral map (plumID:24.005))')
    print(f'reference: {REF_VERSION}')
    print('=' * 70)

    print('\n[1/5] Loading data...')
    X, phi, psi = load_alanine()
    N, p = X.shape
    print(f'  X={X.shape}')

    print(f'\n[2/5] Subsampling N_fit={args.n_fit}, valid_frac={args.valid_frac}...')
    rng = np.random.RandomState(args.seed)
    sub = rng.choice(N, args.n_fit, replace=False)
    Xs = X[sub].astype(np.float32)
    n_valid = int(args.valid_frac * args.n_fit)
    perm = rng.permutation(args.n_fit)
    valid_idx = perm[:n_valid]
    train_idx = perm[n_valid:]
    X_train = torch.from_numpy(Xs[train_idx]).to(device)
    X_valid = torch.from_numpy(Xs[valid_idx]).to(device)
    print(f'  train: {X_train.shape}, valid: {X_valid.shape}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f'\n[3/5] Building Spectral Map model...')
    model = SpectralMap(
        n_eigen=args.n_eigen,
        n_gap=args.n_gap,
        n_neigh=args.n_neigh,
        batch_size=args.batch_size,
        alpha=args.alpha,
        layers=[p, 200, 100, 50, 2],
    )
    model = model.to(device)

    print(f'\n[4/5] Training (n_epochs={args.n_epochs}, lr={args.lr}, '
          f'batch_size={args.batch_size})...')
    t0 = time.time()
    model.fit(
        X_train, X_valid,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        n_stride=10,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - t0
    print(f'  train time: {elapsed/60:.1f} min')
    train_score = model.train_score
    valid_score = model.valid_score
    print(f'  train spectral gap: '
          f'{float(train_score[0]):.4f} -> {float(train_score[-1]):.4f}')
    print(f'  valid spectral gap: '
          f'{float(valid_score[0]):.4f} -> {float(valid_score[-1]):.4f}')
    if float(train_score[-1]) <= float(train_score[0]):
        print('  WARN: training spectral gap did not increase')

    # Embed all 750k frames in chunks
    print(f'\n[4b] Embedding all {N:,} frames...')
    model.eval()
    embed = np.zeros((N, 2), dtype=np.float32)
    chunk = 50000
    with torch.no_grad():
        for i0 in range(0, N, chunk):
            i1 = min(i0 + chunk, N)
            xb = torch.from_numpy(X[i0:i1].astype(np.float32)).to(device)
            yb = model(xb).cpu().numpy()
            embed[i0:i1] = yb
    print(f'  embed shape: {embed.shape}, '
          f'SM1 var={embed[:,0].var():.4e}, SM2 var={embed[:,1].var():.4e}')
    if embed[:, 0].var() < 1e-4:
        print('  WARN: SM1 has trivial variance')

    # Pearson correlations
    sm1 = embed[:, 0]
    sm2 = embed[:, 1]
    r_sm1_psi = float(abs(pearsonr(sm1, psi)[0]))
    r_sm1_phi = float(abs(pearsonr(sm1, phi)[0]))
    r_sm2_psi = float(abs(pearsonr(sm2, psi)[0]))
    r_sm2_phi = float(abs(pearsonr(sm2, phi)[0]))
    print(f'\n  SM1: |r(psi)|={r_sm1_psi:.4f}, |r(phi)|={r_sm1_phi:.4f}')
    print(f'  SM2: |r(psi)|={r_sm2_psi:.4f}, |r(phi)|={r_sm2_phi:.4f}')

    print(f'\n[5/5] Bootstrap (B={args.boot_B}, model fixed, frame-level resamples)...')
    rng_b = np.random.RandomState(args.seed + 9999)
    rs1_psi = np.zeros(args.boot_B); rs1_phi = np.zeros(args.boot_B)
    rs2_psi = np.zeros(args.boot_B); rs2_phi = np.zeros(args.boot_B)
    t0 = time.time()
    for b in range(args.boot_B):
        idx = rng_b.choice(N, N, replace=True)
        rs1_psi[b] = abs(pearsonr(sm1[idx], psi[idx])[0])
        rs1_phi[b] = abs(pearsonr(sm1[idx], phi[idx])[0])
        rs2_psi[b] = abs(pearsonr(sm2[idx], psi[idx])[0])
        rs2_phi[b] = abs(pearsonr(sm2[idx], phi[idx])[0])
        if (b + 1) % 1000 == 0:
            print(f'  {b+1}/{args.boot_B}  '
                  f'elapsed {(time.time()-t0)/60:.1f} min')
    se = lambda v: float(v.std(ddof=1))
    print(f'  SM1 |r(psi)|={r_sm1_psi:.4f} +/- {se(rs1_psi):.4f}')
    print(f'  SM1 |r(phi)|={r_sm1_phi:.4f} +/- {se(rs1_phi):.4f}')
    print(f'  SM2 |r(psi)|={r_sm2_psi:.4f} +/- {se(rs2_psi):.4f}')
    print(f'  SM2 |r(phi)|={r_sm2_phi:.4f} +/- {se(rs2_phi):.4f}')

    np.savez_compressed(
        args.out_npz,
        embed=embed,
        train_score=train_score,
        valid_score=valid_score,
        rs1_psi=rs1_psi, rs1_phi=rs1_phi,
        rs2_psi=rs2_psi, rs2_phi=rs2_phi,
    )
    payload = {
        'method': 'spectral_map',
        'reference': 'spectral map (plumID:24.005), spectral map',
        'implementation_version': REF_VERSION,
        'embedding_dim': 2,
        'architecture': [p, 200, 100, 50, 2],
        'activation': 'ELU',
        'optimizer': 'Adam (maximize=True)',
        'lr': args.lr,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'alpha': args.alpha,
        'n_neigh': args.n_neigh,
        'n_eigen': args.n_eigen,
        'n_gap': args.n_gap,
        'n_fit': args.n_fit,
        'valid_frac': args.valid_frac,
        'final_train_gap': float(train_score[-1]),
        'final_valid_gap': float(valid_score[-1]),
        'sm1_var': float(sm1.var()),
        'sm2_var': float(sm2.var()),
        'SM1': {
            'r_psi': r_sm1_psi, 'r_psi_se': se(rs1_psi),
            'r_phi': r_sm1_phi, 'r_phi_se': se(rs1_phi),
        },
        'SM2': {
            'r_psi': r_sm2_psi, 'r_psi_se': se(rs2_psi),
            'r_phi': r_sm2_phi, 'r_phi_se': se(rs2_phi),
        },
        'bootstrap_B': args.boot_B,
        'bootstrap_note': 'Model fixed; correlations bootstrapped over frames '
                          '(refitting a 4-layer NN 5000 times is intractable).',
    }
    with open(args.out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'  saved {args.out_npz}')
    print(f'  saved {args.out_json}')
    print('\nDONE')


if __name__ == '__main__':
    main()
