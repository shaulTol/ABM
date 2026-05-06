#!/usr/bin/env python3
"""
Diffusion maps baseline on alanine dipeptide (45D heavy-atom distances).

Compares with DA/TICA: does DM recover ψ₂↔ψ (slow), ψ₃↔φ (fast)?

Uses α=1 normalization (Coifman-Lafon), Gaussian kernel, log-log bandwidth
selection. Reports correlations with ground-truth dihedrals across 5 seeds
and 3 bandwidth values {ε*/2, ε*, 2ε*}.
"""
import os, sys, json
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr

EXPDIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(EXPDIR, 'results')
os.makedirs(RESDIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                        'da_directions', 'alanine_dipeptide', 'data')


def load_alanine():
    """Load alanine dipeptide: 45D heavy-atom distances + dihedrals."""
    dist_file = os.path.join(DATA_DIR,
                             'alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
    dihed_file = os.path.join(DATA_DIR,
                              'alanine-dipeptide-3x250ns-backbone-dihedrals.npz')
    dist_data = np.load(dist_file)
    dihed_data = np.load(dihed_file)
    all_dists = np.concatenate([dist_data[f'arr_{i}'] for i in range(3)])
    all_diheds = np.concatenate([dihed_data[f'arr_{i}'] for i in range(3)])
    return all_dists, all_diheds[:, 0], all_diheds[:, 1]


def bandwidth_selection(D_sq, n_eps=50, eps_range=(1e-4, 1e2)):
    """Log-log criterion for bandwidth selection.

    Compute S(ε) = Σ_ij K_ε(x_i, x_j) on a log-spaced grid,
    find the midpoint of the central linear regime in log S vs log ε.
    """
    log_eps = np.linspace(np.log10(eps_range[0]), np.log10(eps_range[1]), n_eps)
    eps_grid = 10.0 ** log_eps
    log_S = np.zeros(n_eps)

    for k, eps in enumerate(eps_grid):
        K = np.exp(-D_sq / eps)
        log_S[k] = np.log10(K.sum())

    # Finite differences → local slope
    dlog_S = np.diff(log_S) / np.diff(log_eps)

    # Linear regime: slope closest to d/2 (expected for Gaussian in d dims)
    # but more robustly: find the plateau in the slope curve
    # Use the region where slope is between 10th and 90th percentile of the
    # middle 60% of slopes (exclude extremes)
    mid = len(dlog_S) // 5
    end = len(dlog_S) - mid
    mid_slopes = dlog_S[mid:end]
    median_slope = np.median(mid_slopes)

    # Find indices where slope is within 20% of median
    tol = 0.2 * abs(median_slope) + 0.1
    linear_mask = np.abs(dlog_S - median_slope) < tol
    linear_indices = np.where(linear_mask)[0]

    if len(linear_indices) > 2:
        mid_idx = linear_indices[len(linear_indices) // 2]
    else:
        mid_idx = len(eps_grid) // 2

    eps_star = eps_grid[mid_idx]

    return eps_star, eps_grid, log_S, log_eps


def diffusion_maps(X, eps, alpha=1.0, n_eigs=6):
    """Diffusion maps with α-normalization.

    Returns eigenvalues and eigenvectors of the diffusion operator.
    Uses the symmetric conjugate for numerical stability.
    """
    N = X.shape[0]
    D_sq = squareform(pdist(X, 'sqeuclidean'))

    # Gaussian kernel
    K = np.exp(-D_sq / eps)

    # α-normalization (Coifman-Lafon)
    q = K.sum(axis=1)  # degree
    K_alpha = K / np.outer(q ** alpha, q ** alpha)

    # Row-normalize to get Markov matrix P
    d = K_alpha.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d)

    # Symmetric conjugate: M = D^{-1/2} K_α D^{-1/2}
    M = K_alpha * np.outer(d_inv_sqrt, d_inv_sqrt)

    # Eigendecompose (largest eigenvalues)
    eigenvalues, eigenvectors = eigsh(M, k=n_eigs, which='LM')

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Convert back from symmetric conjugate: ψ_k = D^{-1/2} φ_k
    psi = eigenvectors * d_inv_sqrt[:, None]

    # Normalize eigenvectors
    for k in range(psi.shape[1]):
        psi[:, k] /= np.linalg.norm(psi[:, k])

    return eigenvalues, psi


def run_single(all_dists, gt_phi, gt_psi, seed, eps_star):
    """Run diffusion maps for one seed and three bandwidths."""
    rng = np.random.RandomState(seed)
    N_total = len(all_dists)
    idx = rng.choice(N_total, 2000, replace=False)
    X = all_dists[idx]
    phi_sub = gt_phi[idx]
    psi_sub = gt_psi[idx]

    results = {}
    for eps_label, eps in [('eps_half', eps_star / 2),
                           ('eps_star', eps_star),
                           ('eps_double', eps_star * 2)]:
        evals, psi_vecs = diffusion_maps(X, eps, alpha=1.0, n_eigs=10)

        # Sanity: λ₁ ≈ 1, ψ₁ ≈ const
        psi1_std = np.std(psi_vecs[:, 0])

        # Correlations for ψ₂, ψ₃, ψ₄ with dihedrals
        corrs = {}
        for k in range(1, 5):  # ψ₂ through ψ₅ (0-indexed: 1-4)
            proj = psi_vecs[:, k]
            r_phi = abs(pearsonr(proj, phi_sub)[0])
            r_psi = abs(pearsonr(proj, psi_sub)[0])
            corrs[f'psi_{k+1}'] = {
                'r_phi': round(float(r_phi), 4),
                'r_psi': round(float(r_psi), 4),
            }

        ordering_correct = (corrs['psi_2']['r_psi'] > corrs['psi_2']['r_phi']
                           and corrs['psi_3']['r_phi'] > corrs['psi_3']['r_psi'])

        results[eps_label] = {
            'eps': float(eps),
            'eigenvalues': [round(float(e), 6) for e in evals],
            'psi1_std': round(float(psi1_std), 6),
            'correlations': corrs,
            'ordering_psi2_psi_psi3_phi': ordering_correct,
        }

    return results


def main():
    print('Loading alanine dipeptide data...')
    all_dists, gt_phi, gt_psi = load_alanine()
    N_total = all_dists.shape[0]
    print(f'  {N_total} frames, {all_dists.shape[1]}D')

    # Bandwidth selection on a representative subsample
    print('Bandwidth selection (N=2000, seed=42)...')
    rng = np.random.RandomState(42)
    idx_bw = rng.choice(N_total, 2000, replace=False)
    X_bw = all_dists[idx_bw]
    D_sq_bw = squareform(pdist(X_bw, 'sqeuclidean'))
    eps_star, eps_grid, log_S, log_eps = bandwidth_selection(D_sq_bw)
    print(f'  ε* = {eps_star:.4f}')

    # Bandwidth selection plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(log_eps, log_S, 'b-', lw=1.5)
    ax.axvline(np.log10(eps_star), color='r', ls='--', lw=1.2,
               label=fr'$\epsilon^* = {eps_star:.3f}$')
    ax.set_xlabel(r'$\log_{10} \epsilon$')
    ax.set_ylabel(r'$\log_{10} S(\epsilon)$')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESDIR, 'bandwidth_selection.png'), dpi=300)
    print(f'  Saved bandwidth_selection.png')
    plt.close(fig)

    # Run across 5 seeds
    seeds = [42, 123, 456, 789, 1024]
    all_results = {
        'eps_star': float(eps_star),
        'seeds': seeds,
        'per_seed': {},
    }

    for seed in seeds:
        print(f'  Seed {seed}...')
        res = run_single(all_dists, gt_phi, gt_psi, seed, eps_star)
        all_results['per_seed'][str(seed)] = res

    # Aggregate across seeds
    print('\nAggregating across seeds...')
    for eps_label in ['eps_half', 'eps_star', 'eps_double']:
        eps_val = all_results['per_seed']['42'][eps_label]['eps']
        print(f'\n  ε = {eps_val:.4f} ({eps_label}):')
        print(f'    {"":8} {"r_phi":>8} {"r_psi":>8}')

        agg = {}
        for k_name in ['psi_2', 'psi_3', 'psi_4', 'psi_5']:
            r_phi_vals = []
            r_psi_vals = []
            for seed in seeds:
                c = all_results['per_seed'][str(seed)][eps_label]['correlations']
                if k_name in c:
                    r_phi_vals.append(c[k_name]['r_phi'])
                    r_psi_vals.append(c[k_name]['r_psi'])

            if r_phi_vals:
                mean_phi = np.mean(r_phi_vals)
                std_phi = np.std(r_phi_vals)
                mean_psi = np.mean(r_psi_vals)
                std_psi = np.std(r_psi_vals)
                print(f'    {k_name:8} {mean_phi:.3f}±{std_phi:.3f} {mean_psi:.3f}±{std_psi:.3f}')
                agg[k_name] = {
                    'r_phi_mean': round(float(mean_phi), 4),
                    'r_phi_std': round(float(std_phi), 4),
                    'r_psi_mean': round(float(mean_psi), 4),
                    'r_psi_std': round(float(std_psi), 4),
                }

        all_results.setdefault('aggregate', {})[eps_label] = agg

    # Eigenvalue spectrum plot (seed=42, ε*)
    print('\nGenerating figures...')
    res42 = all_results['per_seed']['42']['eps_star']
    evals = res42['eigenvalues']

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogy(range(1, len(evals) + 1), evals, 'ko-', markersize=5)
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel(r'$\lambda_k$')
    ax.set_xticks(range(1, len(evals) + 1))
    fig.tight_layout()
    fig.savefig(os.path.join(RESDIR, 'eigenvalue_spectrum.png'), dpi=300)
    plt.close(fig)
    print('  Saved eigenvalue_spectrum.png')

    # Ramachandran colored by ψ₂ and ψ₃
    rng42 = np.random.RandomState(42)
    idx42 = rng42.choice(len(all_dists), 2000, replace=False)
    X42 = all_dists[idx42]
    phi42 = np.degrees(gt_phi[idx42])
    psi42 = np.degrees(gt_psi[idx42])
    _, psi_vecs42 = diffusion_maps(X42, eps_star, alpha=1.0, n_eigs=6)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    for ax_i, (k, label) in enumerate([(1, r'$\psi_2$ (DM)'),
                                        (2, r'$\psi_3$ (DM)'),
                                        (3, r'$\psi_4$ (DM)')]):
        ax = axes[ax_i]
        sc = ax.scatter(phi42, psi42, c=psi_vecs42[:, k], cmap='coolwarm',
                        s=3, alpha=0.7, rasterized=True)
        ax.set_xlabel(r'$\phi$ (deg)')
        if ax_i == 0:
            ax.set_ylabel(r'$\psi$ (deg)')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        r_phi = abs(pearsonr(psi_vecs42[:, k], gt_phi[idx42])[0])
        r_psi = abs(pearsonr(psi_vecs42[:, k], gt_psi[idx42])[0])
        ax.set_title(f'{label}\n|r(φ)|={r_phi:.2f}, |r(ψ)|={r_psi:.2f}',
                     fontsize=9)
        plt.colorbar(sc, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(RESDIR, 'ramachandran_dm.png'), dpi=300)
    plt.close(fig)
    print('  Saved ramachandran_dm.png')

    # Save results
    out_path = os.path.join(RESDIR, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved: {out_path}')
    print('DONE')


if __name__ == '__main__':
    main()
