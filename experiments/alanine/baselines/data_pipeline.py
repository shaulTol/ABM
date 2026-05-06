"""Alanine dipeptide data loader. Mirrors compute_da_comparison.load_alanine."""
import os
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'experiments', 'da_directions',
                        'alanine_dipeptide', 'data')


def load_alanine():
    dist_file = os.path.join(DATA_DIR,
                             'alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
    dihed_file = os.path.join(DATA_DIR,
                              'alanine-dipeptide-3x250ns-backbone-dihedrals.npz')
    if not (os.path.exists(dist_file) and os.path.exists(dihed_file)):
        import mdshare
        os.makedirs(DATA_DIR, exist_ok=True)
        dist_file = mdshare.fetch(
            'alanine-dipeptide-3x250ns-heavy-atom-distances.npz',
            working_directory=DATA_DIR)
        dihed_file = mdshare.fetch(
            'alanine-dipeptide-3x250ns-backbone-dihedrals.npz',
            working_directory=DATA_DIR)

    dist_data = np.load(dist_file)
    dihed_data = np.load(dihed_file)
    dists_list = [dist_data[f'arr_{i}'] for i in range(3)]
    diheds_list = [dihed_data[f'arr_{i}'] for i in range(3)]
    all_dists = np.concatenate(dists_list).astype(np.float32)
    all_diheds = np.concatenate(diheds_list).astype(np.float32)
    phi = all_diheds[:, 0]
    psi = all_diheds[:, 1]
    return all_dists, phi, psi


if __name__ == '__main__':
    from scipy.stats import pearsonr
    X, phi, psi = load_alanine()
    print(f'X shape: {X.shape}')
    print(f'phi range: [{phi.min():.3f}, {phi.max():.3f}]  '
          f'psi range: [{psi.min():.3f}, {psi.max():.3f}]')

    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / Xc.shape[0]
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx].copy()
    pc1 = Xc @ evecs[:, 0]
    r_phi = abs(pearsonr(pc1, phi)[0])
    r_psi = abs(pearsonr(pc1, psi)[0])
    print(f'PCA1: |r(phi)|={r_phi:.4f}, |r(psi)|={r_psi:.4f}')
    expected_phi, expected_psi = 0.595, 0.563
    tol = 0.005
    ok = abs(r_phi - expected_phi) < tol and abs(r_psi - expected_psi) < tol
    print(f'  expected: |r(phi)|~{expected_phi}, |r(psi)|~{expected_psi}')
    print(f'  match within {tol}: {ok}')
