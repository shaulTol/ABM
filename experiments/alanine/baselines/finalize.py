"""Merge sparse K-means and Spectral Map JSONs into the deliverable
results/results.json plus results/SUMMARY.md."""
import json
import os

EXPDIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(EXPDIR, 'results')


def main():
    sk = json.load(open(os.path.join(RESDIR, 'sparse_kmeans.json')))
    sm = json.load(open(os.path.join(RESDIR, 'spectral_map.json')))

    out = {
        'sparse_kmeans': {
            'K': sk['K'],
            's_selected': sk['s_selected'],
            'n_nonzero_weights': sk['n_nonzero_weights'],
            'r_psi': sk['r_psi'], 'r_psi_se': sk['r_psi_se'],
            'r_phi': sk['r_phi'], 'r_phi_se': sk['r_phi_se'],
            'weight_vector': sk['weight_vector'],
            'reference': sk['reference'],
            'n_fit': sk['n_fit'],
            'sparsity_rule': sk['sparsity_rule'],
            'ari_across_inits_min': sk['ari_across_inits_min'],
            'bootstrap_B': sk['bootstrap_B'],
            'bootstrap_note': sk['bootstrap_note'],
        },
        'spectral_map': {
            'implementation_version': sm['implementation_version'],
            'embedding_dim': sm['embedding_dim'],
            'architecture': sm['architecture'],
            'n_fit': sm['n_fit'],
            'n_epochs': sm['n_epochs'],
            'final_train_gap': sm['final_train_gap'],
            'final_valid_gap': sm['final_valid_gap'],
            'sm1_var': sm['sm1_var'],
            'sm2_var': sm['sm2_var'],
            'SM1': sm['SM1'],
            'SM2': sm['SM2'],
            'bootstrap_B': sm['bootstrap_B'],
            'bootstrap_note': sm['bootstrap_note'],
        },
    }
    out_path = os.path.join(RESDIR, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved {out_path}')

    md = []
    md.append('# Alanine Baselines (sparse K-means + Spectral Map)')
    md.append('')
    md.append('Featurization: 45 pairwise heavy-atom distances on the MDShare')
    md.append('alanine-dipeptide-3x250ns trajectories (N=750,000 frames).')
    md.append('Bootstrap B=5000, frame-level resamples.')
    md.append('')
    md.append('| method | top direction | |r(psi)| | |r(phi)| |')
    md.append('|---|---|---|---|')
    sk_d = out['sparse_kmeans']
    md.append(f"| sparse K-means (K={sk_d['K']}, s={sk_d['s_selected']:.3f}, "
              f"nnz={sk_d['n_nonzero_weights']}) | weight vector w* "
              f"| {sk_d['r_psi']:.3f} +/- {sk_d['r_psi_se']:.3f} "
              f"| {sk_d['r_phi']:.3f} +/- {sk_d['r_phi_se']:.3f} |")
    s1 = out['spectral_map']['SM1']
    md.append(f"| Spectral Map | SM1 (nonlinear) "
              f"| {s1['r_psi']:.3f} +/- {s1['r_psi_se']:.3f} "
              f"| {s1['r_phi']:.3f} +/- {s1['r_phi_se']:.3f} |")
    s2 = out['spectral_map']['SM2']
    md.append(f"| Spectral Map | SM2 (nonlinear) "
              f"| {s2['r_psi']:.3f} +/- {s2['r_psi_se']:.3f} "
              f"| {s2['r_phi']:.3f} +/- {s2['r_phi_se']:.3f} |")
    md.append('')
    md.append('Notes:')
    md.append('- Sparse K-means: bootstrap refits sparse K-means on each '
              'frame-level resample of size n_fit.')
    md.append('- Spectral Map: model fixed; correlations bootstrapped over '
              'frames (refitting a 4-layer NN 5000 times is intractable).')
    md.append('- SM1 and SM2 are nonlinear embedding coordinates, not linear '
              'directions in feature space.')
    md.append('- s for sparse K-means selected via the gap-statistic 1-SE rule '
              'on a 20-point grid; sparsity ceiling enforced per task brief.')
    md.append('')

    out_md = os.path.join(RESDIR, 'SUMMARY.md')
    with open(out_md, 'w') as f:
        f.write('\n'.join(md))
    print(f'saved {out_md}')


if __name__ == '__main__':
    main()
