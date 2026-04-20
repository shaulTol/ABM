#!/usr/bin/env python3
"""
Plot Δt-robustness figure: overlay dt=0.01 and dt=0.005 SSA curves with ±1 SD bands.
4-panel layout matching fig_gmm_sweeps_curves.png.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPDIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(EXPDIR, '..', '..', 'figures')

SWEEP_ORDER = ['sep_2d', 'var_2d', 'weight_2d', 'nmodes_fixdist_2d']
XLABELS = {
    'sep_2d': 'Separation',
    'var_2d': 'Variance',
    'weight_2d': 'Weight',
    'nmodes_fixdist_2d': 'Number of modes',
}


def main():
    # Load dt=0.01 results (from repeatability run)
    ref_path = os.path.join(EXPDIR, '..', 'robustness', 'task3_gmm_repeatability',
                            'results_gmm_repeatability.json')
    with open(ref_path) as f:
        ref = json.load(f)

    # Load dt=0.005 results
    new_path = os.path.join(EXPDIR, 'results_dt_0.005.json')
    with open(new_path) as f:
        new = json.load(f)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))

    for i, name in enumerate(SWEEP_ORDER):
        ax = axes[i]

        # dt=0.01
        r_ref = ref[name]
        p_ref = np.array(r_ref['param_values'])
        ssa_ref = np.array(r_ref['ssa_mean'])
        sd_ref = np.array(r_ref['ssa_sd'])

        ax.plot(p_ref, ssa_ref, '-', color='tab:blue', linewidth=2,
                label=r'$\Delta t = 0.01$')
        ax.fill_between(p_ref, ssa_ref - sd_ref, ssa_ref + sd_ref,
                        color='tab:blue', alpha=0.2)

        # dt=0.005
        r_new = new[name]
        p_new = np.array(r_new['param_values'])
        ssa_new = np.array(r_new['ssa_mean'])
        sd_new = np.array(r_new['ssa_sd'])

        ax.plot(p_new, ssa_new, '--', color='tab:orange', linewidth=2,
                label=r'$\Delta t = 0.005$')
        ax.fill_between(p_new, ssa_new - sd_new, ssa_new + sd_new,
                        color='tab:orange', alpha=0.2)

        ax.set_xlabel(XLABELS[name], fontsize=13)
        if i == 0:
            ax.set_ylabel('SSA', fontsize=13)
        ax.tick_params(labelsize=9)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        if i == 0:
            ax.legend(fontsize=12, loc='lower right', framealpha=0.8)

    plt.tight_layout(pad=0.5)
    out_path = os.path.join(FIGDIR, 'fig_gmm_sweeps_dt_robustness.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_path}')

    # Also save in experiment dir
    out_path2 = os.path.join(EXPDIR, 'fig_dt_robustness.png')
    fig2, axes2 = plt.subplots(1, 4, figsize=(14, 4.2))
    # Regenerate (quick hack: just copy the file)
    import shutil
    shutil.copy(out_path, out_path2)
    print(f'Saved: {out_path2}')


if __name__ == '__main__':
    main()
