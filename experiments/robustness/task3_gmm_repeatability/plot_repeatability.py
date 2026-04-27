#!/usr/bin/env python3
"""
Plot GMM sweep curves with ±1 SD shaded bands from repeatability analysis.

Matches the paper figure style: 1×4 row of panels (Separation, Variance,
Weight, Mode count). Blue SSA with shaded ±1 SD band, red MI on twin axis.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPDIR = os.path.dirname(os.path.abspath(__file__))

# Match paper figure panel order and titles
SWEEP_ORDER = ['sep_2d', 'var_2d', 'weight_2d', 'nmodes_fixdist_2d']

TITLES = {
    'sep_2d': 'Separation',
    'var_2d': 'Variance',
    'weight_2d': 'Weight',
    'nmodes_fixdist_2d': 'Mode count',
}

XLABELS = {
    'sep_2d': 'Separation',
    'var_2d': 'Variance',
    'weight_2d': r'Weight $\pi_1$',
    'nmodes_fixdist_2d': 'Number of modes',
}

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
})


def main():
    # Load repeatability results
    with open(os.path.join(EXPDIR, 'results_gmm_repeatability.json')) as f:
        results = json.load(f)


    # ── Main figure: 1×4 row matching paper layout ──
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))

    for i, name in enumerate(SWEEP_ORDER):
        ax = axes[i]
        r = results[name]
        params = np.array(r['param_values'])
        ssa_mean = np.array(r['ssa_mean'])
        ssa_sd = np.array(r['ssa_sd'])
        mi_mean = np.array(r['mi_mean'])

        # SSA mean line + ±1 SD band
        ax.plot(params, ssa_mean, '-', color='tab:blue', linewidth=2,
                label='SSA', zorder=3)
        ax.fill_between(params, ssa_mean - ssa_sd, ssa_mean + ssa_sd,
                        color='tab:blue', alpha=0.25, label=r'$\pm 1$ SD',
                        zorder=2)


        ax.set_xlabel(XLABELS[name])
        if i == 0:
            ax.set_ylabel('SSA', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')

        # MI on twin axis
        ax2 = ax.twinx()
        ax2.plot(params, mi_mean, 's--', color='tab:red', markersize=3,
                 linewidth=1.5, label='MI', zorder=3)
        if i == len(SWEEP_ORDER) - 1:
            ax2.set_ylabel('MI (nats)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        ax.set_title(TITLES[name], fontweight='bold')

        # Legend only on first panel
        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      loc='center right', fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(EXPDIR, 'fig_gmm_repeatability_bands.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')

    # ── CV profile ──
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for name in SWEEP_ORDER:
        r = results[name]
        params = np.array(r['param_values'])
        cvs = np.array(r['ssa_cv']) * 100
        ax.plot(params / params.max(), cvs, 'o-', markersize=5,
                label=f'{TITLES[name]} (max CV={cvs.max():.1f}%)')

    ax.set_xlabel('Normalized parameter value')
    ax.set_ylabel('CV (%)')
    ax.set_title('Coefficient of Variation across 10 seeds')
    ax.axhline(5, ls='--', color='gray', alpha=0.5, label='5% threshold')
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out_path2 = os.path.join(EXPDIR, 'fig_gmm_cv_profile.png')
    fig.savefig(out_path2, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path2}')


if __name__ == '__main__':
    main()
