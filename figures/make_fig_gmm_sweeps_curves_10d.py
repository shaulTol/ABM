#!/usr/bin/env python3
"""
4-panel GMM sweep curves for d=10, N=200 with 1/N floor.
Mean ± 1 SD bands (10 seeds). Matches style of make_fig_gmm_sweeps_split.py.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(FIG_DIR, '..', 'experiments', 'ssa_truncation_redef',
                             'results', 'gmm_10d_n200_all_sweeps.json')

FLOOR_KEY = '1N'  # use 1/N floor


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    sweep_configs = [
        ('sep_10d', 'Separation', 'Separation'),
        ('var_10d', 'Variance', 'Variance'),
        ('weight_10d', 'Weight', 'Weight'),
        ('nmodes_10d', 'Number of modes', 'Mode count'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))

    for col, (config, xlabel, title) in enumerate(sweep_configs):
        ax = axes[col]
        sw = data['sweeps'][config]
        params = np.array(sw['param_values'])
        ssa_arr = np.array(sw[f's_hat_{FLOOR_KEY}'])  # (10, n_params)
        mi_arr = np.array(sw['mi'])                    # (10, n_params)

        ssa_mean = ssa_arr.mean(axis=0)
        ssa_sd = ssa_arr.std(axis=0)
        mi_mean = mi_arr.mean(axis=0)

        # SSA mean + ±1 SD band
        ax.plot(params, ssa_mean, '-', color='tab:blue', linewidth=2, label=r'$\hat{S}$')
        ax.fill_between(params, ssa_mean - ssa_sd, ssa_mean + ssa_sd,
                         color='tab:blue', alpha=0.25, label=r'$\pm 1$ SD')
        ax.set_xlabel(xlabel, fontsize=13, labelpad=3)
        ax.tick_params(axis='x', labelsize=9, pad=1)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')

        if col == 0:
            ax.set_ylabel(r'$\hat{S}$', color='tab:blue', fontsize=13, labelpad=3)
        ax.tick_params(axis='y', labelcolor='tab:blue', labelsize=9, pad=1)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        # MI on twin axis
        ax2 = ax.twinx()
        ax2.plot(params, mi_mean, 's--', color='tab:red', markersize=3,
                 linewidth=1.5, label='MI')
        if col == 3:
            ax2.set_ylabel('MI (nats)', color='tab:red', fontsize=13, labelpad=3)
        ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=9, pad=1)
        for label in ax2.get_yticklabels():
            label.set_fontweight('bold')

        if config == 'nmodes_10d':
            from matplotlib.ticker import MultipleLocator
            ax2.yaxis.set_major_locator(MultipleLocator(0.4))
            ax.set_xticks([1, 2, 3, 4, 5])

        if col == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right',
                      fontsize=14, framealpha=0.8, handlelength=2)

    plt.tight_layout(pad=0.5)
    for ext in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f'fig_gmm_sweeps_curves_10d.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
