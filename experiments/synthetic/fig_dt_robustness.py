#!/usr/bin/env python3
"""
Plot Δt-robustness figure: overlay dt=0.005, dt=0.01, dt=0.02 SSA curves.
4-panel layout (sep, var, weight, nmodes) in 10D, N=200, 1/N floor.

Data files (all in results/):
  gmm_10d_n200_dt_robustness.json    — dt=0.005
  gmm_10d_n200_all_sweeps_T200.json  — dt=0.01
  gmm_10d_dt_robust_dt0p02_T200.json — dt=0.02
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPDIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPDIR, 'results')

SWEEP_ORDER = ['sep_10d', 'var_10d', 'weight_10d', 'nmodes_10d']
XLABELS = {
    'sep_10d': 'Separation',
    'var_10d': 'Variance',
    'weight_10d': 'Weight',
    'nmodes_10d': 'Number of modes',
}
FLOOR_KEY = 's_hat_1N'


def load_sweep(path):
    with open(path) as f:
        return json.load(f)


def main():
    datasets = [
        (os.path.join(RESULTS_DIR, 'gmm_10d_n200_dt_robustness.json'),
         r'$\Delta t = 0.005$', 'tab:green', '--'),
        (os.path.join(RESULTS_DIR, 'gmm_10d_n200_all_sweeps_T200.json'),
         r'$\Delta t = 0.01$', 'tab:blue', '-'),
        (os.path.join(RESULTS_DIR, 'gmm_10d_dt_robust_dt0p02_T200.json'),
         r'$\Delta t = 0.02$', 'tab:orange', '-.'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))

    for path, label, color, ls in datasets:
        data = load_sweep(path)
        for i, name in enumerate(SWEEP_ORDER):
            ax = axes[i]
            sw = data['sweeps'][name]
            params = np.array(sw['param_values'])
            ssa_arr = np.array(sw[FLOOR_KEY])  # (n_seeds, n_params)
            ssa_mean = ssa_arr.mean(axis=0)
            ssa_sd = ssa_arr.std(axis=0)

            ax.plot(params, ssa_mean, ls, color=color, linewidth=2, label=label)
            ax.fill_between(params, ssa_mean - ssa_sd, ssa_mean + ssa_sd,
                            color=color, alpha=0.15)

    for i, name in enumerate(SWEEP_ORDER):
        ax = axes[i]
        ax.set_xlabel(XLABELS[name], fontsize=13, labelpad=3)
        ax.tick_params(labelsize=9)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
        if i == 0:
            ax.set_ylabel(r'$\hat{S}$', fontsize=13, labelpad=3)
            ax.legend(fontsize=11, loc='lower right', framealpha=0.8)
        if name == 'nmodes_10d':
            ax.set_xticks([1, 2, 3, 4, 5])

    plt.tight_layout(pad=0.5)
    for ext in ['png', 'pdf']:
        out = os.path.join(EXPDIR, f'fig_dt_robustness.{ext}')
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == '__main__':
    main()
