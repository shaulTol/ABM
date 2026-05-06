#!/usr/bin/env python
"""Plot S_hat vs MI in the same style as the main paper's GMM sweep curves."""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results', 'gmm_sweeps_redef.json')
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'results')

with open(RESULTS) as f:
    data = json.load(f)

sweep_configs = [
    ('sep_2d', 'Separation', 'Separation'),
    ('var_2d', 'Variance', 'Variance'),
    ('weight_2d', 'Weight', 'Weight'),
    ('nmodes_fixdist_2d', 'Number of modes', 'Mode count'),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))

for col, (config, xlabel, title) in enumerate(sweep_configs):
    ax = axes[col]
    r = data[config]
    params = np.array(r['param_values'])
    n_seeds = len(r['seeds'])

    # Collect per-seed arrays
    all_shat = np.array([s['s_hat'] for s in r['seeds']])  # (n_seeds, n_params)
    all_mi = np.array([s['mi'] for s in r['seeds']])

    shat_mean = all_shat.mean(axis=0)
    shat_sd = all_shat.std(axis=0)
    mi_mean = all_mi.mean(axis=0)

    # SSA (S_hat) with ±1 SD band
    ax.plot(params, shat_mean, '-', color='tab:blue', linewidth=2, label=r'$\hat{S}$')
    ax.fill_between(params, shat_mean - shat_sd, shat_mean + shat_sd,
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

    # Variance panel: MI ticks every 0.1
    if config == 'var_2d':
        from matplotlib.ticker import MultipleLocator
        ax2.yaxis.set_major_locator(MultipleLocator(0.1))
    if config == 'nmodes_fixdist_2d':
        from matplotlib.ticker import MultipleLocator
        ax2.yaxis.set_major_locator(MultipleLocator(0.4))

    if col == 0:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right',
                  fontsize=14, framealpha=0.8, handlelength=2)

plt.tight_layout(pad=0.5)
path = os.path.join(OUTDIR, 'fig_shat_vs_mi.png')
fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved {path}")
