#!/usr/bin/env python3
"""
Generate fig_mnist_ssa_entropy_bars.png — grouped bar chart of SSA and
normalized entropy for all 10 MNIST digit classes, ordered by SSA descending.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP_DIR = os.path.join(os.path.dirname(__file__), '..', 'experiments')
DDPM_DIR = os.path.join(EXP_DIR, 'mnist', 'results')
ENTROPY_DIR = os.path.join(EXP_DIR, 'mnist', 'data')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'fig_mnist_ssa_entropy_bars.png')

# Load SSA from DDPM MNIST (noise_level=1)
ssa_dict = {}
for d in range(10):
    with open(os.path.join(DDPM_DIR, f'ssa_digit_{d}.json')) as f:
        data = json.load(f)
    ssa_dict[d] = data['noise_levels']['1']['ssa']
singles_ssa = pd.Series(ssa_dict)

# Load entropy from PCA-20 kNN
ent_df = pd.read_csv(os.path.join(ENTROPY_DIR, 'entropy_results.csv'))
singles_ent = ent_df[ent_df['digit_i'] == ent_df['digit_j']].copy()
singles_ent = singles_ent.set_index('digit_i')['entropy']

# Normalize entropy to [0, 1] by dividing by max
ent_norm = singles_ent / singles_ent.max()

# Sort by SSA descending
order = singles_ssa.sort_values(ascending=False).index
digits = [str(d) for d in order]
ssa_vals = singles_ssa[order].values
ent_vals = ent_norm[order].values

# Plot
fig, ax = plt.subplots(figsize=(8, 3))
x = np.arange(len(digits))
w = 0.28

ax.bar(x - w/2 - 0.02, ssa_vals, w, color='tab:blue', label='SSA')
ax.bar(x + w/2 + 0.02, ent_vals, w, color='tab:red', label='Normalized entropy')

# Reference line at digit 1 SSA
ax.axhline(y=ssa_vals[0], color='0.6', linestyle='--', linewidth=0.8, zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(digits, fontsize=8)
ax.set_ylim(0, 1)
ax.tick_params(axis='y', labelsize=8)
ax.set_xlabel('Digit', fontsize=8)
ax.legend(fontsize=8, framealpha=0.8)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved {OUT_PATH}")
