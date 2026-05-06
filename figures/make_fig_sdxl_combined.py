#!/usr/bin/env python3
"""
Combined SDXL figure: person guidance grid (left) + elderly DA1/PC1 (right).

Left square: 4 rows × 4 cols (w=1, w=3, w=7.5, w=15 top to bottom)
Right square: 4 rows × 5 cols (DA1 high, DA1 low, [gap], PC1 high, PC1 low)
  with extra vertical gap between DA1 block and PC1 block.

Sources (read-only):
  figures/fig_sdxl_guidance_grid_1.png — person thumbnails
  experiments/sdxl/results_da_pcavda/fig_elderly_da_vs_pca_freeconv_taustar_hires.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

REPO = '/Users/stolk/github/ssa-paper'

# ══════════════════════════════════════════════════════════════════════════════
# Source 1: Person guidance sweep — 256×256 tiles from grid_gs*.png
# ══════════════════════════════════════════════════════════════════════════════
GRID_DIR = f'{REPO}/experiments/sdxl/results_guidance_sweep'
TILE_SIZE = 256
GRID_COLS = 20
SELECTED = [3, 6, 8, 10]


def load_person_row(gs_tag):
    grid = Image.open(f'{GRID_DIR}/grid_gs{gs_tag}.png')
    tiles = []
    for i in SELECTED:
        col = i % GRID_COLS
        row = i // GRID_COLS
        x0 = col * TILE_SIZE
        y0 = row * TILE_SIZE
        tiles.append(grid.crop((x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE)))
    return tiles


person_rows = [
    ('w=1',   load_person_row('1p0')),
    ('w=3',   load_person_row('3p0')),
    ('w=7.5', load_person_row('7p5')),
    ('w=15',  load_person_row('15p0')),
]

# ══════════════════════════════════════════════════════════════════════════════
# Source 2: Elderly DA1/PC1 — crop individual portraits
# ══════════════════════════════════════════════════════════════════════════════
SRC_ELDERLY = f'{REPO}/experiments/sdxl/results_da_pcavda/fig_elderly_da_vs_pca_freeconv_taustar_hires.png'
ELDERLY_ROWS = [(29, 384), (438, 793), (848, 1203), (1257, 1612)]
ELDERLY_COLS = [(517, 872), (1123, 1478), (1730, 2085), (2337, 2692), (2943, 3298)]

big_elderly = Image.open(SRC_ELDERLY)


def crop_elderly_row(row_idx):
    y0, y1 = ELDERLY_ROWS[row_idx]
    return [big_elderly.crop((x0, y0, x1, y1)) for x0, x1 in ELDERLY_COLS]


da1_high = crop_elderly_row(0)[:4]
da1_low = [crop_elderly_row(1)[c] for c in range(4, -1, -1)][:4]
pc1_high = crop_elderly_row(2)[:4]
pc1_low = [crop_elderly_row(3)[c] for c in range(4, -1, -1)][:4]

elderly_rows = [
    ('DA1 high', da1_high),
    ('DA1 low',  da1_low),
    ('PC1 high', pc1_high),
    ('PC1 low',  pc1_low),
]

# ══════════════════════════════════════════════════════════════════════════════
# Combined figure
# ══════════════════════════════════════════════════════════════════════════════
# Left: 4 rows × 4 cols. Right: 5 rows (4 data + 1 gap) × 5 cols.
# Use a single figure with two nested GridSpecs.

fig_w = 12
fig_h = 6
fig = plt.figure(figsize=(fig_w, fig_h))

# Outer grid: 1 row × 2 columns (left=person, right=elderly)
outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1],
                          wspace=0.10, left=0.05, right=0.99, top=0.95, bottom=0.02)

# ── Left panel: person grid (4 rows × 4 cols) ────────────────────────────────
gs_left = gridspec.GridSpecFromSubplotSpec(
    4, 4, subplot_spec=outer[0, 0],
    wspace=0.04, hspace=0.12)

person_axes = {}
for r, (label, imgs) in enumerate(person_rows):
    for c in range(4):
        ax = fig.add_subplot(gs_left[r, c])
        ax.imshow(imgs[c], interpolation='lanczos')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if c == 0:
            person_axes[r] = ax

# ── Right panel: elderly (5 rows × 4 cols, row 2 is gap) ─────────────────────
gs_right = gridspec.GridSpecFromSubplotSpec(
    5, 4, subplot_spec=outer[0, 1],
    height_ratios=[1, 1, 0.15, 1, 1],
    wspace=0.04, hspace=0.06)

elderly_layout = [
    (0, 'DA1 high', da1_high),
    (1, 'DA1 low',  da1_low),
    (3, 'PC1 high', pc1_high),
    (4, 'PC1 low',  pc1_low),
]

elderly_axes = {}
for grid_row, label, imgs in elderly_layout:
    for c in range(4):
        ax = fig.add_subplot(gs_right[grid_row, c])
        ax.imshow(imgs[c], interpolation='lanczos')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if c == 0:
            elderly_axes[grid_row] = ax

# Place labels after layout is finalized
fig.canvas.draw()

# Person row labels — upper-left corner of leftmost image
w_labels = ['w=1', 'w=3', 'w=7.5', 'w=15']
for r, label in enumerate(w_labels):
    person_axes[r].text(0.04, 0.92, label, transform=person_axes[r].transAxes,
                        fontsize=10, fontweight='bold', va='top', ha='left',
                        color='white', bbox=dict(boxstyle='round,pad=0.2',
                        facecolor='black', alpha=0.6, edgecolor='none'))

# Elderly row labels — upper-left corner of leftmost image
elderly_labels = {0: 'DA1 high', 1: 'DA1 low', 3: 'PC1 high', 4: 'PC1 low'}
for grid_row, label in elderly_labels.items():
    elderly_axes[grid_row].text(0.04, 0.92, label,
                                transform=elderly_axes[grid_row].transAxes,
                                fontsize=9, fontweight='bold', va='top', ha='left',
                                color='white', bbox=dict(boxstyle='round,pad=0.2',
                                facecolor='black', alpha=0.6, edgecolor='none'))

out = f'{REPO}/figures/fig_sdxl_combined.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved {out}')
