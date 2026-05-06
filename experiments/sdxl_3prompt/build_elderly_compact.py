"""Task 2 (revised): figures/fig_elderly_da_pca_compact.png

2 rows x 11 columns (5 high | gap | 5 low). Uses the EXACT same images
that appear in fig_elderly_da_vs_pca_hires.png (DA1 high, DA1 low,
PC1 high, PC1 low — 4 rows of 5 in the original).

Source layout in fig_elderly_da_vs_pca_hires.png (3310x1643), detected by
white-space scan:
  rows: y=[194,513], [561,880], [927,1246], [1294,1613] (DA1 high, DA1 low,
        PC1 high, PC1 low)
  cols: x=[535,854], [1142,1461], [1748,2067], [2355,2674], [2962,3281]

Within each row the original ordering is "extreme first":
  high row: leftmost = HIGHEST score, rightmost = 5th-highest
  low row:  leftmost = LOWEST  score, rightmost = 5th-lowest

For the new compact layout we mirror the low side so extremes lie on the
outside: [highest .. 5th-highest | gap | 5th-lowest .. lowest].
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

REPO = '/Users/stolk/github/ssa-paper'
SRC = f'{REPO}/experiments/sdxl/results_da_pcavda/fig_elderly_da_vs_pca_hires.png'
ROWS = [(194, 513), (561, 880), (927, 1246), (1294, 1613)]   # DA1h, DA1l, PC1h, PC1l
COLS = [(535, 854), (1142, 1461), (1748, 2067), (2355, 2674), (2962, 3281)]


def cell(big, row, col):
    y0, y1 = ROWS[row]
    x0, x1 = COLS[col]
    return big.crop((x0, y0, x1, y1))


big = Image.open(SRC)
da1_high = [cell(big, 0, c) for c in range(5)]            # leftmost = highest
da1_low_rev = [cell(big, 1, c) for c in range(4, -1, -1)]  # reverse: lowest at far right
pc1_high = [cell(big, 2, c) for c in range(5)]
pc1_low_rev = [cell(big, 3, c) for c in range(4, -1, -1)]

fig = plt.figure(figsize=(11 * 1.5, 2 * 1.5))
gs = gridspec.GridSpec(
    2, 11, figure=fig,
    width_ratios=[1] * 5 + [0.25] + [1] * 5,
    wspace=0.04, hspace=0.04,
    left=0.05, right=0.99, top=0.99, bottom=0.01,
)


def place_row(row_idx, imgs_left, imgs_right, label):
    cols = list(range(5)) + list(range(6, 11))
    panel_imgs = imgs_left + imgs_right
    first_ax = None
    for k, col in enumerate(cols):
        ax = fig.add_subplot(gs[row_idx, col])
        ax.imshow(panel_imgs[k])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if k == 0:
            first_ax = ax
    first_ax.set_ylabel(label, fontsize=11, rotation=0,
                        labelpad=20, ha='right', va='center')


place_row(0, da1_high, da1_low_rev, 'DA1')
place_row(1, pc1_high, pc1_low_rev, 'PC1')

out = f'{REPO}/figures/fig_elderly_da_pca_compact.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print('saved', out)
