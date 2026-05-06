"""Task 1 (revised v3): figures/fig_sdxl_person_grid.png

2 rows x 9 cols (4 + gap + 4):
  Row 1: 4x w=1   | gap | 4x w=3
  Row 2: 4x w=7.5 | gap | 4x w=15

Source: first 4 thumbs from each person row in
figures/fig_sdxl_guidance_grid_1.png (4 person rows at w=1,3,7.5,15;
each row has 10 thumbs of 64x64).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

REPO = '/Users/stolk/github/ssa-paper'
SRC = f'{REPO}/figures/fig_sdxl_guidance_grid_1.png'
PERSON_ROWS = {
    '1':   (18, 82),
    '3':   (84, 148),
    '7.5': (150, 214),
    '15':  (216, 280),
}
COLS = [(50, 114), (116, 180), (182, 246), (248, 312)]

LAYOUT = [
    (0, 'L', 'w=1',   '1'),
    (0, 'R', 'w=3',   '3'),
    (1, 'L', 'w=7.5', '7.5'),
    (1, 'R', 'w=15',  '15'),
]


def first4(big, w_key):
    y0, y1 = PERSON_ROWS[w_key]
    return [big.crop((x0, y0, x1, y1)) for x0, x1 in COLS]


big = Image.open(SRC)

L = 0.01
R = 0.99
T = 0.92
B = 0.04
fig_w = 9 * 1.5
fig_h = 2 * 1.5 + 0.5
fig = plt.figure(figsize=(fig_w, fig_h))
gs = gridspec.GridSpec(
    2, 9, figure=fig,
    width_ratios=[1] * 4 + [0.3] + [1] * 4,
    wspace=0.04, hspace=0.20,
    left=L, right=R, top=T, bottom=B,
)

for row, side, label, w_key in LAYOUT:
    base = 0 if side == 'L' else 5
    imgs = first4(big, w_key)
    for k in range(4):
        ax = fig.add_subplot(gs[row, base + k])
        ax.imshow(imgs[k], interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

# Compute block centers in figure coordinates and place group labels above each block.
usable_w = R - L
unit = usable_w / sum([1] * 4 + [0.3] + [1] * 4)  # width of one cell unit
# left block cells span columns 0..3 in gs => figure-x [L, L + 4*unit]
# right block cells span columns 5..8 => figure-x [L + 4.3*unit, L + 8.3*unit]
left_block_center = L + 4 * unit / 2
right_block_center = L + 4.3 * unit + 4 * unit / 2

# Vertical positions: just above each row's top.
# Row 0 top is at gs top edge ~ T. Row 1 top at midpoint of grid below row 0.
# Approximate via the first axes location.
row_tops = []
for row in range(2):
    ax_first = fig.add_subplot(gs[row, 0])
    ax_first.set_visible(False)
    bbox = ax_first.get_position()
    row_tops.append(bbox.y1)

for row, side, label, _ in LAYOUT:
    x = left_block_center if side == 'L' else right_block_center
    y = row_tops[row] + 0.02
    fig.text(x, y, label, ha='center', va='bottom', fontsize=11)

out = f'{REPO}/figures/fig_sdxl_person_grid.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print('saved', out)
