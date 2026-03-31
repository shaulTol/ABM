"""
Generate fig_mnist_clusters_compact.png — 3 digits × 2 clusters × 4 samples.
Digits 1, 7, 2 side by side with k-means k=2 clustering.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torchvision.datasets import MNIST
from torchvision import transforms

# Load MNIST by digit
dataset = MNIST(root="data", train=True, download=True,
                transform=transforms.ToTensor())
by_digit = {d: [] for d in range(10)}
for img, label in dataset:
    by_digit[label].append(img.squeeze().numpy())

digits = [1, 7, 2]
n_per_cluster = 4
np.random.seed(42)

# Compute clusters
clusters = {}
for d in digits:
    samples = np.array([img.flatten() for img in by_digit[d]])
    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(samples)
    labels = km.labels_
    clusters[d] = []
    for ci in range(2):
        idx = np.where(labels == ci)[0]
        chosen = np.random.choice(idx, n_per_cluster, replace=False)
        clusters[d].append([by_digit[d][i] for i in chosen])

# Figure: 2 rows × 12 cols, grouped as 3 × (2×4)
fig, axes = plt.subplots(2, 12, figsize=(9, 2.2))

digit_labels = ["(a)", "(b)", "(c)"]
row_labels = ["I", "II"]

for di, d in enumerate(digits):
    col_start = di * 4
    for ci in range(2):
        for col in range(n_per_cluster):
            ax = axes[ci, col_start + col]
            ax.imshow(clusters[d][ci][col], cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # (a), (b), (c) centered above each digit group
    mid_ax = axes[0, col_start + 1]
    mid_ax.set_title(digit_labels[di], fontsize=9, pad=3)

# Row labels on the very left
for ci in range(2):
    axes[ci, 0].set_ylabel(row_labels[ci], fontsize=8, rotation=0,
                           labelpad=8, va="center")

plt.subplots_adjust(wspace=0.04, hspace=0.06, left=0.03, right=0.99,
                    top=0.88, bottom=0.01)

# Draw vertical dividers between digit groups
fig.canvas.draw()
for di in range(1, len(digits)):
    col = di * 4
    x_left = axes[0, col - 1].get_position().x1
    x_right = axes[0, col].get_position().x0
    x_mid = (x_left + x_right) / 2
    y_bot = axes[1, col].get_position().y0
    y_top = axes[0, col].get_position().y1
    fig.add_artist(plt.Line2D([x_mid, x_mid], [y_bot, y_top],
                              transform=fig.transFigure,
                              color="gray", linewidth=0.6))

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_mnist_clusters_compact.png')
fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.03)
plt.close()
print(f"Saved {path}")
