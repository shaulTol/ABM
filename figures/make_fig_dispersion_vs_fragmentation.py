"""
Generate fig_dispersion_vs_fragmentation.png — three 2D distributions side by side.
(a) Unimodal Gaussian, (b) Fragmented GMM, (c) Dispersed Gaussian.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

x = np.linspace(-14, 14, 300)
y = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# (a) Unimodal: standard Gaussian
rv_a = multivariate_normal([0, 0], [[1, 0], [0, 1]])
Z_a = rv_a.pdf(pos)

# (b) Fragmented: two-component GMM, means +-4*e1, unit covariance
rv_b1 = multivariate_normal([-4, 0], [[1, 0], [0, 1]])
rv_b2 = multivariate_normal([4, 0], [[1, 0], [0, 1]])
Z_b = 0.5 * rv_b1.pdf(pos) + 0.5 * rv_b2.pdf(pos)

# (c) Dispersed: single Gaussian matching marginal variance of (b)
rv_c = multivariate_normal([0, 0], [[17, 0], [0, 1]])
Z_c = rv_c.pdf(pos)

labels = ["(a)", "(b)", "(c)"]
densities = [Z_a, Z_b, Z_c]

fig, axes = plt.subplots(1, 3, figsize=(10, 1.6))

for ax, Z, label in zip(axes, densities, labels):
    ax.contourf(X, Y, Z, levels=12, cmap="Blues")
    ax.contour(X, Y, Z, levels=6, colors="steelblue", linewidths=0.4, alpha=0.6)
    ax.set_xlim(-14, 14)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(label, fontsize=9, pad=3)

plt.tight_layout(pad=0.2)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_dispersion_vs_fragmentation.png')
fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.03)
plt.close()
print(f"Saved {path}")
