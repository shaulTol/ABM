"""
make_paper_figures.py — Generate clean paper figures for MNIST DDPM results.
No titles, no subtitles, no annotations. Labels added in LaTeX.
"""
import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
DA_DIR = os.path.join(EXP_DIR, "results_da")
FIGURES_DIR = os.path.join(EXP_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_mnist_by_digit():
    from torchvision.datasets import MNIST
    from torchvision import transforms
    dataset = MNIST(root="data", train=True, download=True,
                    transform=transforms.ToTensor())
    by_digit = {d: [] for d in range(10)}
    for img, label in dataset:
        by_digit[label].append(img.squeeze().numpy())
    return by_digit


def figure_a():
    """DA1 vs PCA1 pixel overlays for digit 1. Two panels, same 5 samples."""
    print("Figure A: DA1 vs PCA1 overlays for digit 1")
    da_data = np.load(os.path.join(DA_DIR, "da_digit_1.npz"))
    da1_abs = np.abs(da_data["da_eigenvectors"][:, 0])
    pca1_abs = np.abs(da_data["pca_eigenvectors"][:, 0])

    top_da = np.argsort(da1_abs)[-10:][::-1]
    top_pca = np.argsort(pca1_abs)[-10:][::-1]

    mnist = load_mnist_by_digit()
    np.random.seed(42)
    sample_ids = np.random.choice(len(mnist[1]), 5, replace=False)
    imgs = [mnist[1][i] for i in sample_ids]

    n_samples = 5
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.5, 3.0))

    for col in range(n_samples):
        # DA1 overlay (red) — top row
        img = imgs[col]
        rgb = np.stack([img] * 3, axis=-1)
        for idx in top_da:
            r, c = divmod(idx, 28)
            rgb[r, c] = [1.0, 0.0, 0.0]
        axes[0, col].imshow(rgb, interpolation="nearest")
        axes[0, col].axis("off")

        # PCA1 overlay (blue) — bottom row
        rgb2 = np.stack([img] * 3, axis=-1)
        for idx in top_pca:
            r, c = divmod(idx, 28)
            rgb2[r, c] = [0.0, 0.3, 1.0]
        axes[1, col].imshow(rgb2, interpolation="nearest")
        axes[1, col].axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    path = os.path.join(FIGURES_DIR, "fig_mnist_ddpm_da1_vs_pca1.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="black")
    plt.close()
    print(f"  Saved {path}")


def figure_b():
    """Cluster comparison: digit 1 vs digit 3. k-means k=2, 6 samples per cluster."""
    print("Figure B: Cluster comparison digit 1 vs digit 3")
    from sklearn.cluster import KMeans

    mnist = load_mnist_by_digit()
    np.random.seed(42)

    n_per_cluster = 6
    digits = [1, 3]

    # 4 rows total: 2 per digit
    fig, axes = plt.subplots(4, n_per_cluster, figsize=(n_per_cluster * 1.2, 4 * 1.2))

    for di, digit in enumerate(digits):
        samples = np.array([img.flatten() for img in mnist[digit]])
        km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(samples)
        labels = km.labels_

        for ci in range(2):
            row = di * 2 + ci
            cluster_idx = np.where(labels == ci)[0]
            chosen = np.random.choice(cluster_idx, n_per_cluster, replace=False)
            for col, idx in enumerate(chosen):
                axes[row, col].imshow(mnist[digit][idx], cmap="gray",
                                      interpolation="nearest")
                axes[row, col].axis("off")

    # Add gap between digit 1 and digit 3 rows
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    # Draw a horizontal separator between row 1 and row 2
    # Use fig coordinates
    line_y = 0.5  # midpoint
    fig.add_artist(plt.Line2D([0.05, 0.95], [line_y, line_y],
                              transform=fig.transFigure, color="gray",
                              linewidth=1.0, linestyle="-"))

    path = os.path.join(FIGURES_DIR, "fig_mnist_cluster_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close()
    print(f"  Saved {path}")


def figure_c():
    """SSA bar chart, all 10 digits sorted descending. Digit 1 highlighted."""
    print("Figure C: SSA bar chart")
    ssa_data = []
    for d in range(10):
        with open(os.path.join(RESULTS_DIR, f"ssa_digit_{d}.json")) as f:
            data = json.load(f)
        ssa_data.append((d, data["noise_levels"]["1"]["ssa"]))

    ssa_data.sort(key=lambda x: -x[1])
    digits = [d for d, _ in ssa_data]
    vals = [v for _, v in ssa_data]
    colors = ["#2166ac" if d == 1 else "#92c5de" for d in digits]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(len(digits)), vals, color=colors, edgecolor="k", linewidth=0.4)
    ax.set_xticks(range(len(digits)))
    ax.set_xticklabels([str(d) for d in digits], fontsize=11)
    ax.set_xlabel("Digit", fontsize=12)
    ax.set_ylabel("SSA", fontsize=12)
    ax.set_ylim(0, max(vals) * 1.08)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_mnist_ddpm_ssa_all_digits.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def figure_d():
    """Eigenvalue gap profile for digit 1. Log x-axis, vertical dashed at peak."""
    print("Figure D: Eigenvalue gap profile digit 1")
    da_data = np.load(os.path.join(DA_DIR, "da_digit_1.npz"))
    da_meta = json.load(open(os.path.join(DA_DIR, "da_digit_1.json")))

    checkpoints = da_data["checkpoints"]
    gap_profile = da_data["gap_profile"]
    dt = da_meta["dt"]
    phys_time = checkpoints * dt

    opt_lag = da_meta["optimal_lag"]
    opt_phys = opt_lag * dt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(phys_time, gap_profile, "o-", color="#2166ac", markersize=4, linewidth=1.5)
    ax.axvline(opt_phys, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(r"Physical time ($\tau \cdot \Delta t$)", fontsize=12)
    ax.set_ylabel(r"$\lambda_1 / \lambda_2$", fontsize=12)
    ax.set_xscale("log")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_mnist_digit1_gap_profile.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def figure_e():
    """|DA1| and |PCA1| heatmaps for digit 1, side by side."""
    print("Figure E: |DA1| vs |PCA1| heatmaps for digit 1")
    da_data = np.load(os.path.join(DA_DIR, "da_digit_1.npz"))
    da1_abs = np.abs(da_data["da_eigenvectors"][:, 0]).reshape(28, 28)
    pca1_abs = np.abs(da_data["pca_eigenvectors"][:, 0]).reshape(28, 28)

    vmax = max(da1_abs.max(), pca1_abs.max())
    diff = da1_abs - pca1_abs
    dmax = np.abs(diff).max()

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    im0 = axes[0].imshow(da1_abs, cmap="inferno", interpolation="nearest",
                          vmin=0, vmax=vmax)
    axes[0].set_title("|DA1|", fontsize=13, fontweight="bold", pad=6)
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, shrink=0.85).ax.tick_params(labelsize=8)

    im1 = axes[1].imshow(pca1_abs, cmap="inferno", interpolation="nearest",
                          vmin=0, vmax=vmax)
    axes[1].set_title("|PCA1|", fontsize=13, fontweight="bold", pad=6)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, shrink=0.85).ax.tick_params(labelsize=8)

    im2 = axes[2].imshow(diff, cmap="RdBu_r", interpolation="nearest",
                          vmin=-dmax, vmax=dmax)
    axes[2].set_title("|DA1| - |PCA1|", fontsize=13, fontweight="bold", pad=6)
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, shrink=0.85).ax.tick_params(labelsize=8)

    plt.subplots_adjust(wspace=0.35)
    path = os.path.join(FIGURES_DIR, "fig_mnist_digit1_da1_pca1_heatmaps.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close()
    print(f"  Saved {path}")


def main():
    figure_a()
    figure_b()
    figure_c()
    figure_d()
    figure_e()
    print("\nDone!")


if __name__ == "__main__":
    main()
