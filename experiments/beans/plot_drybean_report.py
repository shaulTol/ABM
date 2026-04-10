#!/usr/bin/env python3
"""Generate a multi-page PDF report: DA vs PCA on the Dry Bean dataset."""
import sys, os, types, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

sys.path.insert(0, os.path.expanduser('~/github/ssa-paper'))
sys.modules['src.kde.autocorrelation_score'] = types.ModuleType('src.kde.autocorrelation_score')
sys.modules['src.kde.autocorrelation_score'].autocorrelation_score = None

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ── Class colors (consistent across all pages) ──
CLASS_COLORS = {
    'BARBUNYA': '#e41a1c', 'BOMBAY': '#ff7f00', 'CALI': '#4daf4a',
    'DERMASON': '#377eb8', 'HOROZ': '#984ea3', 'SEKER': '#a65628', 'SIRA': '#999999'
}

def load_dry_bean():
    """Load dry bean dataset — reuses logic from interpret_winners.py."""
    import pandas as pd
    path = os.path.join(OUTDIR, 'data', 'DryBeanDataset', 'Dry_Bean_Dataset.xlsx')
    if not os.path.exists(path):
        path = os.path.join(OUTDIR, 'data', 'Dry_Bean_Dataset.xlsx')
    df = pd.read_excel(path)
    feature_cols = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
                    'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
                    'Extent', 'Solidity', 'roundness', 'Compactness',
                    'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']
    X = df[feature_cols].values.astype(float)
    labels_str = df['Class'].values
    return X, labels_str, feature_cols


def backproject(X_raw, da1_pca, max_dim=8):
    """Standardize, PCA-reduce, back-project DA1/PC1 to original feature space."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_raw)
    pca = PCA(n_components=max_dim)
    pca.fit(X_s)
    da1_orig = pca.components_.T @ np.array(da1_pca)
    da1_orig = da1_orig / np.linalg.norm(da1_orig)
    pc1_orig = pca.components_[0]
    proj_da1 = X_s @ da1_orig
    proj_pc1 = X_s @ pc1_orig
    return X_s, da1_orig, pc1_orig, proj_da1, proj_pc1


def page_title(pdf, X_full, labels_str, feature_cols):
    """Page 1: Title + Dataset Overview."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'DA vs PCA on the Dry Bean Dataset', fontsize=20, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)
    ax.text(0.5, 0.91, 'Diffusion-Autocorrelation finds size mode that PCA misses',
            fontsize=12, ha='center', va='top', color='#555555', transform=ax.transAxes)

    # Dataset description
    desc = (
        f"The UCI Dry Bean dataset contains {len(X_full):,} samples of 7 bean varieties,\n"
        f"described by {len(feature_cols)} morphological features extracted from images.\n"
        f"For the DA computation, 2,000 samples were subsampled (seed=42).\n\n"
        f"Key finding: DA1 captures absolute bean size (Area, ConvexArea),\n"
        f"separating the physically large BOMBAY variety from all others.\n"
        f"PC1 instead captures elongation/shape (ShapeFactor2, ShapeFactor4).\n"
        f"The two directions are nearly orthogonal: |cos(DA1, PC1)| = 0.13."
    )
    ax.text(0.08, 0.80, desc, fontsize=11, va='top', transform=ax.transAxes,
            family='monospace', linespacing=1.5)

    # Class table
    classes = sorted(set(labels_str))
    rows = []
    for c in classes:
        mask = labels_str == c
        area = X_full[mask, 0]  # Area is first feature
        major = X_full[mask, 2]  # MajorAxisLength
        rows.append([c, str(np.sum(mask)), f'{area.mean():.0f}', f'{major.mean():.1f}'])

    table = ax.table(cellText=rows,
                     colLabels=['Class', 'N', 'Mean Area', 'Mean MajorAxis'],
                     loc='center', bbox=[0.1, 0.18, 0.8, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Highlight BOMBAY row
    for j in range(4):
        table[0, j].set_text_props(fontweight='bold')  # header
    bombay_idx = classes.index('BOMBAY') + 1
    for j in range(4):
        table[bombay_idx, j].set_facecolor('#fff3e0')

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_scatter(pdf, proj_da1, proj_pc1, labels_str_sub, cos_val):
    """Page 2: PC1 vs DA1 scatter colored by class."""
    fig, ax = plt.subplots(figsize=(8.5, 8))

    classes = sorted(set(labels_str_sub))
    for c in classes:
        mask = labels_str_sub == c
        ax.scatter(proj_pc1[mask], proj_da1[mask], s=8, alpha=0.5,
                   color=CLASS_COLORS[c], label=c, rasterized=True)

    ax.set_xlabel('PC1 projection (standardized)', fontsize=12)
    ax.set_ylabel('DA1 projection (standardized)', fontsize=12)
    ax.set_title(f'PC1 vs DA1 scatter  (|cos| = {cos_val:.3f})', fontsize=14)
    ax.legend(fontsize=9, markerscale=2, loc='upper left')
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_histograms(pdf, proj_da1, proj_pc1, labels_str_sub):
    """Page 3: Marginal histograms of DA1 and PC1 projections."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    classes = sorted(set(labels_str_sub))
    bins_da = np.linspace(proj_da1.min(), proj_da1.max(), 50)
    bins_pc = np.linspace(proj_pc1.min(), proj_pc1.max(), 50)

    for c in classes:
        mask = labels_str_sub == c
        ax1.hist(proj_da1[mask], bins=bins_da, alpha=0.5, color=CLASS_COLORS[c], label=c)
        ax2.hist(proj_pc1[mask], bins=bins_pc, alpha=0.5, color=CLASS_COLORS[c], label=c)

    ax1.set_xlabel('DA1 projection', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('DA1: BOMBAY separated by size', fontsize=12)
    ax1.legend(fontsize=7, ncol=2)

    ax2.set_xlabel('PC1 projection', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('PC1: shape/elongation variance', fontsize=12)
    ax2.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_loadings(pdf, da1_orig, pc1_orig, feature_cols):
    """Page 4: Side-by-side loading bar charts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))

    y = np.arange(len(feature_cols))

    # PC1 loadings
    order_pc = np.argsort(np.abs(pc1_orig))
    ax1.barh(y, pc1_orig[order_pc], color='steelblue', alpha=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels([feature_cols[i] for i in order_pc], fontsize=8)
    ax1.set_xlabel('Loading', fontsize=10)
    ax1.set_title('PC1 loadings', fontsize=12)
    ax1.axvline(0, color='k', lw=0.5)

    # DA1 loadings
    order_da = np.argsort(np.abs(da1_orig))
    ax2.barh(y, da1_orig[order_da], color='darkorange', alpha=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels([feature_cols[i] for i in order_da], fontsize=8)
    ax2.set_xlabel('Loading', fontsize=10)
    ax2.set_title('DA1 loadings', fontsize=12)
    ax2.axvline(0, color='k', lw=0.5)

    fig.suptitle('Feature loadings: PC1 (shape) vs DA1 (size)', fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def page_gap_profile(pdf, profile_path):
    """Page 5: Gap profile and angle vs lag."""
    d = np.load(profile_path)
    checkpoints = d['checkpoints']
    gap_profile = d['gap_profile']
    angles_deg = d['angles_deg']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Gap vs lag
    ax1.semilogx(checkpoints, gap_profile, 'o-', color='darkred', markersize=5)
    ax1.set_xlabel('Lag', fontsize=11)
    ax1.set_ylabel('Gap (λ₁/λ₂)', fontsize=11)
    ax1.set_title('Spectral gap profile', fontsize=12)
    ax1.axhline(gap_profile[-1], color='darkred', ls='--', alpha=0.3)
    ax1.text(checkpoints[-1]*0.5, gap_profile[-1]+0.2, f'gap = {gap_profile[-1]:.2f}',
             fontsize=9, ha='right', color='darkred')
    ax1.grid(True, alpha=0.3)

    # Angle vs lag
    ax2.semilogx(checkpoints, angles_deg, 's-', color='darkblue', markersize=5)
    ax2.set_xlabel('Lag', fontsize=11)
    ax2.set_ylabel('Angle DA1–PC1 (degrees)', fontsize=11)
    ax2.set_title('DA1 stability: angle to PC1', fontsize=12)
    ax2.axhline(90, color='gray', ls=':', alpha=0.5, label='orthogonal')
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Gap still rising at max lag — DA1 direction stable at ~82°', fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def page_class_table(pdf, X_raw, X_s, proj_da1, proj_pc1, labels_str_sub, da1_orig, pc1_orig, feature_cols):
    """Page 6: Per-class mean projections table + feature correlations."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')

    ax.text(0.5, 0.97, 'Per-class Statistics', fontsize=16, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)

    # Per-class table
    classes = sorted(set(labels_str_sub))
    rows = []
    for c in classes:
        mask = labels_str_sub == c
        rows.append([c, str(np.sum(mask)),
                     f'{proj_pc1[mask].mean():+.2f}',
                     f'{proj_da1[mask].mean():+.2f}'])

    table = ax.table(cellText=rows,
                     colLabels=['Class', 'N', 'Mean PC1', 'Mean DA1'],
                     loc='upper center', bbox=[0.1, 0.68, 0.8, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for j in range(4):
        table[0, j].set_text_props(fontweight='bold')
    # Highlight BOMBAY
    bombay_idx = classes.index('BOMBAY') + 1
    for j in range(4):
        table[bombay_idx, j].set_facecolor('#fff3e0')

    # Feature correlations
    ax.text(0.08, 0.62, 'Feature correlations with DA1 projection (Spearman, |r| > 0.3):',
            fontsize=11, fontweight='bold', va='top', transform=ax.transAxes)
    y = 0.58
    for i, fname in enumerate(feature_cols):
        r, _ = spearmanr(X_raw[:, i], proj_da1)
        if abs(r) > 0.3:
            ax.text(0.12, y, f'{fname:25s}  r = {r:+.3f}', fontsize=10,
                    va='top', transform=ax.transAxes, family='monospace')
            y -= 0.025

    y -= 0.02
    ax.text(0.08, y, 'Feature correlations with PC1 projection (Spearman, |r| > 0.3):',
            fontsize=11, fontweight='bold', va='top', transform=ax.transAxes)
    y -= 0.04
    for i, fname in enumerate(feature_cols):
        r, _ = spearmanr(X_raw[:, i], proj_pc1)
        if abs(r) > 0.3:
            ax.text(0.12, y, f'{fname:25s}  r = {r:+.3f}', fontsize=10,
                    va='top', transform=ax.transAxes, family='monospace')
            y -= 0.025

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    # Load data
    X_full, labels_str_full, feature_cols = load_dry_bean()

    # Load results
    with open(os.path.join(OUTDIR, 'winner', 'dry_bean_full_results.json')) as f:
        results = json.load(f)

    # Subsample to match DA run
    np.random.seed(42)
    idx = np.random.choice(len(X_full), 2000, replace=False)
    X_sub = X_full[idx]
    labels_str_sub = labels_str_full[idx]

    # Back-project
    X_s, da1_orig, pc1_orig, proj_da1, proj_pc1 = backproject(X_sub, results['da1'])
    cos_val = abs(np.dot(da1_orig, pc1_orig))

    print(f"|cos(DA1, PC1)| = {cos_val:.4f}")
    print(f"Gap = {results['gap_at_optimal']:.2f}")

    # Generate PDF
    outpath = os.path.join(OUTDIR, 'drybean_report.pdf')
    with PdfPages(outpath) as pdf:
        page_title(pdf, X_full, labels_str_full, feature_cols)
        page_scatter(pdf, proj_da1, proj_pc1, labels_str_sub, cos_val)
        page_histograms(pdf, proj_da1, proj_pc1, labels_str_sub)
        page_loadings(pdf, da1_orig, pc1_orig, feature_cols)
        page_gap_profile(pdf, os.path.join(OUTDIR, 'winner', 'dry_bean_full_profile.npz'))
        page_class_table(pdf, X_sub, X_s, proj_da1, proj_pc1, labels_str_sub,
                         da1_orig, pc1_orig, feature_cols)

    print(f"Saved: {outpath}")


if __name__ == '__main__':
    main()
