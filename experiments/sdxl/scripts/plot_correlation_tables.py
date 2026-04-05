"""
Plot correlation tables as a multi-page PDF for meeting presentation.

Pages:
  1. Scribble inputs + DA1-sorted sample grids (circle & face)
  2. Scribble inputs + PC1-sorted sample grids (circle & face)
  3. DA1 vs PC1 comparison — side by side for both conditions
  4. SSA autocorrelation decay curves (CLIP space, all 6 conditions)
  5. CLIP-space correlation heatmaps (2x3: circle/face x 3 dt values)
  6. Cross-space correlation heatmaps (CLIP + LC)
  7. Key correlations summary bar chart
  8. CLIP DA1-sorted grids for all 6 conditions
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

matplotlib.rcParams.update({"font.size": 9})

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_clip_da")
AUTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_autoresearch")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load_corr(path):
    with open(path) as f:
        return json.load(f)


def short_name(name):
    for prefix in ("CLIP_", "LC_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def plot_heatmap(ax, mat, names, title):
    n = len(names)
    short = [short_name(nm) for nm in names]
    ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for i in range(n):
        for j in range(n):
            val = mat[i][j]
            color = "white" if abs(val) > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5.5, color=color)


def get_clip_corr(data, row_name, col_name):
    names = data["clip_names"]
    i = names.index(row_name)
    j = names.index(col_name)
    return data["clip_corr"][i][j]


CONFIGS = [
    ("clip_cn_circle", "Circle dt=0.0001"),
    ("clip_cn_circle_dt0.001", "Circle dt=0.001"),
    ("clip_cn_circle_dt0.005", "Circle dt=0.005"),
    ("clip_cn_dps_face", "Face dt=0.0001"),
    ("clip_cn_dps_face_dt0.001", "Face dt=0.001"),
    ("clip_cn_dps_face_dt0.005", "Face dt=0.005"),
]


def page_scribbles_and_direction_grids(pdf, direction, direction_label):
    """Scribble inputs + direction-sorted grids for both conditions."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"ControlNet Scribble Inputs & {direction_label}-Sorted Sample Grids",
                 fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2,
                          height_ratios=[1, 1.2])

    if direction == "da1":
        grid_circle = os.path.join(BASE, "interpret_da_sdxl_ssa_cn_circle_grid.png")
        grid_face = os.path.join(BASE, "interpret_da_sdxl_ssa_cn_dps_face_grid.png")
    elif direction == "pc1":
        grid_circle = os.path.join(BASE, "interpret_pc1_sdxl_ssa_cn_circle_grid.png")
        grid_face = os.path.join(BASE, "interpret_pc1_sdxl_ssa_cn_dps_face_grid.png")
    elif direction == "pc2":
        grid_circle = os.path.join(BASE, "interpret_pc2_sdxl_ssa_cn_circle_grid.png")
        grid_face = os.path.join(BASE, "interpret_pc2_sdxl_ssa_cn_dps_face_grid.png")
    else:
        return

    conditions = [
        ("Circle Scribble", os.path.join(DATA, "circle_scribble.png"), grid_circle),
        ("Face Scribble (DPS)", os.path.join(DATA, "dps_face_scribble.png"), grid_face),
    ]

    for col, (label, scribble_path, grid_path) in enumerate(conditions):
        ax_s = fig.add_subplot(gs[0, col])
        img_s = Image.open(scribble_path)
        ax_s.imshow(img_s)
        ax_s.set_title(label, fontsize=12, fontweight="bold")
        ax_s.axis("off")

        ax_g = fig.add_subplot(gs[1, col])
        if os.path.exists(grid_path):
            img_g = Image.open(grid_path)
            ax_g.imshow(img_g)
        else:
            ax_g.text(0.5, 0.5, "Grid not found", ha="center", va="center",
                      transform=ax_g.transAxes, fontsize=12)
        ax_g.set_title(f"{direction_label}: Top-8 (row 1) vs Bottom-8 (row 2)", fontsize=10)
        ax_g.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_da_vs_pca_comparison(pdf):
    """DA1 vs PC1 side-by-side comparison for both conditions."""
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("DA1 vs PC1: Are They the Same Direction?",
                 fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.15)

    grid_pairs = [
        ("Circle — DA1", os.path.join(BASE, "interpret_da_sdxl_ssa_cn_circle_grid.png")),
        ("Circle — PC1", os.path.join(BASE, "interpret_pc1_sdxl_ssa_cn_circle_grid.png")),
        ("Face — DA1", os.path.join(BASE, "interpret_da_sdxl_ssa_cn_dps_face_grid.png")),
        ("Face — PC1", os.path.join(BASE, "interpret_pc1_sdxl_ssa_cn_dps_face_grid.png")),
    ]

    for idx, (title, path) in enumerate(grid_pairs):
        r, c = idx // 2, idx % 2
        ax = fig.add_subplot(gs[r, c])
        if os.path.exists(path):
            img = Image.open(path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Not found", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    # Add correlation annotations
    for cond_name, row, label in [
        ("clip_cn_circle", 0, "Circle"),
        ("clip_cn_dps_face", 1, "Face"),
    ]:
        corr_path = os.path.join(BASE, f"corr_table_{cond_name}.json")
        if os.path.exists(corr_path):
            data = load_corr(corr_path)
            rho = get_clip_corr(data, "CLIP_DA1", "CLIP_PC1")
            fig.text(0.5, 0.52 - row * 0.48, f"{label}: corr(DA1, PC1) = {rho:.3f}",
                     ha="center", fontsize=11, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                               edgecolor="gray"))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_pc2_grids(pdf):
    """PC2-sorted grids for both conditions."""
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("PC2-Sorted Sample Grids: Top-8 vs Bottom-8",
                 fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(1, 2, wspace=0.15)

    grids = [
        ("Circle — PC2", os.path.join(BASE, "interpret_pc2_sdxl_ssa_cn_circle_grid.png")),
        ("Face — PC2", os.path.join(BASE, "interpret_pc2_sdxl_ssa_cn_dps_face_grid.png")),
    ]

    for col, (title, path) in enumerate(grids):
        ax = fig.add_subplot(gs[0, col])
        if os.path.exists(path):
            img = Image.open(path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Not found", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_rho_decay(pdf):
    """SSA autocorrelation decay curves."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("SSA Autocorrelation Decay Curves (CLIP space)",
                 fontsize=14, fontweight="bold")

    for idx, (cond_name, label) in enumerate(CONFIGS):
        row, col = idx // 3, idx % 3
        da_path = os.path.join(BASE, f"clip_da_{cond_name}.npz")

        if os.path.exists(da_path):
            da_data = dict(np.load(da_path, allow_pickle=True))
            if "rho_sq" in da_data and "snapshot_lags" in da_data:
                lags = da_data["snapshot_lags"]
                rho = da_data["rho_sq"]
                axes[row, col].plot(lags, rho, "o-", markersize=3, color="steelblue")
                axes[row, col].set_xlabel("Lag")
                axes[row, col].set_ylabel(r"$\rho^2(\tau)$")
                axes[row, col].set_ylim(0, 1.05)
                axes[row, col].axhline(0, ls="--", color="gray", alpha=0.4)

                json_path = os.path.join(BASE, f"clip_da_{cond_name}.json")
                if os.path.exists(json_path):
                    with open(json_path) as f:
                        jdata = json.load(f)
                    ssa = jdata.get("ssa_clip", None)
                    if ssa is not None:
                        axes[row, col].text(0.95, 0.95, f"SSA={ssa:.3f}",
                                            ha="right", va="top",
                                            transform=axes[row, col].transAxes,
                                            fontsize=10, fontweight="bold",
                                            bbox=dict(boxstyle="round,pad=0.3",
                                                      facecolor="wheat", alpha=0.8))
            else:
                axes[row, col].text(0.5, 0.5, "No rho data",
                                    ha="center", va="center",
                                    transform=axes[row, col].transAxes)
        else:
            axes[row, col].text(0.5, 0.5, "NPZ not found",
                                ha="center", va="center",
                                transform=axes[row, col].transAxes)

        axes[row, col].set_title(label, fontsize=10, fontweight="bold")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_clip_heatmaps(pdf):
    """CLIP-space correlation heatmaps (2x3 grid)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("CLIP-Space Correlation Tables (DA, PCA, Text Probes)",
                 fontsize=14, fontweight="bold", y=0.98)

    for idx, (name, title) in enumerate(CONFIGS):
        path = os.path.join(BASE, f"corr_table_{name}.json")
        data = load_corr(path)
        row, col = idx // 3, idx % 3
        mat = np.array(data["clip_corr"])
        names = data["clip_names"]
        plot_heatmap(axes[row, col], mat, names, title)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-1, 1))
    fig.colorbar(sm, ax=axes, shrink=0.6, label="Spearman rho")
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_cross_heatmaps(pdf):
    """Cross-space correlation heatmaps (CLIP + LC)."""
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle("Cross-Space Correlation Tables (CLIP + LC)",
                 fontsize=14, fontweight="bold", y=0.98)

    for idx, (name, title) in enumerate(CONFIGS):
        path = os.path.join(BASE, f"corr_table_{name}.json")
        data = load_corr(path)
        row, col = idx // 3, idx % 3
        mat = np.array(data["cross_corr"])
        names = data["cross_names"]
        plot_heatmap(axes[row, col], mat, names, title)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-1, 1))
    fig.colorbar(sm, ax=axes, shrink=0.6, label="Spearman rho")
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_summary_bars(pdf):
    """Key correlations summary bar chart."""
    fig, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Key Correlations Across Conditions", fontsize=13, fontweight="bold")

    dts = ["0.0001", "0.001", "0.005"]
    x = np.arange(len(dts))
    width = 0.35

    circle_da1_pc1, face_da1_pc1 = [], []
    circle_da1_gender, face_da1_gender = [], []
    circle_da1_realism, face_da1_realism = [], []

    for name_c, name_f in [
        ("clip_cn_circle", "clip_cn_dps_face"),
        ("clip_cn_circle_dt0.001", "clip_cn_dps_face_dt0.001"),
        ("clip_cn_circle_dt0.005", "clip_cn_dps_face_dt0.005"),
    ]:
        dc = load_corr(os.path.join(BASE, f"corr_table_{name_c}.json"))
        df = load_corr(os.path.join(BASE, f"corr_table_{name_f}.json"))
        circle_da1_pc1.append(abs(get_clip_corr(dc, "CLIP_DA1", "CLIP_PC1")))
        face_da1_pc1.append(abs(get_clip_corr(df, "CLIP_DA1", "CLIP_PC1")))
        circle_da1_gender.append(abs(get_clip_corr(dc, "CLIP_DA1", "CLIP_gender")))
        face_da1_gender.append(abs(get_clip_corr(df, "CLIP_DA1", "CLIP_gender")))
        circle_da1_realism.append(abs(get_clip_corr(dc, "CLIP_DA1", "CLIP_realism")))
        face_da1_realism.append(abs(get_clip_corr(df, "CLIP_DA1", "CLIP_realism")))

    for ax, title, circ, face in [
        (axes3[0], "|DA1 ~ PC1|", circle_da1_pc1, face_da1_pc1),
        (axes3[1], "|DA1 ~ gender|", circle_da1_gender, face_da1_gender),
        (axes3[2], "|DA1 ~ realism|", circle_da1_realism, face_da1_realism),
    ]:
        ax.bar(x - width/2, circ, width, label="Circle", color="steelblue")
        ax.bar(x + width/2, face, width, label="Face", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(dts)
        ax.set_xlabel("dt")
        ax.set_ylabel("|Spearman rho|")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    axes3[0].axhline(0.95, ls="--", color="gray", alpha=0.5, lw=0.8)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_da1_grids_all_dt(pdf):
    """DA1-sorted grids for all 6 conditions (CLIP-space DA)."""
    grid_files = [
        ("clip_da_clip_cn_circle_da1_grid.png", "Circle dt=0.0001"),
        ("clip_da_clip_cn_circle_dt0.001_da1_grid.png", "Circle dt=0.001"),
        ("clip_da_clip_cn_circle_dt0.005_da1_grid.png", "Circle dt=0.005"),
        ("clip_da_clip_cn_dps_face_da1_grid.png", "Face dt=0.0001"),
        ("clip_da_clip_cn_dps_face_dt0.001_da1_grid.png", "Face dt=0.001"),
        ("clip_da_clip_cn_dps_face_dt0.005_da1_grid.png", "Face dt=0.005"),
    ]

    existing = [(f, t) for f, t in grid_files if os.path.exists(os.path.join(BASE, f))]
    if not existing:
        return

    n = len(existing)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows))
    fig.suptitle("CLIP DA1-Sorted Grids: Top-8 vs Bottom-8 (all dt values)",
                 fontsize=14, fontweight="bold")
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (fname, title) in enumerate(existing):
        r, c = idx // 2, idx % 2
        img = Image.open(os.path.join(BASE, fname))
        axes[r, c].imshow(img)
        axes[r, c].set_title(title, fontsize=11, fontweight="bold")
        axes[r, c].axis("off")

    for idx in range(n, rows * 2):
        r, c = idx // 2, idx % 2
        axes[r, c].axis("off")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


AUTO_CONFIGS = [
    ("autoresearch_egg", "Egg"),
    ("autoresearch_square", "Square"),
    ("autoresearch_street", "Street"),
]


def _get_auto_stats(name):
    """Get key stats string from autoresearch JSON."""
    json_path = os.path.join(AUTO, f"clip_da_{name}.json")
    if not os.path.exists(json_path):
        return ""
    with open(json_path) as f:
        jdata = json.load(f)
    ssa = jdata.get("ssa_clip", 0)
    gap = jdata.get("da1_gap", 0)
    lag = jdata.get("da1_optimal_lag", 0)
    corrs = jdata.get("da1_correlations", {})
    top_probes = sorted(corrs.items(), key=lambda x: abs(x[1]["rho"]), reverse=True)[:3]
    probe_str = ", ".join(f"{p}={v['rho']:.2f}" for p, v in top_probes)
    return f"SSA={ssa:.3f}  |  DA1 gap={gap:.2f} @ lag={lag}  |  {probe_str}"


def _get_auto_corr_str(name, dir1, dir2):
    """Get corr(dir1, dir2) from autoresearch correlation table."""
    path = os.path.join(AUTO, f"corr_table_{name}.json")
    if not os.path.exists(path):
        return ""
    data = load_corr(path)
    try:
        rho = get_clip_corr(data, f"CLIP_{dir1}", f"CLIP_{dir2}")
        return f"corr({dir1},{dir2})={rho:.3f}"
    except (ValueError, KeyError):
        return ""


def page_auto_da_vs_pca(pdf, name, label):
    """One page per autoresearch experiment: DA1 vs PC1 side by side + curves + heatmap."""
    # Strip prefix for grid filenames
    short = name.replace("autoresearch_", "")

    fig = plt.figure(figsize=(16, 16))
    fig.suptitle(f"{label}: DA vs PCA Comparison (CLIP space, dt=0.001)\n{_get_auto_stats(name)}",
                 fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.15, height_ratios=[1, 1, 1.2])

    da1_grid = os.path.join(AUTO, f"clip_da_{name}_da1_grid.png")
    pc1_grid = os.path.join(AUTO, f"{name}_pc1_grid.png")
    pc2_grid = os.path.join(AUTO, f"{name}_pc2_grid.png")
    curves = os.path.join(AUTO, f"clip_da_{name}_curves.png")
    corr_path = os.path.join(AUTO, f"corr_table_{name}.json")

    corr_da1_gender = _get_auto_corr_str(name, "DA1", "gender")
    corr_da1_pc1 = _get_auto_corr_str(name, "DA1", "PC1")

    # Row 0, left: DA1 grid
    ax_da1 = fig.add_subplot(gs[0, 0])
    if os.path.exists(da1_grid):
        ax_da1.imshow(Image.open(da1_grid))
    else:
        ax_da1.text(0.5, 0.5, "DA1 grid not found", ha="center", va="center",
                    transform=ax_da1.transAxes)
    ax_da1.set_title(f"DA1 sorted: Top-8 vs Bottom-8\n{corr_da1_gender}", fontsize=10, fontweight="bold")
    ax_da1.axis("off")

    # Row 0, right: PC1 grid
    ax_pc1 = fig.add_subplot(gs[0, 1])
    if os.path.exists(pc1_grid):
        ax_pc1.imshow(Image.open(pc1_grid))
    else:
        ax_pc1.text(0.5, 0.5, "PC1 grid not found", ha="center", va="center",
                    transform=ax_pc1.transAxes)
    ax_pc1.set_title(f"PC1 sorted: Top-8 vs Bottom-8\n{corr_da1_pc1}", fontsize=10, fontweight="bold")
    ax_pc1.axis("off")

    # Row 1, left: PC2 grid
    ax_pc2 = fig.add_subplot(gs[1, 0])
    if os.path.exists(pc2_grid):
        ax_pc2.imshow(Image.open(pc2_grid))
    else:
        ax_pc2.text(0.5, 0.5, "PC2 grid not found", ha="center", va="center",
                    transform=ax_pc2.transAxes)
    ax_pc2.set_title("PC2 sorted: Top-8 vs Bottom-8", fontsize=10, fontweight="bold")
    ax_pc2.axis("off")

    # Row 1, right: Rho decay curves
    ax_rho = fig.add_subplot(gs[1, 1])
    if os.path.exists(curves):
        ax_rho.imshow(Image.open(curves))
        ax_rho.set_title("Rho decay & gap profiles", fontsize=10, fontweight="bold")
        ax_rho.axis("off")
    else:
        da_path = os.path.join(AUTO, f"clip_da_{name}.npz")
        if os.path.exists(da_path):
            da_data = dict(np.load(da_path, allow_pickle=True))
            if "rho_sq" in da_data and "snapshot_lags" in da_data:
                ax_rho.plot(da_data["snapshot_lags"], da_data["rho_sq"], "o-",
                            markersize=3, color="steelblue")
                ax_rho.set_xlabel("Lag")
                ax_rho.set_ylabel(r"$\rho^2(\tau)$")
                ax_rho.set_ylim(0, 1.05)
                ax_rho.set_title("Rho decay", fontsize=10, fontweight="bold")

    # Row 2, left: Correlation heatmap
    ax_hm = fig.add_subplot(gs[2, 0])
    if os.path.exists(corr_path):
        data = load_corr(corr_path)
        mat = np.array(data["clip_corr"])
        names = data["clip_names"]
        plot_heatmap(ax_hm, mat, names, f"{label} — CLIP Correlation Table")
    else:
        ax_hm.text(0.5, 0.5, "Corr table not found", ha="center", va="center",
                   transform=ax_hm.transAxes)

    # Row 2, right: Text summary
    ax_txt = fig.add_subplot(gs[2, 1])
    ax_txt.axis("off")
    json_path = os.path.join(AUTO, f"clip_da_{name}.json")
    if os.path.exists(json_path) and os.path.exists(corr_path):
        with open(json_path) as f:
            jdata = json.load(f)
        data = load_corr(corr_path)

        lines = [f"{label} Summary", "=" * 40, ""]
        lines.append(f"SSA (CLIP): {jdata.get('ssa_clip', 0):.3f}")
        lines.append(f"SSA (LC):   {jdata.get('ssa_lc', 0):.3f}")
        lines.append("")
        lines.append(f"DA1 gap: {jdata.get('da1_gap', 0):.2f} @ lag={jdata.get('da1_optimal_lag', 0)}")
        lines.append(f"DA2 gap: {jdata.get('da2_gap', 0):.2f} @ lag={jdata.get('da2_optimal_lag', 0)}")
        lines.append("")
        lines.append("DA1 correlations:")
        for probe, vals in sorted(jdata.get("da1_correlations", {}).items(),
                                   key=lambda x: abs(x[1]["rho"]), reverse=True):
            lines.append(f"  {probe:>12}: rho={vals['rho']:+.3f}")
        lines.append("")
        lines.append("DA vs PCA:")
        for pair in [("DA1", "PC1"), ("DA1", "PC2"), ("DA2", "PC1"), ("DA2", "PC2")]:
            lines.append(f"  corr({pair[0]},{pair[1]}): {get_clip_corr(data, f'CLIP_{pair[0]}', f'CLIP_{pair[1]}'):.3f}")
        lines.append("")
        lines.append(f"Cross-space DA1 Spearman: {jdata.get('cross_space_da1_spearman', 0):.3f}")

        ax_txt.text(0.05, 0.95, "\n".join(lines), transform=ax_txt.transAxes,
                    fontsize=9, fontfamily="monospace", va="top")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_auto_heatmaps(pdf):
    """CLIP-space correlation heatmaps for the 3 autoresearch experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Autoresearch: CLIP-Space Correlation Tables (DA, PCA, Text Probes)",
                 fontsize=14, fontweight="bold")

    for idx, (name, label) in enumerate(AUTO_CONFIGS):
        path = os.path.join(AUTO, f"corr_table_{name}.json")
        if os.path.exists(path):
            data = load_corr(path)
            mat = np.array(data["clip_corr"])
            names = data["clip_names"]
            plot_heatmap(axes[idx], mat, names, label)
        else:
            axes[idx].text(0.5, 0.5, "Not found", ha="center", va="center",
                           transform=axes[idx].transAxes)
            axes[idx].set_title(label)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_raw_vs_orth_heatmaps(pdf, results_dir, configs, section_title):
    """Side-by-side raw vs orthogonalized heatmaps for a set of experiments.

    Only emitted if the JSON files contain 'clip_orth_corr'.
    """
    # Check if any config has orthogonalized data
    has_orth = False
    for name, label in configs:
        path = os.path.join(results_dir, f"corr_table_{name}.json")
        if os.path.exists(path):
            data = load_corr(path)
            if "clip_orth_corr" in data:
                has_orth = True
                break

    if not has_orth:
        return

    n_exp = len(configs)
    fig, axes = plt.subplots(n_exp, 2, figsize=(16, 7 * n_exp))
    fig.suptitle(f"{section_title}: Raw vs Orthogonalized Probe Correlations",
                 fontsize=14, fontweight="bold", y=0.98)

    if n_exp == 1:
        axes = axes.reshape(1, -1)

    for idx, (name, label) in enumerate(configs):
        path = os.path.join(results_dir, f"corr_table_{name}.json")
        if not os.path.exists(path):
            for c in range(2):
                axes[idx, c].text(0.5, 0.5, "Not found", ha="center", va="center",
                                  transform=axes[idx, c].transAxes)
            continue

        data = load_corr(path)

        # Raw
        mat = np.array(data["clip_corr"])
        names = data["clip_names"]
        plot_heatmap(axes[idx, 0], mat, names, f"{label} — Raw")

        # Orthogonalized
        if "clip_orth_corr" in data:
            orth_mat = np.array(data["clip_orth_corr"])
            orth_names = data["clip_orth_names"]
            plot_heatmap(axes[idx, 1], orth_mat, orth_names, f"{label} — Orthogonalized")
        else:
            axes[idx, 1].text(0.5, 0.5, "No orthogonalized data", ha="center",
                              va="center", transform=axes[idx, 1].transAxes)
            axes[idx, 1].set_title(f"{label} — Orthogonalized")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_auto_rho_decay(pdf):
    """SSA rho decay curves for the 3 autoresearch experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Autoresearch: SSA Autocorrelation Decay (CLIP space, dt=0.001)",
                 fontsize=14, fontweight="bold")

    for idx, (name, label) in enumerate(AUTO_CONFIGS):
        da_path = os.path.join(AUTO, f"clip_da_{name}.npz")
        if os.path.exists(da_path):
            da_data = dict(np.load(da_path, allow_pickle=True))
            if "rho_sq" in da_data and "snapshot_lags" in da_data:
                lags = da_data["snapshot_lags"]
                rho = da_data["rho_sq"]
                axes[idx].plot(lags, rho, "o-", markersize=3, color="steelblue")
                axes[idx].set_ylim(0, 1.05)
                axes[idx].axhline(0, ls="--", color="gray", alpha=0.4)

                json_path = os.path.join(AUTO, f"clip_da_{name}.json")
                if os.path.exists(json_path):
                    with open(json_path) as f:
                        jdata = json.load(f)
                    ssa = jdata.get("ssa_clip", None)
                    if ssa is not None:
                        axes[idx].text(0.95, 0.95, f"SSA={ssa:.3f}",
                                       ha="right", va="top",
                                       transform=axes[idx].transAxes,
                                       fontsize=10, fontweight="bold",
                                       bbox=dict(boxstyle="round,pad=0.3",
                                                 facecolor="wheat", alpha=0.8))

        axes[idx].set_xlabel("Lag")
        axes[idx].set_ylabel(r"$\rho^2(\tau)$")
        axes[idx].set_title(label, fontsize=11, fontweight="bold")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_auto_summary_comparison(pdf):
    """Bar chart comparing all 5 experiments: circle, face, egg, square, street."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("All Experiments Comparison (dt=0.001)",
                 fontsize=14, fontweight="bold")

    experiments = []
    labels = []

    # Circle & face dt=0.001
    for name, label in [("clip_cn_circle_dt0.001", "Circle"), ("clip_cn_dps_face_dt0.001", "Face")]:
        path = os.path.join(BASE, f"corr_table_{name}.json")
        if os.path.exists(path):
            experiments.append(load_corr(path))
            labels.append(label)

    # Autoresearch
    for name, label in AUTO_CONFIGS:
        path = os.path.join(AUTO, f"corr_table_{name}.json")
        if os.path.exists(path):
            experiments.append(load_corr(path))
            labels.append(label)

    if not experiments:
        plt.close(fig)
        return

    x = np.arange(len(labels))
    width = 0.6
    colors = ["steelblue", "coral", "#2ecc71", "#9b59b6", "#e67e22"]

    da1_pc1 = [abs(get_clip_corr(d, "CLIP_DA1", "CLIP_PC1")) for d in experiments]
    da1_gender = [abs(get_clip_corr(d, "CLIP_DA1", "CLIP_gender")) for d in experiments]
    da1_age = [abs(get_clip_corr(d, "CLIP_DA1", "CLIP_age")) for d in experiments]

    for ax, title, vals in [
        (axes[0], "|DA1 ~ PC1|", da1_pc1),
        (axes[1], "|DA1 ~ gender|", da1_gender),
        (axes[2], "|DA1 ~ age|", da1_age),
    ]:
        ax.bar(x, vals, width, color=colors[:len(labels)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("|Spearman rho|")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        # Add value labels
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_one_experiment(pdf, label, da1_grid_path, pc1_grid_path, corr_path,
                        scribble_path=None, prompt=None):
    """One page per experiment: DA1 vs PC1 grids (top) + raw vs orth heatmaps (bottom)."""
    title = label
    if prompt:
        title += f'\nprompt: "{prompt}"'

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    has_scribble = scribble_path and os.path.exists(scribble_path)

    # Use nested gridspecs: top row adapts to scribble, bottom always 2 cols
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    outer = GridSpec(2, 1, figure=fig, hspace=0.3, height_ratios=[1, 1.3],
                     top=0.92, bottom=0.04)
    if has_scribble:
        gs_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0],
                                         width_ratios=[0.6, 1, 1], wspace=0.15)
    else:
        gs_top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.15)
    gs_bot = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.25)

    if has_scribble:
        ax_scr = fig.add_subplot(gs_top[0, 0])
        ax_scr.imshow(Image.open(scribble_path))
        ax_scr.set_title("Scribble", fontsize=11, fontweight="bold")
        ax_scr.axis("off")
        col_da, col_pc = 1, 2
    else:
        col_da, col_pc = 0, 1

    ax_da = fig.add_subplot(gs_top[0, col_da])
    if os.path.exists(da1_grid_path):
        ax_da.imshow(Image.open(da1_grid_path))
    else:
        ax_da.text(0.5, 0.5, "DA1 grid not found", ha="center", va="center",
                   transform=ax_da.transAxes)
    ax_da.set_title("DA1 sorted: Top-8 vs Bottom-8", fontsize=11, fontweight="bold")
    ax_da.axis("off")

    ax_pc = fig.add_subplot(gs_top[0, col_pc])
    if os.path.exists(pc1_grid_path):
        ax_pc.imshow(Image.open(pc1_grid_path))
    else:
        ax_pc.text(0.5, 0.5, "PC1 grid not found", ha="center", va="center",
                   transform=ax_pc.transAxes)
    ax_pc.set_title("PC1 sorted: Top-8 vs Bottom-8", fontsize=11, fontweight="bold")
    ax_pc.axis("off")

    ax_raw = fig.add_subplot(gs_bot[0, 0])
    ax_orth = fig.add_subplot(gs_bot[0, 1])

    if os.path.exists(corr_path):
        data = load_corr(corr_path)
        mat = np.array(data["clip_corr"])
        names = data["clip_names"]
        plot_heatmap(ax_raw, mat, names, "Raw correlations")

        if "clip_orth_corr" in data:
            orth_mat = np.array(data["clip_orth_corr"])
            orth_names = data["clip_orth_names"]
            plot_heatmap(ax_orth, orth_mat, orth_names, "Orthogonalized probes")
        else:
            ax_orth.text(0.5, 0.5, "No orthogonalized data", ha="center",
                         va="center", transform=ax_orth.transAxes)
            ax_orth.set_title("Orthogonalized probes")
    else:
        ax_raw.text(0.5, 0.5, "Corr table not found", ha="center",
                    va="center", transform=ax_raw.transAxes)
        ax_orth.text(0.5, 0.5, "Corr table not found", ha="center",
                     va="center", transform=ax_orth.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_all_images(pdf, label, all_images_path):
    """Full page showing all N generated images as a grid."""
    fig = plt.figure(figsize=(16, 16))
    fig.suptitle(f"{label} — All Generated Images", fontsize=14, fontweight="bold")

    ax = fig.add_subplot(111)
    if os.path.exists(all_images_path):
        img = Image.open(all_images_path)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f"Not found:\n{os.path.basename(all_images_path)}",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
    ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    out_path = os.path.join(BASE, "correlation_analysis.pdf")

    # (label, da1_grid, pc1_grid, corr_json, scribble, prompt, all_images)
    experiments = [
        ("Circle",
         os.path.join(BASE, "clip_da_clip_cn_circle_dt0.001_da1_grid.png"),
         os.path.join(BASE, "clip_cn_circle_dt0.001_pc1_grid.png"),
         os.path.join(BASE, "corr_table_clip_cn_circle_dt0.001.json"),
         os.path.join(DATA, "circle_scribble.png"),
         "a superrealistic professional photograph of",
         os.path.join(BASE, "all_images_clip_cn_circle_dt0.001.png")),
        ("Face (DPS)",
         os.path.join(BASE, "clip_da_clip_cn_dps_face_dt0.001_da1_grid.png"),
         os.path.join(BASE, "clip_cn_dps_face_dt0.001_pc1_grid.png"),
         os.path.join(BASE, "corr_table_clip_cn_dps_face_dt0.001.json"),
         os.path.join(DATA, "dps_face_scribble.png"),
         "a superrealistic professional photograph of",
         os.path.join(BASE, "all_images_clip_cn_dps_face_dt0.001.png")),
        ("Square",
         os.path.join(AUTO, "clip_da_autoresearch_square_da1_grid.png"),
         os.path.join(AUTO, "autoresearch_square_pc1_grid.png"),
         os.path.join(AUTO, "corr_table_autoresearch_square.json"),
         os.path.join(DATA, "face_boxed_scribble.png"),
         "a street portrait",
         os.path.join(AUTO, "all_images_autoresearch_square.png")),
    ]

    with PdfPages(out_path) as pdf:
        for label, da1, pc1, corr, scribble, prompt, all_img in experiments:
            print(f"{label}...")
            page_one_experiment(pdf, label, da1, pc1, corr,
                                scribble_path=scribble, prompt=prompt)
            page_all_images(pdf, label, all_img)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
