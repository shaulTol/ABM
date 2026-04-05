"""
plot_dps_comparison.py — Compare 3 DPS scribbles: SSA, gender-sliced SSA, entropy.

Produces a multi-page PDF:
  Page 1: Scribble images + DA1-sorted grids (one row per experiment)
  Page 2: Correlation heatmaps (one per experiment, both dt values)
  Page 3: SSA decay curves overlay (3 experiments × 2 dt)
  Page 4: Summary bar charts — SSA, sliced SSA, Gaussian entropy, kNN entropy

Usage:
    python scripts/plot_dps_comparison.py \
        --results_dir results_clip_da \
        --output_path results_clip_da/dps_comparison.pdf
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

matplotlib.rcParams.update({"font.size": 9})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(EXP_DIR, "data")


# --- Experiment definitions ---
EXPERIMENTS = [
    {
        "label": "DPS Face (zeta5)",
        "scribble": os.path.join(DATA_DIR, "dps_face_scribble.png"),
        "name_dt001": "dps_face_dt001",
        "name_dt005": "dps_face_dt005",
    },
    {
        "label": "DPS First (seed 1)",
        "scribble": os.path.join(DATA_DIR, "dps_first_scribble.jpeg"),
        "name_dt001": "dps_first_dt001",
        "name_dt005": "dps_first_dt005",
    },
    {
        "label": "DPS Second (seed 2)",
        "scribble": os.path.join(DATA_DIR, "dps_second_scribble.jpeg"),
        "name_dt001": "dps_second_dt001",
        "name_dt005": "dps_second_dt005",
    },
]


def load_json_safe(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_npz_safe(path):
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    return None


def short_name(name):
    for prefix in ("CLIP_", "LC_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def plot_heatmap(ax, mat, names, title):
    n = len(names)
    short = [short_name(nm) for nm in names]
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_title(title, fontsize=9, fontweight="bold")
    for i in range(n):
        for j in range(n):
            val = mat[i][j]
            color = "white" if abs(val) > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5, color=color)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str,
                   default=os.path.join(EXP_DIR, "results_clip_da"))
    p.add_argument("--output_path", type=str, default=None)
    args = p.parse_args()

    rd = args.results_dir
    out_path = args.output_path or os.path.join(rd, "dps_comparison.pdf")

    # Collect data
    all_data = {}
    for exp in EXPERIMENTS:
        for dt_key in ["dt001", "dt005"]:
            name = exp[f"name_{dt_key}"]
            json_path = os.path.join(rd, f"clip_da_{name}.json")
            npz_path = os.path.join(rd, f"clip_da_{name}.npz")
            corr_path = os.path.join(rd, f"corr_table_{name}.json")
            entropy_path = os.path.join(rd, f"entropy_{name}.json")
            grid_path = os.path.join(rd, f"clip_da_{name}_da1_grid.png")

            all_data[name] = {
                "json": load_json_safe(json_path),
                "npz": load_npz_safe(npz_path),
                "corr": load_json_safe(corr_path),
                "entropy": load_json_safe(entropy_path),
                "grid_path": grid_path,
                "label": exp["label"],
                "scribble": exp["scribble"],
                "dt_key": dt_key,
            }

    with PdfPages(out_path) as pdf:
        # ==================================================================
        # Page 1: Scribble + DA1 grid for each experiment (dt=0.001)
        # ==================================================================
        fig, axes = plt.subplots(3, 2, figsize=(12, 14))
        fig.suptitle("DPS Scribble Comparison — Scribbles & DA1 Grids (dt=0.001)",
                      fontsize=13, fontweight="bold")

        for row, exp in enumerate(EXPERIMENTS):
            name = exp["name_dt001"]
            d = all_data[name]

            # Scribble
            ax = axes[row, 0]
            if os.path.exists(d["scribble"]):
                img = Image.open(d["scribble"])
                ax.imshow(img)
            ax.set_title(f"{exp['label']}\nScribble", fontsize=9)
            ax.axis("off")

            # DA1 grid
            ax = axes[row, 1]
            if os.path.exists(d["grid_path"]):
                img = Image.open(d["grid_path"])
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Grid not found", ha="center", va="center",
                        transform=ax.transAxes)
            ssa_str = ""
            if d["json"]:
                ssa_str = f"SSA={d['json'].get('ssa_clip', '?'):.4f}"
                sliced = d["json"].get("ssa_gender_sliced")
                if sliced is not None:
                    ssa_str += f"  sliced={sliced:.4f}"
            ax.set_title(f"DA1-sorted grid\n{ssa_str}", fontsize=9)
            ax.axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        print("  Page 1: Scribbles & DA1 grids")

        # ==================================================================
        # Page 2: Correlation heatmaps (dt=0.001)
        # ==================================================================
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        fig.suptitle("CLIP-Space Correlation Heatmaps (dt=0.001)",
                      fontsize=13, fontweight="bold")

        for col, exp in enumerate(EXPERIMENTS):
            name = exp["name_dt001"]
            d = all_data[name]
            ax = axes[col]
            if d["corr"] and "clip_matrix" in d["corr"]:
                mat = np.array(d["corr"]["clip_matrix"])
                names = d["corr"]["clip_names"]
                plot_heatmap(ax, mat, names, exp["label"])
            else:
                ax.text(0.5, 0.5, "No corr data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(exp["label"])

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        print("  Page 2: Correlation heatmaps")

        # ==================================================================
        # Page 3: SSA rho² decay curves (all 6 conditions)
        # ==================================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        dt_labels = {"dt001": "dt=0.001", "dt005": "dt=0.005"}

        for ax_idx, dt_key in enumerate(["dt001", "dt005"]):
            ax = axes[ax_idx]
            for i, exp in enumerate(EXPERIMENTS):
                name = exp[f"name_{dt_key}"]
                d = all_data[name]
                if d["npz"] and "rho_sq" in d["npz"]:
                    rho_sq = d["npz"]["rho_sq"]
                    lags = d["npz"]["snapshot_lags"]
                    ssa = float(np.mean(rho_sq))
                    ax.plot(lags, rho_sq, "o-", color=colors[i], markersize=3,
                            label=f"{exp['label']} (SSA={ssa:.4f})")

                    # Gender-sliced overlay
                    if "rho_sq_gender" in d["npz"]:
                        rho_g = d["npz"]["rho_sq_gender"]
                        ssa_g = float(np.mean(rho_g))
                        ax.plot(lags[:len(rho_g)], rho_g, "s--", color=colors[i],
                                markersize=2, alpha=0.6,
                                label=f"  gender-sliced ({ssa_g:.4f})")

            ax.set_xlabel("Lag")
            ax.set_ylabel("ρ²(τ)")
            ax.set_title(f"SSA Decay — {dt_labels[dt_key]}", fontweight="bold")
            ax.legend(fontsize=7, loc="upper right")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Autocorrelation Decay: 3 DPS Scribbles",
                      fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        print("  Page 3: SSA decay curves")

        # ==================================================================
        # Page 4: Summary bar charts
        # ==================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Metric Summary: 3 DPS Scribbles × 2 dt Values",
                      fontsize=13, fontweight="bold")

        bar_width = 0.25
        x = np.arange(3)  # 3 experiments
        exp_labels = [e["label"] for e in EXPERIMENTS]

        # Helper to get metric value
        def get_metric(exp_idx, dt_key, metric_fn):
            name = EXPERIMENTS[exp_idx][f"name_{dt_key}"]
            d = all_data[name]
            return metric_fn(d)

        # 4a: SSA (full)
        ax = axes[0, 0]
        for di, dt_key in enumerate(["dt001", "dt005"]):
            vals = []
            for ei in range(3):
                v = get_metric(ei, dt_key,
                               lambda d: d["json"].get("ssa_clip") if d["json"] else None)
                vals.append(v if v is not None else 0)
            ax.bar(x + di * bar_width, vals, bar_width,
                   label=dt_labels[dt_key], alpha=0.8)
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(exp_labels, fontsize=8)
        ax.set_ylabel("SSA")
        ax.set_title("Full SSA (CLIP 768D)", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # 4b: Gender-sliced SSA
        ax = axes[0, 1]
        for di, dt_key in enumerate(["dt001", "dt005"]):
            vals = []
            for ei in range(3):
                v = get_metric(ei, dt_key,
                               lambda d: d["json"].get("ssa_gender_sliced") if d["json"] else None)
                vals.append(v if v is not None else 0)
            ax.bar(x + di * bar_width, vals, bar_width,
                   label=dt_labels[dt_key], alpha=0.8)
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(exp_labels, fontsize=8)
        ax.set_ylabel("SSA (gender-sliced)")
        ax.set_title("Gender-Sliced SSA", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # 4c: Gaussian entropy (k=10)
        ax = axes[1, 0]
        for di, dt_key in enumerate(["dt001", "dt005"]):
            vals = []
            for ei in range(3):
                v = get_metric(ei, dt_key,
                               lambda d: d["entropy"].get("gaussian_entropy", {}).get("10")
                               if d["entropy"] else None)
                vals.append(v if v is not None else 0)
            ax.bar(x + di * bar_width, vals, bar_width,
                   label=dt_labels[dt_key], alpha=0.8)
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(exp_labels, fontsize=8)
        ax.set_ylabel("H_gauss (k=10)")
        ax.set_title("Gaussian Entropy (10 PCs)", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # 4d: kNN entropy (k_pca=10)
        ax = axes[1, 1]
        for di, dt_key in enumerate(["dt001", "dt005"]):
            vals = []
            for ei in range(3):
                v = get_metric(ei, dt_key,
                               lambda d: d["entropy"].get("knn_entropy", {}).get("10")
                               if d["entropy"] else None)
                vals.append(v if v is not None else 0)
            ax.bar(x + di * bar_width, vals, bar_width,
                   label=dt_labels[dt_key], alpha=0.8)
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(exp_labels, fontsize=8)
        ax.set_ylabel("H_knn (k_pca=10)")
        ax.set_title("kNN Entropy (10 PCs)", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        print("  Page 4: Summary bar charts")

        # ==================================================================
        # Page 5: Entropy comparison table
        # ==================================================================
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")
        fig.suptitle("Entropy Estimates — All Methods & Dimensions",
                      fontsize=13, fontweight="bold")

        # Build table data
        col_headers = ["Experiment", "dt"]
        for k in [5, 10, 20, 50]:
            col_headers.append(f"H_gauss\nk={k}")
        for k in [5, 10, 20, 50]:
            col_headers.append(f"H_knn\nk={k}")
        for k in [2, 5]:
            col_headers.append(f"H_kde\nk={k}")

        table_data = []
        for exp in EXPERIMENTS:
            for dt_key in ["dt001", "dt005"]:
                name = exp[f"name_{dt_key}"]
                d = all_data[name]
                row = [exp["label"], "0.001" if dt_key == "dt001" else "0.005"]
                ent = d["entropy"] if d["entropy"] else {}
                for k in [5, 10, 20, 50]:
                    v = ent.get("gaussian_entropy", {}).get(str(k))
                    row.append(f"{v:.2f}" if v is not None else "—")
                for k in [5, 10, 20, 50]:
                    v = ent.get("knn_entropy", {}).get(str(k))
                    row.append(f"{v:.2f}" if v is not None else "—")
                for k in [2, 5]:
                    v = ent.get("kde_entropy", {}).get(str(k))
                    row.append(f"{v:.2f}" if v is not None else "—")
                table_data.append(row)

        if table_data:
            table = ax.table(cellText=table_data, colLabels=col_headers,
                             loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1.0, 1.4)
            # Header styling
            for j in range(len(col_headers)):
                cell = table[0, j]
                cell.set_facecolor("#4472C4")
                cell.set_text_props(color="white", fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No entropy data available yet",
                    ha="center", va="center", transform=ax.transAxes, fontsize=14)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        print("  Page 5: Entropy table")

    print(f"\nSaved PDF: {out_path}")


if __name__ == "__main__":
    main()
