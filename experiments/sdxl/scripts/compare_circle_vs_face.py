"""
compare_circle_vs_face.py — Compare circle vs DPS face across LC and CLIP modes.

Reads results from both results_controlnet/ (LC-mode) and results_clip_da/ (CLIP-mode),
generates comparison figures and summary table.

Usage:
    python scripts/compare_circle_vs_face.py \
        --lc_dir results_controlnet \
        --clip_dir results_clip_da \
        --output_dir figures_comparison
"""

import os
import sys
import json
import argparse

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare circle vs DPS face SSA/DA across LC and CLIP modes"
    )
    p.add_argument("--lc_dir", type=str, default="results_controlnet",
                   help="Directory with LC-mode results")
    p.add_argument("--clip_dir", type=str, default="results_clip_da",
                   help="Directory with CLIP-mode results")
    p.add_argument("--output_dir", type=str, default="figures_comparison")
    return p.parse_args()


def main():
    args = parse_args()

    lc_dir = os.path.join(EXP_DIR, args.lc_dir) if not os.path.isabs(args.lc_dir) else args.lc_dir
    clip_dir = os.path.join(EXP_DIR, args.clip_dir) if not os.path.isabs(args.clip_dir) else args.clip_dir
    output_dir = os.path.join(EXP_DIR, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Load LC-mode results ---
    lc_configs = {
        "circle": "sdxl_ssa_sdxl_ssa_cn_circle",
        "dps_face": "sdxl_ssa_sdxl_ssa_cn_dps_face",
    }
    lc_results = {}
    lc_da = {}
    for label, name in lc_configs.items():
        json_path = os.path.join(lc_dir, f"sdxl_ssa_{name}.json")
        npz_path = os.path.join(lc_dir, f"sdxl_da_{name}.npz")
        if os.path.exists(json_path):
            lc_results[label] = load_json(json_path)
            print(f"  LC {label}: SSA={lc_results[label]['ssa']:.4f}")
        if os.path.exists(npz_path):
            lc_da[label] = dict(np.load(npz_path))

    # --- Load CLIP-mode results ---
    clip_configs = {
        "circle": "clip_cn_circle",
        "dps_face": "clip_cn_dps_face",
    }
    clip_results = {}
    clip_da = {}
    for label, name in clip_configs.items():
        json_path = os.path.join(clip_dir, f"clip_da_{name}.json")
        npz_path = os.path.join(clip_dir, f"clip_da_{name}.npz")
        if os.path.exists(json_path):
            clip_results[label] = load_json(json_path)
            print(f"  CLIP {label}: SSA={clip_results[label]['ssa_clip']:.4f}")
        if os.path.exists(npz_path):
            clip_da[label] = dict(np.load(npz_path))

    # --- Load interpretation results ---
    interp_results = {}
    for label, name in lc_configs.items():
        interp_path = os.path.join(clip_dir, f"interpret_da_{name}.json")
        if os.path.exists(interp_path):
            interp_results[label] = load_json(interp_path)

    # --- Summary table ---
    print("\n" + "=" * 80)
    print("COMPARISON: Circle vs DPS Face — LC (512D) vs CLIP (768D)")
    print("=" * 80)

    header = f"{'Metric':<30} {'Circle LC':>12} {'Circle CLIP':>12} {'Face LC':>12} {'Face CLIP':>12}"
    print(header)
    print("-" * len(header))

    def val(d, key, fmt=".4f"):
        v = d.get(key)
        if v is None:
            return "—"
        return f"{v:{fmt}}"

    # SSA
    row = f"{'SSA':.<30}"
    row += f" {val(lc_results.get('circle', {}), 'ssa'):>12}"
    row += f" {val(clip_results.get('circle', {}), 'ssa_clip'):>12}"
    row += f" {val(lc_results.get('dps_face', {}), 'ssa'):>12}"
    row += f" {val(clip_results.get('dps_face', {}), 'ssa_clip'):>12}"
    print(row)

    # Entropy
    row = f"{'Entropy':.<30}"
    row += f" {val(lc_results.get('circle', {}), 'entropy', '.1f'):>12}"
    row += f" {val(clip_results.get('circle', {}), 'entropy_clip', '.1f'):>12}"
    row += f" {val(lc_results.get('dps_face', {}), 'entropy', '.1f'):>12}"
    row += f" {val(clip_results.get('dps_face', {}), 'entropy_clip', '.1f'):>12}"
    print(row)

    # DA1 gap
    row = f"{'DA1 gap':.<30}"
    row += f" {val(lc_results.get('circle', {}), 'da1_gap', '.2f'):>12}"
    row += f" {val(clip_results.get('circle', {}), 'da1_gap', '.2f'):>12}"
    row += f" {val(lc_results.get('dps_face', {}), 'da1_gap', '.2f'):>12}"
    row += f" {val(clip_results.get('dps_face', {}), 'da1_gap', '.2f'):>12}"
    print(row)

    # DA1 optimal lag
    row = f"{'DA1 optimal lag':.<30}"
    row += f" {val(lc_results.get('circle', {}), 'da1_optimal_lag', 'd'):>12}"
    row += f" {val(clip_results.get('circle', {}), 'da1_optimal_lag', 'd'):>12}"
    row += f" {val(lc_results.get('dps_face', {}), 'da1_optimal_lag', 'd'):>12}"
    row += f" {val(clip_results.get('dps_face', {}), 'da1_optimal_lag', 'd'):>12}"
    print(row)

    # DA2 gap
    row = f"{'DA2 gap':.<30}"
    row += f" {val(lc_results.get('circle', {}), 'da2_gap', '.2f'):>12}"
    row += f" {val(clip_results.get('circle', {}), 'da2_gap', '.2f'):>12}"
    row += f" {val(lc_results.get('dps_face', {}), 'da2_gap', '.2f'):>12}"
    row += f" {val(clip_results.get('dps_face', {}), 'da2_gap', '.2f'):>12}"
    print(row)

    # Cross-space DA1 Spearman (from CLIP-mode results)
    for label in ["circle", "dps_face"]:
        if label in clip_results and "cross_space_da1_spearman" in clip_results[label]:
            print(f"\n  Cross-space DA1 Spearman ({label}): "
                  f"{clip_results[label]['cross_space_da1_spearman']:.4f}")

    # DA1 text probe cosines (CLIP-mode only — direct, in-space cosine)
    for label in ["circle", "dps_face"]:
        if label in clip_results and "da1_cosines" in clip_results[label]:
            print(f"\n  DA1 text probe cosines ({label}, CLIP-mode):")
            for probe, cos_val in clip_results[label]["da1_cosines"].items():
                print(f"    {probe:<15} cos = {cos_val:>7.4f}")

    # LC-DA1 interpretation via post-hoc CLIP probes
    for label in ["circle", "dps_face"]:
        if label in interp_results:
            corr = interp_results[label].get("da1_correlations_full", {})
            if corr:
                print(f"\n  LC-DA1 post-hoc CLIP probe Spearman ({label}):")
                for probe, r in corr.items():
                    print(f"    {probe:<15} rho = {r['rho']:>7.4f} (p={r['pvalue']:.2e})")

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot 1: rho^2 decay comparison (4 curves: circle/face x LC/CLIP)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, label, title in [(axes[0], "circle", "Circle"),
                                  (axes[1], "dps_face", "DPS Face")]:
            if label in lc_da:
                lags = lc_da[label]["snapshot_lags"]
                rho = lc_da[label]["rho_sq"]
                ax.plot(lags, rho, "o-", label=f"LC (512D)", color="tab:orange")
            if label in clip_da:
                lags = clip_da[label]["snapshot_lags"]
                rho = clip_da[label]["rho_sq"]
                ax.plot(lags, rho, "s-", label=f"CLIP (768D)", color="tab:blue")
            ax.set_xlabel("Lag (Langevin steps)")
            ax.set_ylabel("rho(tau)^2")
            ax.set_title(f"{title}: Autocorrelation Decay")
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "compare_rho_decay.png"), dpi=150)
        plt.close(fig)
        print(f"\n  Saved: compare_rho_decay.png")

        # Plot 2: Gap profiles (4 panels)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for col, (label, title) in enumerate([("circle", "Circle"), ("dps_face", "DPS Face")]):
            for row, (mode, mode_label, data_dict) in enumerate([
                ("lc", "LC (512D)", lc_da), ("clip", "CLIP (768D)", clip_da)
            ]):
                ax = axes[row, col]
                if label in data_dict:
                    lags = data_dict[label]["snapshot_lags"]
                    ax.plot(lags, data_dict[label]["gap12"], "o-", label="gap12 (DA1)")
                    ax.plot(lags, data_dict[label]["gap23"], "s-", label="gap23 (DA2)")
                ax.set_xlabel("Lag")
                ax.set_ylabel("Eigenvalue ratio")
                ax.set_title(f"{title} — {mode_label}")
                ax.legend()
                ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "compare_gap_profiles.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: compare_gap_profiles.png")

        # Plot 3: Bar chart — SSA comparison
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        conditions = []
        values = []
        colors = []
        for label, title in [("circle", "Circle"), ("dps_face", "DPS Face")]:
            if label in lc_results:
                conditions.append(f"{title}\nLC (512D)")
                values.append(lc_results[label]["ssa"])
                colors.append("tab:orange")
            if label in clip_results:
                conditions.append(f"{title}\nCLIP (768D)")
                values.append(clip_results[label]["ssa_clip"])
                colors.append("tab:blue")
        ax.bar(range(len(conditions)), values, color=colors)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions)
        ax.set_ylabel("SSA")
        ax.set_title("SSA: Circle vs DPS Face — LC vs CLIP")
        ax.set_ylim(0, 1)
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "compare_ssa_bar.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: compare_ssa_bar.png")

    except Exception as e:
        print(f"  Plotting failed: {e}")

    print(f"\nDone! Results in {output_dir}")


if __name__ == "__main__":
    main()
