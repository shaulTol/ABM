"""Plot multi-prompt guidance sweep results.

Reads per-(prompt, guidance) JSONs from results/ and makes:
1. Per-prompt dual-axis bar charts (SSA vs entropy) — same style as fig_sdxl_ssa_vs_entropy
2. Aggregate figure showing argmax-SSA and argmax-entropy guidance scales across prompts
3. Summary table

Usage:
    python fig_sdxl_guidance.py [--results_dir results]
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROMPTS = [
    "a photograph of a cat sitting on a windowsill",
    "a landscape photograph of mountains at sunset",
    "a still life painting of flowers in a vase",
    "a street photograph of a busy city intersection",
    "a photograph of a plate of food at a restaurant",
    "a portrait of an elderly person",
    "a photograph of a car parked on a street",
]
PROMPT_LABELS = ["cat", "mountains", "flowers", "street", "food", "elderly", "car"]

GS_VALUES = [1.0, 3.0, 7.5, 15.0]
GS_LABELS = ["gs1p0", "gs3p0", "gs7p5", "gs15p0"]


def load_data(results_dir):
    """Return dict[prompt_label] = {gs: {'ssa_fixed20', 'H_k5_d50', 'entropy_suite'}}."""
    data = {}
    for pi, plabel in enumerate(PROMPT_LABELS):
        data[plabel] = {}
        for gs, gslabel in zip(GS_VALUES, GS_LABELS):
            name = f"mp_p{pi}_{plabel}_{gslabel}"
            ssa_path = os.path.join(results_dir, f"clip_da_{name}.json")
            ent_path = os.path.join(results_dir, f"gen_{name}_entropy.json")
            entry = {}
            if os.path.exists(ssa_path):
                d = json.load(open(ssa_path))
                entry["ssa_fixed20"] = d.get("ssa_fixed20")
                entry["ssa_clip"] = d.get("ssa_clip")
                entry["ssa_log"] = d.get("ssa_log")
            if os.path.exists(ent_path):
                d = json.load(open(ent_path))
                entry["H_k5_d50"] = d["entropy"]["knn"]["k5"]["50"]
                entry["H_k3_d50"] = d["entropy"]["knn"]["k3"]["50"]
                entry["H_k5_d20"] = d["entropy"]["knn"]["k5"]["20"]
            data[plabel][gs] = entry
    return data


def plot_per_prompt(data, out_dir):
    """One dual-axis bar chart per prompt."""
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
    })

    for pi, plabel in enumerate(PROMPT_LABELS):
        d = data[plabel]
        ssa_vals = [d[gs].get("ssa_fixed20") for gs in GS_VALUES]
        ent_vals = [d[gs].get("H_k5_d50") for gs in GS_VALUES]
        if any(v is None for v in ssa_vals + ent_vals):
            print(f"  [skip {plabel}] missing results")
            continue

        x = np.arange(len(GS_VALUES))
        w = 0.28

        ssa_base = min(ssa_vals) - 0.02
        ssa_top = max(ssa_vals) + 0.02
        ent_base = min(ent_vals) - 5
        ent_top = max(ent_vals) + 5

        fig, ax1 = plt.subplots(figsize=(5.5, 2.8))
        ax2 = ax1.twinx()

        b1 = ax1.bar(x - w/2 - 0.02, np.array(ssa_vals) - ssa_base, w, bottom=ssa_base,
                     color="steelblue", label="SSA")
        b2 = ax2.bar(x + w/2 + 0.02, np.array(ent_vals) - ent_base, w, bottom=ent_base,
                     color="firebrick", label=r"$k$NN entropy")

        ax1.set_ylabel("SSA", color="steelblue")
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax1.set_ylim(ssa_base, ssa_top)
        ax2.set_ylabel(r"$k$NN entropy", color="firebrick")
        ax2.tick_params(axis="y", labelcolor="firebrick")
        ax2.set_ylim(ent_base, ent_top)

        ax1.set_xticks(x)
        ax1.set_xticklabels([str(g) for g in GS_VALUES])
        ax1.set_xlabel(r"Guidance scale $w$")
        ax1.set_title(f'"{PROMPTS[pi]}"', fontsize=9, style="italic")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   ncol=2, frameon=False)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        out = os.path.join(out_dir, f"fig_mp_p{pi}_{plabel}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


def plot_aggregate(data, out_dir):
    """7-panel grid (one per prompt) + argmax summary."""
    plt.rcParams.update({
        "font.size": 8,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
    })

    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    axes = axes.flatten()

    for pi, plabel in enumerate(PROMPT_LABELS):
        ax1 = axes[pi]
        ax2 = ax1.twinx()
        d = data[plabel]
        ssa_vals = [d[gs].get("ssa_fixed20") for gs in GS_VALUES]
        ent_vals = [d[gs].get("H_k5_d50") for gs in GS_VALUES]
        if any(v is None for v in ssa_vals + ent_vals):
            ax1.text(0.5, 0.5, "(pending)", ha="center", va="center", transform=ax1.transAxes)
            ax1.set_title(plabel, fontsize=9)
            continue

        x = np.arange(len(GS_VALUES))
        w = 0.28
        ssa_base = min(ssa_vals) - 0.02
        ent_base = min(ent_vals) - 3

        ax1.bar(x - w/2 - 0.02, np.array(ssa_vals) - ssa_base, w, bottom=ssa_base,
                color="steelblue")
        ax2.bar(x + w/2 + 0.02, np.array(ent_vals) - ent_base, w, bottom=ent_base,
                color="firebrick")

        ax1.set_ylim(ssa_base, max(ssa_vals) + 0.015)
        ax2.set_ylim(ent_base, max(ent_vals) + 3)
        ax1.tick_params(axis="y", labelcolor="steelblue", labelsize=7)
        ax2.tick_params(axis="y", labelcolor="firebrick", labelsize=7)
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(g) for g in GS_VALUES], fontsize=7)
        ax1.set_title(plabel, fontsize=9)
        argmax_ssa = GS_VALUES[int(np.argmax(ssa_vals))]
        argmax_H = GS_VALUES[int(np.argmax(ent_vals))]
        ax1.text(0.02, 0.98, f"SSA*={argmax_ssa}\nH*={argmax_H}",
                 transform=ax1.transAxes, fontsize=7, va="top",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

    # 8th panel: bar chart of argmax SSA vs argmax entropy counts
    ax = axes[7]
    ssa_peaks = []
    ent_peaks = []
    for plabel in PROMPT_LABELS:
        d = data[plabel]
        sv = [d[gs].get("ssa_fixed20") for gs in GS_VALUES]
        ev = [d[gs].get("H_k5_d50") for gs in GS_VALUES]
        if any(v is None for v in sv + ev):
            continue
        ssa_peaks.append(GS_VALUES[int(np.argmax(sv))])
        ent_peaks.append(GS_VALUES[int(np.argmax(ev))])
    if ssa_peaks:
        counts_ssa = [ssa_peaks.count(g) for g in GS_VALUES]
        counts_ent = [ent_peaks.count(g) for g in GS_VALUES]
        x = np.arange(len(GS_VALUES))
        w = 0.38
        ax.bar(x - w/2, counts_ssa, w, color="steelblue", label="argmax SSA")
        ax.bar(x + w/2, counts_ent, w, color="firebrick", label=r"argmax $k$NN entropy")
        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in GS_VALUES])
        ax.set_ylabel("# prompts")
        ax.set_title("Aggregate peak locations")
        ax.legend(fontsize=7)

    plt.suptitle("Multi-prompt guidance sweep: SSA vs entropy per prompt",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig_mp_aggregate.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def print_table(data):
    print("\n=== Summary Table ===")
    header = f"{'prompt':<12}" + "".join(f"{'SSA '+str(g):>12}" for g in GS_VALUES) + \
             "".join(f"{'H '+str(g):>10}" for g in GS_VALUES) + f"{'SSA*':>8}{'H*':>8}"
    print(header)
    print("-" * len(header))
    for plabel in PROMPT_LABELS:
        d = data[plabel]
        sv = [d[gs].get("ssa_fixed20") for gs in GS_VALUES]
        ev = [d[gs].get("H_k5_d50") for gs in GS_VALUES]
        if any(v is None for v in sv + ev):
            print(f"{plabel:<12}  (pending)")
            continue
        row = f"{plabel:<12}"
        for v in sv: row += f"{v:>12.4f}"
        for v in ev: row += f"{v:>10.2f}"
        row += f"{GS_VALUES[int(np.argmax(sv))]:>8}"
        row += f"{GS_VALUES[int(np.argmax(ev))]:>8}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    data = load_data(args.results_dir)
    print_table(data)
    plot_per_prompt(data, out_dir)
    plot_aggregate(data, out_dir)


if __name__ == "__main__":
    main()
