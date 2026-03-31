"""
visualize_results.py — Plot SDXL SSA results: guidance sweep + base vs turbo.

Produces:
1. SSA and entropy vs guidance scale (dual y-axis)
2. rho_sq decay curves for all guidance scales
3. DA gap profiles
4. Base vs Turbo comparison bar chart

Usage:
    python scripts/visualize_results.py
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FIGURES_DIR = os.path.join(EXP_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_guidance_sweep():
    """Load all guidance sweep results."""
    results_dir = os.path.join(EXP_DIR, "results_guidance_sweep")
    guidance_values = [1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0]
    results = []
    npz_data = []

    for w in guidance_values:
        json_path = os.path.join(results_dir, f"sdxl_ssa_sdxl_base_w{w}.json")
        npz_path = os.path.join(results_dir, f"sdxl_da_sdxl_base_w{w}.npz")

        with open(json_path) as f:
            r = json.load(f)
        results.append(r)

        d = np.load(npz_path, allow_pickle=True)
        npz_data.append(d)

    return guidance_values, results, npz_data


def load_base_vs_turbo():
    """Load base vs turbo results."""
    results_dir = os.path.join(EXP_DIR, "results_base_vs_turbo")
    out = {}
    for name in ["sdxl_base", "sdxl_turbo"]:
        json_path = os.path.join(results_dir, f"sdxl_ssa_{name}.json")
        npz_path = os.path.join(results_dir, f"sdxl_da_{name}.npz")
        with open(json_path) as f:
            out[name] = {"json": json.load(f), "npz": np.load(npz_path, allow_pickle=True)}
    return out


def plot_guidance_sweep(guidance_values, results, npz_data):
    """Figure 1: SSA and entropy vs guidance scale."""
    ssas = [r["ssa"] for r in results]
    entropies = [r["entropy"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    color_ssa = "#2196F3"
    color_ent = "#FF5722"

    ax1.plot(guidance_values, ssas, "o-", color=color_ssa, linewidth=2,
             markersize=8, label="SSA", zorder=3)
    ax1.set_xlabel("Guidance scale $w$")
    ax1.set_ylabel("SSA", color=color_ssa)
    ax1.tick_params(axis="y", labelcolor=color_ssa)

    ax2 = ax1.twinx()
    ax2.plot(guidance_values, entropies, "s--", color=color_ent, linewidth=2,
             markersize=8, label="kNN Entropy", zorder=3)
    ax2.set_ylabel("kNN Entropy (CLIP 512D)", color=color_ent)
    ax2.tick_params(axis="y", labelcolor=color_ent)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")

    ax1.set_title("SDXL Guidance Sweep: SSA vs Entropy in CLIP Space")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig_guidance_ssa_entropy.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_rho_sq_curves(guidance_values, results, npz_data):
    """Figure 2: rho_sq decay curves for all guidance scales."""
    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.cm.viridis
    colors = [cmap(i / (len(guidance_values) - 1)) for i in range(len(guidance_values))]

    for i, (w, d) in enumerate(zip(guidance_values, npz_data)):
        lags = d["snapshot_lags"]
        rho_sq = d["rho_sq"]
        # Physical time = lag * dt
        phys_t = lags * 0.02
        ax.plot(phys_t, rho_sq, "o-", color=colors[i], linewidth=1.5,
                markersize=4, label=f"w={w}")

    ax.set_xlabel("Physical time ($\\tau \\cdot dt$)")
    ax.set_ylabel("$\\rho(\\tau)^2$")
    ax.set_title("Autocorrelation Decay: SDXL Guidance Sweep")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig_guidance_rho_sq_curves.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_da_gaps(guidance_values, results, npz_data):
    """Figure 3: DA gap profiles for all guidance scales."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cmap = plt.cm.viridis
    colors = [cmap(i / (len(guidance_values) - 1)) for i in range(len(guidance_values))]

    for i, (w, d) in enumerate(zip(guidance_values, npz_data)):
        lags = d["snapshot_lags"]
        phys_t = lags * 0.02
        gap12 = d["gap12"]
        gap23 = d["gap23"]

        ax1.plot(phys_t, gap12, "o-", color=colors[i], linewidth=1.5,
                 markersize=4, label=f"w={w}")
        ax2.plot(phys_t, gap23, "o-", color=colors[i], linewidth=1.5,
                 markersize=4, label=f"w={w}")

    ax1.set_xlabel("Physical time")
    ax1.set_ylabel("$\\lambda_1 / \\lambda_2$")
    ax1.set_title("DA1 Gap Profile")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Physical time")
    ax2.set_ylabel("$\\lambda_2 / \\lambda_3$")
    ax2.set_title("DA2 Gap Profile")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("DA Eigenvalue Gap Profiles: SDXL Guidance Sweep", fontsize=14)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig_guidance_da_gaps.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_base_vs_turbo(bvt):
    """Figure 4: Base vs Turbo comparison."""
    names = ["Base (w=7.5)", "Turbo (w=0)"]
    keys = ["sdxl_base", "sdxl_turbo"]

    ssas = [bvt[k]["json"]["ssa"] for k in keys]
    entropies = [bvt[k]["json"]["entropy"] for k in keys]
    da1_gaps = [bvt[k]["json"]["da1_gap"] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    bar_colors = ["#2196F3", "#FF9800"]

    axes[0].bar(names, ssas, color=bar_colors)
    axes[0].set_ylabel("SSA")
    axes[0].set_title("SSA")

    axes[1].bar(names, entropies, color=bar_colors)
    axes[1].set_ylabel("kNN Entropy")
    axes[1].set_title("Entropy")

    axes[2].bar(names, da1_gaps, color=bar_colors)
    axes[2].set_ylabel("DA1 Gap ($\\lambda_1/\\lambda_2$)")
    axes[2].set_title("DA1 Gap")

    fig.suptitle("SDXL Base vs Turbo", fontsize=14)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig_base_vs_turbo.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_summary_table(guidance_values, results, bvt):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("GUIDANCE SWEEP RESULTS")
    print("=" * 80)
    print(f"{'w':>6} | {'SSA':>8} | {'Entropy':>10} | {'DA1 gap':>8} | {'DA1 lag':>8} | {'DA2 gap':>8} | {'var_f':>8}")
    print("-" * 80)
    for w, r in zip(guidance_values, results):
        print(f"{w:>6.1f} | {r['ssa']:>8.4f} | {r['entropy']:>10.1f} | "
              f"{r['da1_gap']:>8.2f} | {r['da1_optimal_lag']:>8d} | "
              f"{r['da2_gap']:>8.2f} | {r['var_f']:>8.3f}")

    print("\n" + "=" * 80)
    print("BASE vs TURBO")
    print("=" * 80)
    for name, key in [("Base (w=7.5)", "sdxl_base"), ("Turbo (w=0)", "sdxl_turbo")]:
        r = bvt[key]["json"]
        print(f"{name:>15} | SSA={r['ssa']:.4f} | Entropy={r['entropy']:.1f} | "
              f"DA1 gap={r['da1_gap']:.2f} | var_f={r['var_f']:.3f}")


def main():
    print("Loading results...")
    guidance_values, results, npz_data = load_guidance_sweep()
    bvt = load_base_vs_turbo()

    plot_summary_table(guidance_values, results, bvt)
    plot_guidance_sweep(guidance_values, results, npz_data)
    plot_rho_sq_curves(guidance_values, results, npz_data)
    plot_da_gaps(guidance_values, results, npz_data)
    plot_base_vs_turbo(bvt)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
