#!/usr/bin/env python3
"""
Generate comparison figures and tables for SSA truncation redefinition.

Figures:
  1. GMM sweep curves: S_old vs S_hat (4 panels, mean ± 1 SD)
  2. SDXL guidance table: 8 prompts × 4 scales, S_old vs S_hat
  3. Gaussian baseline bar chart
  4. Correlation scatter: S_old vs S_hat across all GMM sweep points

Tables:
  - SDXL guidance: Spearman rank correlation S_old vs S_hat
  - Summary statistics
"""
import sys
import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

CODEDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.join(CODEDIR, '..')
RESDIR = os.path.join(BASEDIR, 'results')
FIGDIR = os.path.join(BASEDIR, 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def load_json(name):
    path = os.path.join(RESDIR, name)
    with open(path) as f:
        return json.load(f)


# ── Figure 1: GMM Sweep Curves ──

def fig_gmm_sweeps():
    """4-panel: S_old vs S_hat curves with ±1 SD bands."""
    data = load_json('gmm_sweeps_redef.json')
    sweep_names = ['sep_2d', 'var_2d', 'weight_2d', 'nmodes_fixdist_2d']
    sweep_labels = ['Separation', 'Variance', 'Weight asymmetry', 'Number of modes']

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))

    for i, name in enumerate(sweep_names):
        ax = axes[i]
        d = data[name]
        params = np.array(d['param_values'])
        n_seeds = d['n_seeds']

        # Aggregate across seeds
        s_old_all = np.array([s['s_old'] for s in d['seeds']])  # (n_seeds, n_params)
        s_hat_all = np.array([s['s_hat'] for s in d['seeds']])
        mi_all = np.array([s['mi'] for s in d['seeds']])

        mean_old = s_old_all.mean(axis=0)
        std_old = s_old_all.std(axis=0)
        mean_hat = s_hat_all.mean(axis=0)
        std_hat = s_hat_all.std(axis=0)
        mean_mi = mi_all.mean(axis=0)

        # S_old (solid blue)
        ax.plot(params, mean_old, '-', color='tab:blue', lw=1.5, label='$S_{\\mathrm{old}}$')
        ax.fill_between(params, mean_old - std_old, mean_old + std_old,
                         alpha=0.15, color='tab:blue')

        # S_hat (dashed red)
        ax.plot(params, mean_hat, '--', color='tab:red', lw=1.5, label='$\\hat{S}$')
        ax.fill_between(params, mean_hat - std_hat, mean_hat + std_hat,
                         alpha=0.15, color='tab:red')

        # MI on twin axis
        ax2 = ax.twinx()
        ax2.plot(params, mean_mi, 's--', color='tab:green', ms=3, lw=1, alpha=0.7, label='MI')
        ax2.set_ylabel('MI (nats)', fontsize=10, color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        ax.set_xlabel(d['param_name'], fontsize=11)
        if i == 0:
            ax.set_ylabel('SSA', fontsize=11)
        ax.set_title(sweep_labels[i], fontsize=11)
        ax.grid(True, alpha=0.3)

        # Legend (only first panel)
        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

    plt.tight_layout()
    out = os.path.join(FIGDIR, 'fig_gmm_sweeps_redef.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


# ── Figure 2: S_old vs S_hat scatter across all GMM points ──

def fig_gmm_scatter():
    """Scatter plot: S_old vs S_hat for all GMM sweep points (all seeds)."""
    data = load_json('gmm_sweeps_redef.json')
    sweep_names = ['sep_2d', 'var_2d', 'weight_2d', 'nmodes_fixdist_2d']

    all_old = []
    all_hat = []
    for name in sweep_names:
        d = data[name]
        for seed_data in d['seeds']:
            all_old.extend(seed_data['s_old'])
            all_hat.extend(seed_data['s_hat'])

    all_old = np.array(all_old)
    all_hat = np.array(all_hat)

    rho, pval = spearmanr(all_old, all_hat)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(all_old, all_hat, s=8, alpha=0.4, edgecolors='none')
    lo = min(all_old.min(), all_hat.min()) * 0.95
    hi = max(all_old.max(), all_hat.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, lw=1)
    ax.set_xlabel('$S_{\\mathrm{old}}$', fontsize=12)
    ax.set_ylabel('$\\hat{S}$', fontsize=12)
    ax.set_title(f'GMM Sweeps: $S_{{\\mathrm{{old}}}}$ vs $\\hat{{S}}$\n'
                 f'Spearman $\\rho$ = {rho:.4f}', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGDIR, 'fig_gmm_scatter_redef.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")
    print(f"  Spearman ρ = {rho:.4f} (p = {pval:.2e})")


# ── Figure 3: SDXL Guidance Table ──

def fig_sdxl_table():
    """Bar chart or table: S_old vs S_hat for 8 prompts × 4 scales."""
    data = load_json('sdxl_guidance_redef.json')

    prompts = ['person', 'cat', 'mountains', 'flowers', 'street', 'food', 'elderly', 'car']
    gs_keys = ['gs_1p0', 'gs_3p0', 'gs_7p5', 'gs_15p0']
    gs_labels = ['1.0', '3.0', '7.5', '15.0']

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for pi, prompt in enumerate(prompts):
        ax = axes[pi]
        s_old_vals = []
        s_hat_vals = []
        for gs_key in gs_keys:
            r = data[prompt][gs_key]
            s_old_vals.append(r['s_old_fixed20'])
            s_hat_vals.append(r['s_hat'])

        x = np.arange(len(gs_labels))
        width = 0.35
        ax.bar(x - width/2, s_old_vals, width, label='$S_{\\mathrm{old}}$', color='tab:blue', alpha=0.8)
        ax.bar(x + width/2, s_hat_vals, width, label='$\\hat{S}$', color='tab:red', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(gs_labels, fontsize=8)
        ax.set_title(prompt, fontsize=10)
        ax.set_ylim(0.4, 0.75)
        ax.grid(True, alpha=0.3, axis='y')
        if pi == 0:
            ax.legend(fontsize=7)
        if pi % 4 == 0:
            ax.set_ylabel('SSA', fontsize=10)
        if pi >= 4:
            ax.set_xlabel('Guidance scale', fontsize=9)

    plt.suptitle('SDXL Guidance Sweep: $S_{\\mathrm{old}}$ vs $\\hat{S}$', fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIGDIR, 'fig_sdxl_guidance_redef.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")

    # Compute Spearman across all (prompt, gs) pairs
    all_old = []
    all_hat = []
    for prompt in prompts:
        for gs_key in gs_keys:
            r = data[prompt][gs_key]
            all_old.append(r['s_old_fixed20'])
            all_hat.append(r['s_hat'])
    rho, _ = spearmanr(all_old, all_hat)
    print(f"  SDXL Spearman ρ(S_old, S_hat) = {rho:.4f}")

    # Check: does gs=7.5 still peak for all prompts?
    n_peak = 0
    for prompt in prompts:
        vals = [data[prompt][gs_key]['s_hat'] for gs_key in gs_keys]
        peak_idx = np.argmax(vals)
        if peak_idx == 2:  # gs=7.5 is index 2
            n_peak += 1
    print(f"  gs=7.5 peaks for S_hat: {n_peak}/8 prompts")


# ── Figure 4: Gaussian Baseline ──

def fig_gaussian_baseline():
    """Bar chart: S_old vs S_hat for Gaussian (should be near zero)."""
    data = load_json('gaussian_baseline.json')

    configs = ['numerical_2d', 'numerical_10d']
    labels = ['2D Gaussian', '10D Gaussian']

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(configs))
    width = 0.35
    s_old_vals = [data[c]['numerical']['s_old'] for c in configs]
    s_hat_vals = [data[c]['numerical']['s_hat'] for c in configs]
    analytic_vals = [data[c]['analytic']['s_hat'] for c in configs]

    bars1 = ax.bar(x - width/2, s_old_vals, width, label='$S_{\\mathrm{old}}$ (numerical)', color='tab:blue')
    bars2 = ax.bar(x + width/2, s_hat_vals, width, label='$\\hat{S}$ (numerical)', color='tab:red')

    # Add analytic markers
    for i, av in enumerate(analytic_vals):
        ax.plot(x[i] + width/2, av, 'k*', ms=10, zorder=5)
    ax.plot([], [], 'k*', ms=10, label='$\\hat{S}$ (analytic)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('SSA')
    ax.set_title('Gaussian Baseline: $S_{\\mathrm{old}}$ vs $\\hat{S}$ (unimodal)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    out = os.path.join(FIGDIR, 'fig_gaussian_baseline.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


# ── Summary Report ──

def write_summary():
    """Write a text summary of all results."""
    lines = []
    lines.append("=" * 70)
    lines.append("SSA TRUNCATION REDEFINITION — SUMMARY REPORT")
    lines.append("=" * 70)

    # SDXL
    sdxl = load_json('sdxl_guidance_redef.json')
    prompts = ['person', 'cat', 'mountains', 'flowers', 'street', 'food', 'elderly', 'car']
    gs_keys = ['gs_1p0', 'gs_3p0', 'gs_7p5', 'gs_15p0']

    lines.append("\n1. SDXL Guidance Sweep (8 prompts × 4 scales)")
    lines.append("-" * 50)
    lines.append(f"{'Prompt':>12s} {'gs':>5s} {'S_old':>8s} {'S_hat':>8s} {'τ*':>8s} {'L*':>5s}")
    for prompt in prompts:
        for gs_key in gs_keys:
            r = sdxl[prompt][gs_key]
            gs_val = gs_key.replace('gs_', '').replace('p', '.')
            lines.append(f"{prompt:>12s} {gs_val:>5s} {r['s_old_fixed20']:8.4f} "
                         f"{r['s_hat']:8.4f} {r['tau_star']:8.2f} {r['L_star']:5d}")

    all_old = [sdxl[p][g]['s_old_fixed20'] for p in prompts for g in gs_keys]
    all_hat = [sdxl[p][g]['s_hat'] for p in prompts for g in gs_keys]
    rho_sdxl, _ = spearmanr(all_old, all_hat)
    lines.append(f"\nSpearman ρ(S_old, S_hat) = {rho_sdxl:.4f}")
    n_peak = sum(1 for p in prompts if np.argmax([sdxl[p][g]['s_hat'] for g in gs_keys]) == 2)
    lines.append(f"gs=7.5 peaks for S_hat: {n_peak}/8 prompts")
    lines.append(f"NOTE: τ* = T for all configs (noise floor never reached, N=200, d=768)")

    # GMM (if available)
    gmm_path = os.path.join(RESDIR, 'gmm_sweeps_redef.json')
    if os.path.exists(gmm_path):
        gmm = load_json('gmm_sweeps_redef.json')
        lines.append("\n\n2. GMM Sweeps (4 sweeps × 10 seeds)")
        lines.append("-" * 50)

        all_old_gmm = []
        all_hat_gmm = []
        for name in ['sep_2d', 'var_2d', 'weight_2d', 'nmodes_fixdist_2d']:
            d = gmm[name]
            for seed_data in d['seeds']:
                all_old_gmm.extend(seed_data['s_old'])
                all_hat_gmm.extend(seed_data['s_hat'])

            # Per-sweep summary
            s_old_mean = np.mean([np.mean(s['s_old']) for s in d['seeds']])
            s_hat_mean = np.mean([np.mean(s['s_hat']) for s in d['seeds']])
            tau_star_mean = np.mean([np.mean(s['tau_star']) for s in d['seeds']])
            lines.append(f"  {name}: mean S_old={s_old_mean:.4f}, mean S_hat={s_hat_mean:.4f}, "
                         f"mean τ*={tau_star_mean:.3f}")

        rho_gmm, _ = spearmanr(all_old_gmm, all_hat_gmm)
        lines.append(f"\nSpearman ρ(S_old, S_hat) = {rho_gmm:.4f}")

    # Gaussian (if available)
    gauss_path = os.path.join(RESDIR, 'gaussian_baseline.json')
    if os.path.exists(gauss_path):
        gauss = load_json('gaussian_baseline.json')
        lines.append("\n\n3. Gaussian Baseline")
        lines.append("-" * 50)
        for key in ['numerical_2d', 'numerical_10d']:
            r = gauss[key]
            lines.append(f"  {key}: S_old={r['numerical']['s_old']:.6f}, "
                         f"S_hat={r['numerical']['s_hat']:.6f}, "
                         f"analytic S_hat={r['analytic']['s_hat']:.6f}")

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)
    out_path = os.path.join(BASEDIR, 'SUMMARY.md')
    with open(out_path, 'w') as f:
        f.write(report)
    print(f"\nSaved summary to {out_path}")
    print(report)


def main():
    # Always try SDXL (already computed locally)
    sdxl_path = os.path.join(RESDIR, 'sdxl_guidance_redef.json')
    if os.path.exists(sdxl_path):
        fig_sdxl_table()

    # GMM (requires cluster results)
    gmm_path = os.path.join(RESDIR, 'gmm_sweeps_redef.json')
    if os.path.exists(gmm_path):
        fig_gmm_sweeps()
        fig_gmm_scatter()
    else:
        print("GMM results not yet available (waiting for cluster)")

    # Gaussian (requires cluster results)
    gauss_path = os.path.join(RESDIR, 'gaussian_baseline.json')
    if os.path.exists(gauss_path):
        fig_gaussian_baseline()
    else:
        print("Gaussian baseline not yet available (waiting for cluster)")

    # Always write summary with whatever data is available
    write_summary()


if __name__ == '__main__':
    main()
