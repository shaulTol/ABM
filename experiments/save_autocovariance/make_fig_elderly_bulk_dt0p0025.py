"""Eigenvalue-distribution figure at tau*=5.0 for the new elderly DA run
(dt=0.0025, T=2000, N=200, free-conv stopping criterion never triggered).

Three panels at tau*=5.0 (=T*dt):
  (a) free-conv density with naive sigma^2_f
  (b) free-conv density with peel-fitted sigma^2_bulk
  (c) Monte-Carlo null edge

C0 reused from results/elderly_gs7p5_n200_C_tau_full.npz (same x0 latents).
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'noise_floor'))
from free_conv_utils import edges as fc_edges, density as fc_density

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Reuse the helpers from the existing companion script
sys.path.insert(0, os.path.dirname(__file__))
from make_fig_sdxl_elderly_bulk import peel_fit, mc_null_edge, N_MC_REPS

REPO = '/Users/stolk/github/ssa-paper'
NEW_NPZ = f'{REPO}/experiments/sdxl/results_da_pcavda/clip_da_elderly_gs7p5_fc_dt0p0025.npz'
OLD_C0_NPZ = f'{REPO}/experiments/save_autocovariance/results/elderly_gs7p5_n200_C_tau_full.npz'
OUT_DIR = f'{REPO}/experiments/save_autocovariance/results'


def main():
    new = np.load(NEW_NPZ)
    eigs = np.array(new['eigenvalues_full_at_tau_star']).astype(np.float64).copy()
    sigma_f_sq = float(new['freeconv_sigma_f_sq'])
    gamma = float(new['freeconv_gamma'])
    tau_star_lag = int(new['tau_star_freeconv'])

    # dt and T are not stored in the npz; pull from JSON sidecar
    cfg = json.load(open(NEW_NPZ.replace('.npz', '.json')))
    dt = float(cfg['dt']); T = int(cfg['T'])
    N = int(cfg['n_samples'])
    d = int(cfg['embed_dim'])
    tau_star = tau_star_lag * dt

    # Sort eigenvalues descending (compute_ssa_da already did, but be safe)
    eigs = np.sort(np.abs(eigs))[::-1].copy()

    # Reuse C0 from the previous elderly run (same x0 -> same CLIP encoding -> same C0)
    old = np.load(OLD_C0_NPZ)
    C0 = old['C0']
    assert C0.shape == (d, d), f'C0 shape mismatch: {C0.shape} vs ({d},{d})'
    eigs_C0 = np.linalg.eigvalsh(C0)[::-1].copy()
    sigma_naive = np.sqrt(sigma_f_sq)

    print(f'tau* = {tau_star_lag} steps × {dt} = {tau_star:.4f} (phys time)')
    print(f'd={d}, N={N}, gamma={gamma:.4f}, sigma_f^2={sigma_f_sq:.6e}')

    fit = peel_fit(eigs, eigs_C0, tau_star, gamma, sigma_naive)
    print(f"  peel: {fit['n_spikes']} spikes above naive lp, sigma_bulk^2={fit['sigma_bulk_sq']:.3e}")
    print(f"  bulk edges: lambda+={fit['lp_bulk']:.4e}, lambda-={fit['lm_bulk']:.4e}")

    print(f'  Computing MC null ({N_MC_REPS} reps)...')
    mc = mc_null_edge(C0, tau_star, N, n_reps=N_MC_REPS)
    print(f"  MC edge: mean={mc['mc_edge_mean']:.4e}, 95%={mc['mc_edge_95']:.4e}")

    # Rank-deficiency atom at lambda=0
    w0 = max(0.0, 1.0 - 2.0 / gamma)
    tr_C0 = np.trace(C0)
    tol = 1e-10 * tr_C0
    n_near_zero = int(np.sum(np.abs(eigs) < tol))

    # === Figure ===
    lm, lp = fit['lm_bulk'], fit['lp_bulk']
    lm_naive, lp_naive = fit['lm_naive'], fit['lp_naive']
    sw = lp - lm
    sw_naive = lp_naive - lm_naive
    mc_edge = mc['mc_edge_95']
    lam1 = float(eigs[0])

    pad_naive = 0.3 * sw_naive
    lam_grid_naive = np.linspace(lm_naive - pad_naive, lp_naive + pad_naive, 800)
    rho_grid_naive = fc_density(lam_grid_naive, tau_star, sigma_naive, gamma=gamma)

    pad = 0.3 * sw
    lam_grid_bulk = np.linspace(lm - pad, lp + pad, 800)
    rho_grid_bulk = fc_density(lam_grid_bulk, tau_star, fit['sigma_bulk'], gamma=gamma)

    # Linear-x histogram bins
    eig_min, eig_max = float(eigs.min()), float(eigs.max())
    n_bins = 60
    bins = np.linspace(eig_min - 0.02 * (eig_max - eig_min),
                       eig_max + 0.02 * (eig_max - eig_min), n_bins)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    plt.subplots_adjust(wspace=0.25, left=0.05, right=0.97, top=0.82, bottom=0.22)
    panel_configs = [
        (r'(a) Free-conv ($\sigma^2_f$, naive)', 'naive', '#E67E22'),
        (r'(b) Free-conv ($\sigma^2_{\mathrm{bulk}}$, peel)', 'bulk', '#C0392B'),
        ('(c) Monte-Carlo null', 'mc', '#27AE60'),
    ]

    for ax_i, (title, kind, color) in enumerate(panel_configs):
        ax = axes[ax_i]
        ax.hist(eigs, bins=bins, density=True, color='#7BAFD4', alpha=0.65,
                edgecolor='white', linewidth=0.3, zorder=2)
        if kind == 'naive':
            ax.plot(lam_grid_naive, rho_grid_naive, color=color, lw=2.2, zorder=4)
            ax.axvline(lp_naive, color=color, ls='--', lw=1.2, zorder=3)
            ax.axvline(lm_naive, color=color, ls='--', lw=1.0, zorder=3, alpha=0.6)
            if w0 > 0:
                bw = bins[1] - bins[0]
                ax.bar(0, w0 / bw, width=bw, color=color, alpha=0.35,
                       edgecolor=color, linewidth=1.2, zorder=6)
        elif kind == 'bulk':
            ax.plot(lam_grid_bulk, rho_grid_bulk, color=color, lw=2.2, zorder=4)
            ax.axvline(lp, color=color, ls='--', lw=1.2, zorder=3)
            ax.axvline(lm, color=color, ls='--', lw=1.0, zorder=3, alpha=0.6)
            if w0 > 0:
                bw = bins[1] - bins[0]
                ax.bar(0, w0 / bw, width=bw, color=color, alpha=0.35,
                       edgecolor=color, linewidth=1.2, zorder=6)
        else:  # mc
            ax.axvline(mc_edge, color=color, ls='-.', lw=2.0, zorder=5)
            ax.axvline(mc['mc_edge_mean'], color=color, ls=':', lw=1.2,
                       zorder=4, alpha=0.6)
        ax.axvline(lam1, color='#C0392B', ls='-', lw=1.0, zorder=5, alpha=0.7)
        ax.set_yscale('symlog', linthresh=100)
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(r'$\lambda$', fontsize=9)
        if ax_i == 0:
            ax.set_ylabel('Density (symlog)', fontsize=9)
        ax.tick_params(labelsize=7)

    # Legends
    leg_a = [
        Patch(facecolor='#7BAFD4', alpha=0.65, edgecolor='white', label='Eigenvalues'),
        Line2D([0], [0], color='#E67E22', lw=2.2, label=r'$\rho_\tau^\gamma$ density'),
        Line2D([0], [0], color='#E67E22', ls='--', lw=1.2,
               label=fr'$\lambda_\pm$  ($\sigma^2_f={sigma_f_sq:.2e}$)'),
        Patch(facecolor='#E67E22', alpha=0.35, edgecolor='#E67E22',
              label=fr'$\delta_0$ atom (mass={w0:.3f})'),
        Line2D([0], [0], color='#C0392B', lw=1.0, alpha=0.7,
               label=fr'$\lambda_1={lam1:.2e}$ (DA1)'),
    ]
    leg_b = [
        Patch(facecolor='#7BAFD4', alpha=0.65, edgecolor='white', label='Eigenvalues'),
        Line2D([0], [0], color='#C0392B', lw=2.2, label=r'$\rho_\tau^\gamma$ density'),
        Line2D([0], [0], color='#C0392B', ls='--', lw=1.2,
               label=fr'$\lambda_\pm$  ($\sigma^2_{{\mathrm{{bulk}}}}={fit["sigma_bulk_sq"]:.2e}$)'),
        Patch(facecolor='#C0392B', alpha=0.35, edgecolor='#C0392B',
              label=fr'$\delta_0$ atom (mass={w0:.3f})'),
        Line2D([0], [0], color='#C0392B', lw=1.0, alpha=0.7,
               label=fr'$\lambda_1={lam1:.2e}$ (DA1)'),
    ]
    leg_c = [
        Patch(facecolor='#7BAFD4', alpha=0.65, edgecolor='white', label='Eigenvalues'),
        Line2D([0], [0], color='#27AE60', ls='-.', lw=2.0,
               label=f'MC 95th pctl = {mc_edge:.2e}'),
        Line2D([0], [0], color='#27AE60', ls=':', lw=1.2, alpha=0.6,
               label=f'MC mean = {mc["mc_edge_mean"]:.2e}'),
        Line2D([0], [0], color='#C0392B', lw=1.0, alpha=0.7,
               label=fr'$\lambda_1={lam1:.2e}$ (DA1)'),
    ]
    axes[0].legend(handles=leg_a, fontsize=6.5, loc='upper center',
                   bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True)
    axes[1].legend(handles=leg_b, fontsize=6.5, loc='upper center',
                   bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True)
    axes[2].legend(handles=leg_c, fontsize=6.5, loc='upper center',
                   bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
    fig.suptitle(fr'SDXL elderly ($w=7.5$, dt=0.0025), CLIP space '
                 fr'($d={d}$, $N={N}$, $\gamma={gamma:.1f}$, '
                 fr'$\tau^*={tau_star:.3f}$)',
                 fontsize=11, y=0.96)

    out_png = os.path.join(OUT_DIR, 'fig_sdxl_elderly_bulk_dt0p0025.png')
    fig.savefig(out_png, dpi=200)
    print(f'\nSaved {out_png}')

    # === symlog-x version ===
    linthresh_x = 5e-4
    def symlog(x, lt=linthresh_x):
        return np.sign(x) * np.log10(1 + np.abs(x) / lt)
    def inv_symlog(y, lt=linthresh_x):
        return np.sign(y) * lt * (10 ** np.abs(y) - 1)

    bin_lo = eig_min - 0.02 * (eig_max - eig_min)
    bin_hi = eig_max + 0.02 * (eig_max - eig_min)
    n_bins_sl = 100
    bins_sl = inv_symlog(np.linspace(symlog(bin_lo), symlog(bin_hi), n_bins_sl + 1))

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5.2))
    plt.subplots_adjust(wspace=0.25, left=0.05, right=0.97, top=0.82, bottom=0.22)
    for ax_i, (title, kind, color) in enumerate(panel_configs):
        ax = axes2[ax_i]
        ax.hist(eigs, bins=bins_sl, density=True, color='#7BAFD4', alpha=0.65,
                edgecolor='white', linewidth=0.3, zorder=2)
        if kind == 'naive':
            ax.plot(lam_grid_naive, rho_grid_naive, color=color, lw=2.2, zorder=4)
            ax.axvline(lp_naive, color=color, ls='--', lw=1.2, zorder=3)
            ax.axvline(lm_naive, color=color, ls='--', lw=1.0, zorder=3, alpha=0.6)
        elif kind == 'bulk':
            ax.plot(lam_grid_bulk, rho_grid_bulk, color=color, lw=2.2, zorder=4)
            ax.axvline(lp, color=color, ls='--', lw=1.2, zorder=3)
            ax.axvline(lm, color=color, ls='--', lw=1.0, zorder=3, alpha=0.6)
        else:
            ax.axvline(mc_edge, color=color, ls='-.', lw=2.0, zorder=5)
            ax.axvline(mc['mc_edge_mean'], color=color, ls=':', lw=1.2, zorder=4, alpha=0.6)
        ax.axvline(lam1, color='#C0392B', lw=1.0, zorder=5, alpha=0.7)
        ax.set_xscale('symlog', linthresh=linthresh_x)
        ax.set_yscale('symlog', linthresh=100)
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(r'$\lambda$ (symlog)', fontsize=9)
        if ax_i == 0:
            ax.set_ylabel('Density (symlog)', fontsize=9)
        ax.tick_params(labelsize=7)

    axes2[0].legend(handles=leg_a, fontsize=6.5, loc='upper center',
                    bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
    axes2[1].legend(handles=leg_b, fontsize=6.5, loc='upper center',
                    bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
    axes2[2].legend(handles=leg_c, fontsize=6.5, loc='upper center',
                    bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
    fig2.suptitle(fr'SDXL elderly ($w=7.5$, dt=0.0025), CLIP space '
                  fr'($d={d}$, $N={N}$, $\gamma={gamma:.1f}$, '
                  fr'$\tau^*={tau_star:.3f}$)  [symlog x]',
                  fontsize=11, y=0.96)
    out_png2 = os.path.join(OUT_DIR, 'fig_sdxl_elderly_bulk_dt0p0025_symlogx.png')
    fig2.savefig(out_png2, dpi=200)
    print(f'Saved {out_png2}')

    # JSON sidecar
    out_json = os.path.join(OUT_DIR, 'fig_sdxl_elderly_bulk_dt0p0025.json')
    json.dump({
        'tau_star': tau_star, 'tau_star_lag': tau_star_lag, 'dt': dt, 'T': T,
        'N': N, 'd': d, 'gamma': gamma,
        'sigma_naive_sq': sigma_f_sq,
        'sigma_bulk_sq': fit['sigma_bulk_sq'],
        'lp_naive': fit['lp_naive'], 'lm_naive': fit['lm_naive'],
        'lp_bulk': fit['lp_bulk'], 'lm_bulk': fit['lm_bulk'],
        'n_spikes_naive': fit['n_spikes'],
        'spike_values': fit['spike_vals'].tolist(),
        'mc_null': mc,
        'lambda_1': lam1, 'lambda_2': float(eigs[1]),
        'gap12': float(eigs[0] / eigs[1]) if eigs[1] > 0 else None,
        'lambda_1_above_mc_edge': bool(lam1 > mc_edge),
        'lambda_1_above_lp_naive': bool(lam1 > fit['lp_naive']),
        'lambda_1_above_lp_bulk': bool(lam1 > fit['lp_bulk']),
        'atom_mass_w0': w0,
        'n_near_zero_empirical': n_near_zero,
        'all_eigenvalues': eigs.tolist(),
    }, open(out_json, 'w'), indent=2)
    print(f'Saved {out_json}')


if __name__ == '__main__':
    main()
