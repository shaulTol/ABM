#!/usr/bin/env python3
"""
Bulk-fit validation figure for SDXL elderly (w=7.5) — dt=0.0025 free-conv stopping run.

Same format as make_fig_sdxl_elderly_bulk.py but using:
  - Eigenvalues from clip_da_elderly_gs7p5_fc_dt0p0025.npz (τ*=5.0s)
  - C(0) from the companion elderly_gs7p5_n200_C_tau_full.npz (same x0)

Output:
  results/fig_sdxl_elderly_bulk_dt0p0025.png
  results/fig_sdxl_elderly_bulk_dt0p0025_symlogx.png
  results/fig_sdxl_elderly_bulk_dt0p0025.json
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'noise_floor'))
from free_conv_utils import edges as fc_edges, density as fc_density

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RESDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
N_MC_REPS = 200


def peel_fit(eigs_tau, eigs_C0, tau, gamma, sigma_naive):
    """One-iteration peel: identify spikes with naive sigma^2, refit on bulk."""
    d = len(eigs_C0)
    tr_C0 = eigs_C0.sum()

    lm_naive, lp_naive = fc_edges(tau, sigma_naive, gamma=gamma)
    spike_mask = eigs_tau > lp_naive
    n_spikes = int(spike_mask.sum())
    spike_vals = eigs_tau[spike_mask]

    if n_spikes > 0:
        sigma_bulk_sq = (tr_C0 - eigs_C0[:n_spikes].sum()) / d
    else:
        sigma_bulk_sq = tr_C0 / d
    sigma_bulk = np.sqrt(max(sigma_bulk_sq, 1e-20))
    lm, lp = fc_edges(tau, sigma_bulk, gamma=gamma)

    n_below = int(np.sum(eigs_tau < lm))
    below_vals = eigs_tau[eigs_tau < lm]

    return {
        'n_spikes': n_spikes,
        'spike_vals': spike_vals,
        'sigma_naive_sq': sigma_naive**2,
        'sigma_bulk_sq': float(sigma_bulk_sq),
        'sigma_bulk': sigma_bulk,
        'lp_naive': lp_naive,
        'lm_naive': lm_naive,
        'lp_bulk': lp,
        'lm_bulk': lm,
        'n_below_lm': n_below,
        'below_lm_vals': below_vals.tolist(),
    }


def mc_null_edge(C0, tau, N, n_reps=N_MC_REPS, seed=42):
    """Monte-Carlo null: max eigenvalue of C_sym(tau) under iid OU draws."""
    d = C0.shape[0]
    rng = np.random.RandomState(seed)

    eigvals_C0, eigvecs_C0 = np.linalg.eigh(C0)
    eigvals_C0 = np.maximum(eigvals_C0, 0)
    sqrt_C0 = eigvecs_C0 * np.sqrt(eigvals_C0)[np.newaxis, :]

    rho = np.exp(-tau / 2.0)
    noise_scale = np.sqrt(max(1.0 - np.exp(-tau), 1e-20))

    max_eigs = []
    for _ in range(n_reps):
        z0 = rng.randn(N, d) @ sqrt_C0.T
        z_noise = rng.randn(N, d) @ sqrt_C0.T
        x_tau = rho * z0 + noise_scale * z_noise

        mu = z0.mean(axis=0)
        z0_c = z0 - mu
        x_tau_c = x_tau - mu

        C_raw = (x_tau_c.T @ z0_c) / N
        C_sym = 0.5 * (C_raw + C_raw.T)

        top_eig = np.linalg.eigvalsh(C_sym)[-1]
        max_eigs.append(top_eig)

    max_eigs = np.array(max_eigs)
    return {
        'mc_edge_mean': float(max_eigs.mean()),
        'mc_edge_95': float(np.percentile(max_eigs, 95)),
        'mc_edge_max': float(max_eigs.max()),
        'mc_edge_all': max_eigs.tolist(),
        'n_reps': n_reps,
    }


def main():
    # Load dt=0.0025 eigenvalues
    fc_path = os.path.join(os.path.dirname(__file__), '..',
                           'sdxl', 'results_da_pcavda',
                           'clip_da_elderly_gs7p5_fc_dt0p0025.npz')
    if not os.path.exists(fc_path):
        print(f"ERROR: {fc_path} not found")
        sys.exit(1)

    fc_data = np.load(fc_path, allow_pickle=True)
    eigs_raw = fc_data['eigenvalues_full_at_tau_star']  # signed, 768
    C_tau_star = fc_data['C_tau_star']

    # Sort descending (signed — keep negatives)
    eigs = np.sort(eigs_raw)[::-1].copy()

    dt = 0.0025
    T = 2000
    tau_star = dt * T  # 5.0s

    # Load C0 from companion file (same x0)
    c0_path = os.path.join(RESDIR, 'elderly_gs7p5_n200_C_tau_full.npz')
    c0_data = np.load(c0_path)
    C0 = c0_data['C0']
    sigma_f_sq = float(c0_data['sigma_f_sq'])
    d = int(c0_data['d'])
    N = int(c0_data['N'])

    gamma = d / N
    sigma_naive = np.sqrt(sigma_f_sq)
    eigs_C0 = np.linalg.eigvalsh(C0)[::-1].copy()

    print(f"dt=0.0025 run: d={d}, N={N}, gamma={gamma:.4f}, sigma_f^2={sigma_f_sq:.6f}")
    print(f"  tau*={tau_star:.1f}s (dt={dt}, T={T})")
    print(f"  Eigenvalue range: [{eigs[-1]:.4e}, {eigs[0]:.4e}]")
    print(f"  #negative: {np.sum(eigs < 0)}, #near-zero: {np.sum(np.abs(eigs) < 1e-8)}")

    fit = peel_fit(eigs, eigs_C0, tau_star, gamma, sigma_naive)
    print(f"  Peel: {fit['n_spikes']} above naive lp, sigma_bulk^2={fit['sigma_bulk_sq']:.2e}")
    print(f"  lambda_+={fit['lp_bulk']:.4e}, lambda_-={fit['lm_bulk']:.4e}")
    print(f"  {fit['n_below_lm']} below lambda_-")

    # MC null edge
    print(f"\n  Computing MC null ({N_MC_REPS} reps)...")
    mc = mc_null_edge(C0, tau_star, N, n_reps=N_MC_REPS)
    print(f"  MC edge: mean={mc['mc_edge_mean']:.4e}, 95%={mc['mc_edge_95']:.4e}")

    # Rank-deficiency atom
    w0 = max(0.0, 1.0 - 2.0 / gamma)
    tr_C0 = np.trace(C0)
    tol = 1e-10 * tr_C0
    n_near_zero = int(np.sum(np.abs(eigs) < tol))
    frac_near_zero = n_near_zero / d
    print(f"\n  Rank-deficiency atom (gamma={gamma:.2f} > 1):")
    print(f"    Predicted atom mass: w0 = {w0:.4f}")
    print(f"    Empirical |lambda| < {tol:.2e}: {n_near_zero}/{d} = {frac_near_zero:.4f}")

    # === Precompute density curves ===
    lm, lp = fit['lm_bulk'], fit['lp_bulk']
    support_width = lp - lm
    lm_naive, lp_naive = fit['lm_naive'], fit['lp_naive']
    sw_naive = lp_naive - lm_naive
    mc_edge = mc['mc_edge_95']
    lam1 = eigs[0]

    pad_naive = 0.3 * sw_naive
    lam_grid_naive = np.linspace(lm_naive - pad_naive, lp_naive + pad_naive, 800)
    rho_grid_naive = fc_density(lam_grid_naive, tau_star, sigma_naive, gamma=gamma)

    pad = 0.3 * support_width
    lam_grid_bulk = np.linspace(lm - pad, lp + pad, 800)
    rho_grid_bulk = fc_density(lam_grid_bulk, tau_star, fit['sigma_bulk'], gamma=gamma)

    # Shared histogram bins
    eig_min, eig_max = eigs.min(), eigs.max()
    bin_lo = eig_min - 0.02 * (eig_max - eig_min)
    bin_hi = eig_max + 0.02 * (eig_max - eig_min)
    n_bins = 60
    bins = np.linspace(bin_lo, bin_hi, n_bins)

    # === Figure 1: linear x ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    plt.subplots_adjust(wspace=0.25, left=0.05, right=0.97, top=0.82, bottom=0.22)

    panel_configs = [
        {'title': r'(a) Free-conv. ($\sigma^2_f$, naive)', 'overlay': 'naive', 'color': '#E67E22'},
        {'title': r'(b) Free-conv. ($\sigma^2_{\mathrm{bulk}}$, peel)', 'overlay': 'bulk', 'color': '#C0392B'},
        {'title': '(c) Monte-Carlo null', 'overlay': 'mc', 'color': '#27AE60'},
    ]

    for ax_i, cfg in enumerate(panel_configs):
        ax = axes[ax_i]
        ax.hist(eigs, bins=bins, density=True, color='#7BAFD4', alpha=0.65,
                edgecolor='white', linewidth=0.3, zorder=2)

        if cfg['overlay'] == 'naive':
            ax.plot(lam_grid_naive, rho_grid_naive, color=cfg['color'], lw=2.2, zorder=4)
            ax.axvline(lp_naive, color=cfg['color'], ls='--', lw=1.2, zorder=3)
            ax.axvline(lm_naive, color=cfg['color'], ls='--', lw=1.0, zorder=3, alpha=0.6)
            if w0 > 0:
                bar_width = bins[1] - bins[0]
                ax.bar(0, w0 / bar_width, width=bar_width, color=cfg['color'],
                       alpha=0.35, edgecolor=cfg['color'], linewidth=1.2, zorder=6)

        elif cfg['overlay'] == 'bulk':
            ax.plot(lam_grid_bulk, rho_grid_bulk, color=cfg['color'], lw=2.2, zorder=4)
            ax.axvline(lp, color=cfg['color'], ls='--', lw=1.2, zorder=3)
            ax.axvline(lm, color=cfg['color'], ls='--', lw=1.0, zorder=3, alpha=0.6)
            if w0 > 0:
                bar_width = bins[1] - bins[0]
                ax.bar(0, w0 / bar_width, width=bar_width, color=cfg['color'],
                       alpha=0.35, edgecolor=cfg['color'], linewidth=1.2, zorder=6)

        elif cfg['overlay'] == 'mc':
            ax.axvline(mc_edge, color=cfg['color'], ls='-.', lw=2.0, zorder=5)
            ax.axvline(mc['mc_edge_mean'], color=cfg['color'], ls=':', lw=1.2,
                       zorder=4, alpha=0.6)

        ax.axvline(lam1, color='#C0392B', ls='-', lw=1.0, zorder=5, alpha=0.7)
        ax.set_yscale('symlog', linthresh=100)
        ax.set_ylim(bottom=0)
        ax.set_title(cfg['title'], fontsize=9.5)
        ax.set_xlabel(r'$\lambda$', fontsize=9)
        if ax_i == 0:
            ax.set_ylabel('Density (symlog)', fontsize=9)
        ax.tick_params(labelsize=7)

    # Legends below panels
    leg_a = [
        Patch(facecolor='#7BAFD4', alpha=0.65, edgecolor='white', label='Eigenvalues'),
        Line2D([0], [0], color='#E67E22', lw=2.2, label=r'$\rho_\tau^\gamma$ density'),
        Line2D([0], [0], color='#E67E22', ls='--', lw=1.2,
               label=fr'$\lambda_\pm$  ($\sigma^2_f = {sigma_f_sq:.2e}$)'),
        Patch(facecolor='#E67E22', alpha=0.35, edgecolor='#E67E22',
              label=fr'$\delta_0$ atom (mass $= {w0:.3f}$)'),
        Line2D([0], [0], color='#C0392B', ls='-', lw=1.0, alpha=0.7,
               label=fr'$\lambda_1 = {lam1:.2e}$ (DA1)'),
    ]
    axes[0].legend(handles=leg_a, fontsize=6.5, loc='upper center',
                   bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True,
                   fancybox=True, edgecolor='#cccccc')

    leg_b = [
        Patch(facecolor='#7BAFD4', alpha=0.65, edgecolor='white', label='Eigenvalues'),
        Line2D([0], [0], color='#C0392B', lw=2.2, label=r'$\rho_\tau^\gamma$ density'),
        Line2D([0], [0], color='#C0392B', ls='--', lw=1.2,
               label=fr'$\lambda_\pm$  ($\sigma^2_{{\mathrm{{bulk}}}} = {fit["sigma_bulk_sq"]:.2e}$)'),
        Patch(facecolor='#C0392B', alpha=0.35, edgecolor='#C0392B',
              label=fr'$\delta_0$ atom (mass $= {w0:.3f}$)'),
        Line2D([0], [0], color='#C0392B', ls='-', lw=1.0, alpha=0.7,
               label=fr'$\lambda_1 = {lam1:.2e}$ (DA1)'),
    ]
    axes[1].legend(handles=leg_b, fontsize=6.5, loc='upper center',
                   bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True,
                   fancybox=True, edgecolor='#cccccc')

    leg_c = [
        Patch(facecolor='#7BAFD4', alpha=0.65, edgecolor='white', label='Eigenvalues'),
        Line2D([0], [0], color='#27AE60', ls='-.', lw=2.0,
               label=f'MC 95th pctl = {mc_edge:.2e}'),
        Line2D([0], [0], color='#27AE60', ls=':', lw=1.2, alpha=0.6,
               label=f'MC mean = {mc["mc_edge_mean"]:.2e}'),
        Line2D([0], [0], color='#C0392B', ls='-', lw=1.0, alpha=0.7,
               label=fr'$\lambda_1 = {lam1:.2e}$ (DA1)'),
    ]
    axes[2].legend(handles=leg_c, fontsize=6.5, loc='upper center',
                   bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True,
                   fancybox=True, edgecolor='#cccccc')

    fig.suptitle(fr'SDXL elderly ($w=7.5$), CLIP ($d={d}$, $N={N}$, '
                 fr'$\gamma={gamma:.1f}$, $\tau={tau_star:.1f}$s, dt={dt})',
                 fontsize=11, y=0.96)
    fig_path = os.path.join(RESDIR, 'fig_sdxl_elderly_bulk_dt0p0025.png')
    fig.savefig(fig_path, dpi=200)
    print(f'\nSaved: {fig_path}')

    # === Figure 2: symlog x ===
    linthresh_x = 5e-4
    def symlog(x, lt=linthresh_x):
        return np.sign(x) * np.log10(1 + np.abs(x) / lt)
    def inv_symlog(y, lt=linthresh_x):
        return np.sign(y) * lt * (10**np.abs(y) - 1)

    sl_lo = symlog(bin_lo)
    sl_hi = symlog(bin_hi)
    n_bins_sl = 100
    bins_sl = inv_symlog(np.linspace(sl_lo, sl_hi, n_bins_sl + 1))

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5.2))
    plt.subplots_adjust(wspace=0.25, left=0.05, right=0.97, top=0.82, bottom=0.22)

    for ax_i, cfg in enumerate(panel_configs):
        ax = axes2[ax_i]
        ax.hist(eigs, bins=bins_sl, density=True, color='#7BAFD4', alpha=0.65,
                edgecolor='white', linewidth=0.3, zorder=2)

        if cfg['overlay'] == 'naive':
            ax.plot(lam_grid_naive, rho_grid_naive, color=cfg['color'], lw=2.2, zorder=4)
            ax.axvline(lp_naive, color=cfg['color'], ls='--', lw=1.2, zorder=3)
            ax.axvline(lm_naive, color=cfg['color'], ls='--', lw=1.0, zorder=3, alpha=0.6)
            if w0 > 0:
                bw_sl = bins_sl[len(bins_sl)//2] - bins_sl[len(bins_sl)//2 - 1]
                ax.bar(0, w0 / bw_sl, width=bw_sl, color=cfg['color'],
                       alpha=0.35, edgecolor=cfg['color'], linewidth=1.2, zorder=6)
        elif cfg['overlay'] == 'bulk':
            ax.plot(lam_grid_bulk, rho_grid_bulk, color=cfg['color'], lw=2.2, zorder=4)
            ax.axvline(lp, color=cfg['color'], ls='--', lw=1.2, zorder=3)
            ax.axvline(lm, color=cfg['color'], ls='--', lw=1.0, zorder=3, alpha=0.6)
            if w0 > 0:
                bw_sl = bins_sl[len(bins_sl)//2] - bins_sl[len(bins_sl)//2 - 1]
                ax.bar(0, w0 / bw_sl, width=bw_sl, color=cfg['color'],
                       alpha=0.35, edgecolor=cfg['color'], linewidth=1.2, zorder=6)
        elif cfg['overlay'] == 'mc':
            ax.axvline(mc_edge, color=cfg['color'], ls='-.', lw=2.0, zorder=5)
            ax.axvline(mc['mc_edge_mean'], color=cfg['color'], ls=':', lw=1.2,
                       zorder=4, alpha=0.6)

        ax.axvline(lam1, color='#C0392B', ls='-', lw=1.0, zorder=5, alpha=0.7)
        ax.set_xscale('symlog', linthresh=linthresh_x)
        ax.set_yscale('symlog', linthresh=100)
        ax.set_ylim(bottom=0)
        ax.set_title(cfg['title'], fontsize=9.5)
        ax.set_xlabel(r'$\lambda$ (symlog)', fontsize=9)
        if ax_i == 0:
            ax.set_ylabel('Density (symlog)', fontsize=9)
        ax.tick_params(labelsize=7)

    axes2[0].legend(handles=leg_a, fontsize=6.5, loc='upper center',
                    bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True,
                    fancybox=True, edgecolor='#cccccc')
    axes2[1].legend(handles=leg_b, fontsize=6.5, loc='upper center',
                    bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True,
                    fancybox=True, edgecolor='#cccccc')
    axes2[2].legend(handles=leg_c, fontsize=6.5, loc='upper center',
                    bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True,
                    fancybox=True, edgecolor='#cccccc')

    fig2.suptitle(fr'SDXL elderly ($w=7.5$), CLIP ($d={d}$, $N={N}$, '
                  fr'$\gamma={gamma:.1f}$, $\tau={tau_star:.1f}$s, dt={dt})  [symlog x]',
                  fontsize=11, y=0.96)
    fig2_path = os.path.join(RESDIR, 'fig_sdxl_elderly_bulk_dt0p0025_symlogx.png')
    fig2.savefig(fig2_path, dpi=200)
    print(f'Saved: {fig2_path}')

    # JSON sidecar
    json_data = {
        'tau_s': float(tau_star),
        'dt': dt,
        'T': T,
        'gamma': gamma,
        'd': d,
        'N': N,
        'sigma_naive_sq': float(fit['sigma_naive_sq']),
        'atom_mass_w0': w0,
        'empirical_near_zero_count': n_near_zero,
        'empirical_near_zero_fraction': frac_near_zero,
        'peel_1': {
            'n_spikes': fit['n_spikes'],
            'spike_values': fit['spike_vals'].tolist(),
            'sigma_bulk_sq': fit['sigma_bulk_sq'],
            'lm': fit['lm_bulk'],
            'lp': fit['lp_bulk'],
            'n_below_lm': fit['n_below_lm'],
        },
        'mc_null': {k: v for k, v in mc.items() if k != 'mc_edge_all'},
        'lambda_1': float(eigs[0]),
        'lambda_2': float(eigs[1]),
        'gap12': float(eigs[0] / eigs[1]) if eigs[1] > 0 else None,
    }
    json_path = os.path.join(RESDIR, 'fig_sdxl_elderly_bulk_dt0p0025.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f'Saved: {json_path}')

    # Summary
    print(f'\n  Summary:')
    print(f'    tau={tau_star:.1f}s, dt={dt}, T={T}')
    print(f'    lambda_1={eigs[0]:.4e}, lambda_2={eigs[1]:.4e}, gap={eigs[0]/eigs[1]:.2f}x')
    print(f'    Analytic edge (bulk): lambda_+={fit["lp_bulk"]:.4e}')
    print(f'    MC null edge (95%): {mc["mc_edge_95"]:.4e}')
    print(f'    lambda_1 vs MC edge: {"ABOVE" if eigs[0] > mc_edge else "BELOW"}')
    print(f'    delta_0 atom mass: {w0:.4f}, empirical near-zero: {n_near_zero}/{d}')


if __name__ == '__main__':
    main()
