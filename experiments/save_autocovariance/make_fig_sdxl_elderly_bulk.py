#!/usr/bin/env python3
"""
Bulk-fit validation figure for SDXL elderly person (w=7.5) in CLIP space.

Shows histogram of all 768 eigenvalues of Ĉ_sym(τ) at the last available lag
with overlays:
  (i)   free-convolution density ρ_{τ}^γ (σ²_bulk fitted via one-iteration peel)
  (ii)  analytic edge λ₊(τ) from free-conv
  (iii) Monte-Carlo null edge (max eigenvalue of matched-covariance null)
  (iv)  spike marker at λ₁ (DA1 → gender)

Note: With γ=3.84 (d=768 ≫ N=200), many eigenvalues exceed the analytic edge.
The DA1 gap never converged to m=1 within T=2000 steps. The analytic density
is approximate because CLIP embeddings violate the iid assumption of the
free-convolution theorem. The MC null edge is the rigorous reference.

Requires:
  results/elderly_gs7p5_n200_C_tau_full.npz  (from save_sdxl_C_tau.py)

Output:
  results/fig_sdxl_elderly_bulk.png
  results/fig_sdxl_elderly_bulk.json
"""
import sys, os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'noise_floor'))
from free_conv_utils import edges as fc_edges, density as fc_density

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RESDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
N_MC_REPS = 200  # Monte-Carlo null repetitions


def peel_fit(eigs_tau, eigs_C0, tau, gamma, sigma_naive):
    """One-iteration peel: identify spikes with naive σ², refit on bulk."""
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
    """Monte-Carlo null: max eigenvalue of Ĉ_sym(τ) under iid draws from N(0, C0).

    Simulates the OU process X_τ = e^{-τ/2} X_0 + √(1-e^{-τ}) Z
    where X_0, Z ~ N(0, C0) iid, then computes the symmetrized
    cross-covariance and takes its top eigenvalue.
    """
    d = C0.shape[0]
    rng = np.random.RandomState(seed)

    # Eigendecomposition for sampling (numerically stable)
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
    npz_path = os.path.join(RESDIR, 'elderly_gs7p5_n200_C_tau_full.npz')
    if not os.path.exists(npz_path):
        print(f"ERROR: {npz_path} not found. Run save_sdxl_C_tau.py first.")
        sys.exit(1)

    data = np.load(npz_path)
    C0 = data['C0']
    lags = data['lags']
    phys_times = data['phys_times']
    eigenvalues = data['eigenvalues']
    sigma_f_sq = float(data['sigma_f_sq'])
    d = int(data['d'])
    N = int(data['N'])
    dt = float(data['dt'])

    gamma = d / N
    sigma_naive = np.sqrt(sigma_f_sq)
    eigs_C0 = np.linalg.eigvalsh(C0)[::-1].copy()

    print(f"Loaded: d={d}, N={N}, gamma={gamma:.4f}, sigma_f^2={sigma_f_sq:.6f}")
    print(f"  {len(lags)} lags, phys_times: [{phys_times[0]:.4f}, {phys_times[-1]:.4f}]s")

    # Use last lag (τ = T·dt = 1.0s). Note: DA1 gap never converged to m=1.
    idx_star = len(lags) - 1
    tau_star = phys_times[idx_star]
    print(f"  Using last lag: tau={tau_star:.4f}s (idx={idx_star})")

    eigs = eigenvalues[idx_star].copy()
    fit = peel_fit(eigs, eigs_C0, tau_star, gamma, sigma_naive)

    print(f"  Peel: {fit['n_spikes']} above naive lp, sigma_bulk^2={fit['sigma_bulk_sq']:.2e}")
    print(f"  lambda_+={fit['lp_bulk']:.4e}, lambda_-={fit['lm_bulk']:.4e}")
    print(f"  {fit['n_below_lm']} below lambda_-")

    # MC null edge
    print(f"\n  Computing MC null ({N_MC_REPS} reps)...")
    mc = mc_null_edge(C0, tau_star, N, n_reps=N_MC_REPS)
    print(f"  MC edge: mean={mc['mc_edge_mean']:.4e}, 95%={mc['mc_edge_95']:.4e}")

    # === Rank-deficiency atom at λ=0 when γ > 1 ===
    w0 = max(0.0, 1.0 - 2.0 / gamma)  # atom mass
    # Empirical near-zero count: |λ| < tol relative to tr(C0)
    tr_C0 = np.trace(C0)
    tol = 1e-10 * tr_C0
    n_near_zero = int(np.sum(np.abs(eigs) < tol))
    frac_near_zero = n_near_zero / d
    print(f"\n  Rank-deficiency atom (gamma={gamma:.2f} > 1):")
    print(f"    Predicted atom mass: w0 = 1 - 2/gamma = {w0:.4f}")
    print(f"    Empirical |lambda| < {tol:.2e}: {n_near_zero}/{d} = {frac_near_zero:.4f}")
    print(f"    Expected: d - 2N = {d} - {2*N} = {d - 2*N}, fraction = {(d-2*N)/d:.4f}")

    # === Figure: 3 panels ===
    lm, lp = fit['lm_bulk'], fit['lp_bulk']
    support_width = lp - lm
    lm_naive, lp_naive = fit['lm_naive'], fit['lp_naive']
    sw_naive = lp_naive - lm_naive
    mc_edge = mc['mc_edge_95']
    lam1 = eigs[0]

    # Precompute density curves
    pad_naive = 0.3 * sw_naive
    lam_grid_naive = np.linspace(lm_naive - pad_naive, lp_naive + pad_naive, 800)
    rho_grid_naive = fc_density(lam_grid_naive, tau_star, sigma_naive, gamma=gamma)

    pad = 0.3 * support_width
    lam_grid_bulk = np.linspace(lm - pad, lp + pad, 800)
    rho_grid_bulk = fc_density(lam_grid_bulk, tau_star, fit['sigma_bulk'], gamma=gamma)

    # MC null: histogram of max eigenvalues
    mc_all = np.array(mc['mc_edge_all'])

    # Shared histogram bins
    all_eigs = eigs
    eig_min, eig_max = all_eigs.min(), all_eigs.max()
    bin_lo = eig_min - 0.02 * (eig_max - eig_min)
    bin_hi = eig_max + 0.02 * (eig_max - eig_min)
    n_bins = 60
    bins = np.linspace(bin_lo, bin_hi, n_bins)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    plt.subplots_adjust(wspace=0.25, left=0.05, right=0.97, top=0.82, bottom=0.22)

    panel_configs = [
        {
            'title': r'(a) Free-conv. ($\sigma^2_f$, naive)',
            'overlay': 'naive',
            'color': '#E67E22',
        },
        {
            'title': r'(b) Free-conv. ($\sigma^2_{\mathrm{bulk}}$, peel)',
            'overlay': 'bulk',
            'color': '#C0392B',
        },
        {
            'title': '(c) Monte-Carlo null',
            'overlay': 'mc',
            'color': '#27AE60',
        },
    ]

    for ax_i, cfg in enumerate(panel_configs):
        ax = axes[ax_i]

        # Histogram
        ax.hist(all_eigs, bins=bins, density=True, color='#7BAFD4', alpha=0.65,
                edgecolor='white', linewidth=0.3, zorder=2)

        if cfg['overlay'] == 'naive':
            # Naive free-conv density (continuous part) + edges
            ax.plot(lam_grid_naive, rho_grid_naive, color=cfg['color'], lw=2.2, zorder=4)
            ax.axvline(lp_naive, color=cfg['color'], ls='--', lw=1.2, zorder=3)
            ax.axvline(lm_naive, color=cfg['color'], ls='--', lw=1.0, zorder=3, alpha=0.6)
            # δ₀ atom bar
            if w0 > 0:
                bar_width = bins[1] - bins[0]
                bar_height = w0 / bar_width
                ax.bar(0, bar_height, width=bar_width, color=cfg['color'],
                       alpha=0.35, edgecolor=cfg['color'], linewidth=1.2, zorder=6)

        elif cfg['overlay'] == 'bulk':
            # Bulk-fitted free-conv density (continuous part) + edges
            ax.plot(lam_grid_bulk, rho_grid_bulk, color=cfg['color'], lw=2.2, zorder=4)
            ax.axvline(lp, color=cfg['color'], ls='--', lw=1.2, zorder=3)
            ax.axvline(lm, color=cfg['color'], ls='--', lw=1.0, zorder=3, alpha=0.6)
            # δ₀ atom bar
            if w0 > 0:
                bar_width = bins[1] - bins[0]
                bar_height = w0 / bar_width
                ax.bar(0, bar_height, width=bar_width, color=cfg['color'],
                       alpha=0.35, edgecolor=cfg['color'], linewidth=1.2, zorder=6)

        elif cfg['overlay'] == 'mc':
            # MC null edge
            ax.axvline(mc_edge, color=cfg['color'], ls='-.', lw=2.0, zorder=5)
            ax.axvline(mc['mc_edge_mean'], color=cfg['color'], ls=':', lw=1.2,
                       zorder=4, alpha=0.6)

        # DA1 spike on all panels
        ax.axvline(lam1, color='#C0392B', ls='-', lw=1.0, zorder=5, alpha=0.7)

        # Symlog y-axis
        ax.set_yscale('symlog', linthresh=100)
        ax.set_ylim(bottom=0)

        ax.set_title(cfg['title'], fontsize=9.5)
        ax.set_xlabel(r'$\lambda$', fontsize=9)
        if ax_i == 0:
            ax.set_ylabel('Density (symlog)', fontsize=9)
        ax.tick_params(labelsize=7)

    # --- All legends / annotations OUTSIDE the plot areas ---

    # Build legend handles for each panel, placed below the axes
    # Panel (a): naive
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

    # Panel (b): bulk peel
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

    # Panel (c): MC null
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

    # Shared supertitle
    fig.suptitle(fr'SDXL elderly ($w=7.5$), CLIP space ($d={d}$, $N={N}$, '
                 fr'$\gamma={gamma:.1f}$, $\tau={tau_star:.3f}$s)',
                 fontsize=11, y=0.96)
    fig_path = os.path.join(RESDIR, 'fig_sdxl_elderly_bulk.png')
    fig.savefig(fig_path, dpi=200)
    print(f'\nSaved: {fig_path}')

    # JSON sidecar
    json_data = {
        'tau_s': float(tau_star),
        'tau_star_idx': int(idx_star),
        'note': 'DA1 gap never converged to m=1 within T=2000. tau=T*dt is used.',
        'gamma': gamma,
        'd': d,
        'N': N,
        'dt': dt,
        'sigma_naive_sq': float(fit['sigma_naive_sq']),
        'atom_mass_w0': w0,
        'empirical_near_zero_count': n_near_zero,
        'empirical_near_zero_fraction': frac_near_zero,
        'expected_zero_count': max(0, d - 2*N),
        'peel_1': {
            'n_spikes': fit['n_spikes'],
            'spike_values': fit['spike_vals'].tolist(),
            'sigma_bulk_sq': fit['sigma_bulk_sq'],
            'lm': fit['lm_bulk'],
            'lp': fit['lp_bulk'],
            'n_below_lm': fit['n_below_lm'],
            'below_lm_values': fit['below_lm_vals'],
        },
        'mc_null': mc,
        'lambda_1': float(eigs[0]),
        'lambda_2': float(eigs[1]),
        'gap12': float(eigs[0] / eigs[1]) if eigs[1] > 0 else None,
        'n_bins': n_bins,
        'all_eigenvalues': eigs.tolist(),
    }
    json_path = os.path.join(RESDIR, 'fig_sdxl_elderly_bulk.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f'Saved: {json_path}')

    # === Second figure: symlog x-axis with symlog-spaced bins ===
    linthresh_x = 5e-4  # linear region around zero
    # Build bins uniform in symlog-transformed space
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

        # Histogram with symlog-spaced bins
        ax.hist(all_eigs, bins=bins_sl, density=True, color='#7BAFD4', alpha=0.65,
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

        # DA1 spike on all panels
        ax.axvline(lam1, color='#C0392B', ls='-', lw=1.0, zorder=5, alpha=0.7)

        # Symlog both axes
        ax.set_xscale('symlog', linthresh=linthresh_x)
        ax.set_yscale('symlog', linthresh=100)
        ax.set_ylim(bottom=0)

        ax.set_title(cfg['title'], fontsize=9.5)
        ax.set_xlabel(r'$\lambda$ (symlog)', fontsize=9)
        if ax_i == 0:
            ax.set_ylabel('Density (symlog)', fontsize=9)
        ax.tick_params(labelsize=7)

    # Legends below panels (same as linear version)
    axes2[0].legend(handles=leg_a, fontsize=6.5, loc='upper center',
                    bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True,
                    fancybox=True, edgecolor='#cccccc')
    axes2[1].legend(handles=leg_b, fontsize=6.5, loc='upper center',
                    bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True,
                    fancybox=True, edgecolor='#cccccc')
    axes2[2].legend(handles=leg_c, fontsize=6.5, loc='upper center',
                    bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True,
                    fancybox=True, edgecolor='#cccccc')

    fig2.suptitle(fr'SDXL elderly ($w=7.5$), CLIP space ($d={d}$, $N={N}$, '
                  fr'$\gamma={gamma:.1f}$, $\tau={tau_star:.3f}$s)  [symlog x]',
                  fontsize=11, y=0.96)
    fig2_path = os.path.join(RESDIR, 'fig_sdxl_elderly_bulk_symlogx.png')
    fig2.savefig(fig2_path, dpi=200)
    print(f'Saved: {fig2_path}')

    # Summary
    print(f'\n  Summary:')
    print(f'    tau={tau_star:.4f}s (last lag, gap never converged)')
    print(f'    lambda_1={eigs[0]:.4e}, lambda_2={eigs[1]:.4e}, gap={eigs[0]/eigs[1]:.2f}')
    print(f'    Analytic edge: lambda_+={fit["lp_bulk"]:.4e}')
    print(f'    MC null edge (95%): {mc["mc_edge_95"]:.4e}')
    print(f'    lambda_1 vs MC edge: {"ABOVE" if eigs[0] > mc_edge else "BELOW"}')
    print(f'    sigma_bulk^2={fit["sigma_bulk_sq"]:.2e}')
    print(f'    delta_0 atom mass (predicted): {w0:.4f}')
    print(f'    Near-zero eigenvalues (empirical): {n_near_zero}/{d} = {frac_near_zero:.4f}')
    print(f'    Panels (a) and (b) updated with delta_0 atom bar.')


if __name__ == '__main__':
    main()
