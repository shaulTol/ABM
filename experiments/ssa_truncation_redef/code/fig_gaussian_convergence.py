#!/usr/bin/env python3
"""
Plot Gaussian Ŝ convergence to 1.

Two-panel figure:
  Left:  ρ²(τ) — empirical vs analytic exp(-τ), with noise floor 1/N marked
  Right: cumulative Ŝ(τ) = ∫₀^τ ρ²(t)dt — converging to 1, with τ* marked

Also shows the free-convolution noise floor upper edge if available.
"""
import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CODEDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.join(CODEDIR, '..')
REPODIR = os.path.join(BASEDIR, '..', '..')
sys.path.insert(0, REPODIR)

RESULTS = os.path.join(BASEDIR, 'results', 'gaussian_profile.json')
OUTDIR = os.path.join(BASEDIR, 'results')


def main():
    with open(RESULTS) as f:
        data = json.load(f)

    lags = np.array(data['lags'])
    rho_sq_emp = np.array(data['rho_sq_empirical'])
    rho_sq_ana = np.array(data['rho_sq_analytic'])
    cumsum_emp = np.array(data['cumsum_empirical'])
    cumsum_ana = np.array(data['cumsum_analytic'])
    tau_star = data['tau_star']
    N = data['N']
    d = data['d']
    dt = data['dt']

    # Free-conv noise floor (upper edge of eigenvalue support for null)
    try:
        from experiments.noise_floor.free_conv_utils import edges
        sigma = 1.0  # var_f = 1
        gamma = d / N
        fc_upper = []
        fc_lags = lags[1:]  # skip lag 0
        for tau in fc_lags:
            _, hi = edges(float(tau), sigma=sigma, d=d, N=N)
            # Upper edge is for leading eigenvalue of C(τ).
            # For scalar ρ² = (tr C(τ)/tr C(0))²,
            # the noise floor in ρ² ≈ d * hi² / (tr C(0))²
            # But simpler: ρ(τ) noise ≈ sqrt(d * hi / tr_c0) ...
            # Actually for trace: ρ = tr(C(τ))/tr(C(0)),
            # and tr(C(τ)) for null ≈ d * mean_eigenvalue ≈ d * σ²*exp(-τ) for signal
            # For pure noise (no signal at this lag), tr(C(τ)) ≈ 0 + noise
            # The 1/N floor is simpler and well-justified for scalar ρ²
            fc_upper.append(hi)
        fc_upper = np.array(fc_upper)
        # Convert eigenvalue upper edge to ρ² scale:
        # Each eigenvalue contributes to tr(C(τ)). For null, largest eigenvalue = hi.
        # ρ_null ≈ sqrt(Σ λ_i² / (d * tr_C0²)) ... this gets complicated.
        # Just plot 1/N as the scalar noise floor, and fc_upper as reference.
        has_freeconv = True
    except ImportError:
        has_freeconv = False

    lags_plot = lags[1:]  # skip lag 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left panel: ρ²(τ) ──
    ax1.semilogy(lags_plot, rho_sq_emp[1:], '-', color='tab:blue', linewidth=1.5,
                 alpha=0.7, label='Empirical')
    ax1.semilogy(lags_plot, rho_sq_ana[1:], '--', color='black', linewidth=1.5,
                 label=r'Analytic $e^{-\tau}$')
    ax1.axhline(1.0 / N, color='tab:red', linestyle=':', linewidth=1.5,
                label=f'Noise floor $1/N = {1/N:.0e}$')
    ax1.axvline(tau_star, color='tab:orange', linestyle='--', linewidth=1.5,
                alpha=0.7, label=f'$\\tau^* = {tau_star:.1f}$')

    ax1.set_xlabel(r'Lag $\tau$', fontsize=13)
    ax1.set_ylabel(r'$\rho^2(\tau)$', fontsize=13)
    ax1.set_title(r'Autocorrelation decay', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.set_ylim(bottom=1e-6)

    # ── Right panel: cumulative Ŝ(τ) ──
    ax2.plot(lags_plot, cumsum_emp, '-', color='tab:blue', linewidth=2,
             label='Empirical')
    ax2.plot(lags_plot, cumsum_ana, '--', color='black', linewidth=1.5,
             label=r'Analytic $1 - e^{-\tau}$')
    ax2.axhline(1.0, color='tab:green', linestyle=':', linewidth=1.5,
                label=r'$\int_0^\infty e^{-\tau} d\tau = 1$')
    ax2.axvline(tau_star, color='tab:orange', linestyle='--', linewidth=1.5,
                alpha=0.7, label=f'$\\tau^* = {tau_star:.1f}$')

    # Mark Ŝ value at τ*
    tau_star_idx = np.searchsorted(lags_plot, tau_star)
    if tau_star_idx < len(cumsum_emp):
        s_at_tau = cumsum_emp[tau_star_idx]
        ax2.plot(tau_star, s_at_tau, 'o', color='tab:orange', markersize=8, zorder=5)
        ax2.annotate(f'$\\hat{{S}} = {s_at_tau:.3f}$',
                     xy=(tau_star, s_at_tau), xytext=(tau_star + 1, s_at_tau - 0.1),
                     fontsize=12, color='tab:orange',
                     arrowprops=dict(arrowstyle='->', color='tab:orange'))

    ax2.set_xlabel(r'Lag $\tau$', fontsize=13)
    ax2.set_ylabel(r'$\hat{S}(\tau) = \Delta t \sum_{\ell=1}^{\tau/\Delta t} \rho(\ell)^2$',
                   fontsize=13)
    ax2.set_title(r'Cumulative $\hat{S}$ converges to 1', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.set_ylim(-0.05, 1.15)

    plt.tight_layout(pad=1.0)
    path = os.path.join(OUTDIR, 'fig_gaussian_convergence.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == '__main__':
    main()
