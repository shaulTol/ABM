#!/usr/bin/env python3
"""
Shared pipeline: simulate Langevin diffusion → save C(τ) and trajectories.

Given a score function ∇log f and sampling config, runs the canonical
isotropic Langevin diffusion:
    dX = (σ²_f/2) ∇log f(X) dt + σ_f dW
computes the symmetrized autocovariance C_sym(τ) at each lag, and saves
eigenvalues, physical times, and metadata to an NPZ file.

Usage (as library):
    from cli.save_lags import save_lags
    save_lags(score_fn, X0, sigma_f_sq, dt, T, lags, out_path, ...)

Usage (as CLI):
    python -m cli.save_lags --help
"""
import os
import numpy as np
import torch


def compute_symmetrized_autocovariance(Z, lags):
    """Compute C_sym(τ) = (C(τ) + C(τ)^T) / 2 for each lag.

    Args:
        Z: (N, T+1, d) centered trajectories.
        lags: list of integer lag values.

    Returns:
        C_tau_all: (len(lags), d, d) array of symmetrized autocovariance matrices.
    """
    N, T_plus_1, d = Z.shape
    C_tau_all = np.zeros((len(lags), d, d))
    for i, lag in enumerate(lags):
        if lag >= T_plus_1:
            continue
        Z_t = Z[:, :T_plus_1 - lag, :]        # (N, T+1-lag, d)
        Z_th = Z[:, lag:, :]                   # (N, T+1-lag, d)
        # Flatten over N and time
        Z_t_flat = Z_t.reshape(-1, d)          # (M, d)
        Z_th_flat = Z_th.reshape(-1, d)        # (M, d)
        C = Z_t_flat.T @ Z_th_flat / Z_t_flat.shape[0]
        C_tau_all[i] = (C + C.T) / 2
    return C_tau_all


def save_lags(score_fn, X0, sigma_f_sq, dt, T_steps, lags, out_path,
              device='cpu', dtype=torch.float64, seed=None):
    """Simulate Langevin and save autocovariance data.

    Args:
        score_fn: callable(X) -> ∇log f(X), where X is (N, d) tensor.
        X0: (N, d) initial positions (numpy or torch).
        sigma_f_sq: scalar diffusion coefficient tr(Σ_f)/d.
        dt: time step.
        T_steps: number of Langevin steps.
        lags: list/array of integer lag indices.
        out_path: path for output NPZ file.
        device: torch device.
        dtype: torch dtype.
        seed: optional RNG seed.

    Saves NPZ with keys:
        C_tau_all: (n_lags, d, d) symmetrized autocovariance matrices.
        eigenvalues: (n_lags, d) eigenvalues of C_sym(τ), sorted descending.
        phys_times: (n_lags,) physical times τ = lag * dt.
        sigma_f_sq: scalar diffusion coefficient.
        gamma: aspect ratio d/N.
        d: dimensionality.
        N: number of trajectories.
        dt: time step.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    X0_t = torch.as_tensor(X0, dtype=dtype, device=device)
    N, d = X0_t.shape
    sigma_f = float(np.sqrt(sigma_f_sq))
    noise_scale = sigma_f * np.sqrt(dt)

    # Simulate Langevin
    X = X0_t.clone()
    trajectory = torch.zeros(N, T_steps + 1, d, dtype=dtype, device=device)
    trajectory[:, 0] = X

    for t in range(T_steps):
        grad = score_fn(X)
        X = X + (sigma_f_sq / 2) * grad * dt + noise_scale * torch.randn_like(X)
        trajectory[:, t + 1] = X

    # Center trajectories
    Z = trajectory.cpu().numpy()
    mu = Z.mean(axis=(0, 1))  # global mean across N and T
    Z = Z - mu[None, None, :]

    # Compute autocovariance at each lag
    lags = np.asarray(lags)
    C_tau_all = compute_symmetrized_autocovariance(Z, lags)

    # Eigendecompose each C_sym(τ)
    eigenvalues = np.zeros((len(lags), d))
    for i in range(len(lags)):
        eigs = np.linalg.eigvalsh(C_tau_all[i])
        eigenvalues[i] = eigs[::-1]  # descending

    phys_times = lags.astype(float) * dt
    gamma = d / N

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez(out_path,
             C_tau_all=C_tau_all,
             eigenvalues=eigenvalues,
             phys_times=phys_times,
             sigma_f_sq=sigma_f_sq,
             gamma=gamma,
             d=d,
             N=N,
             dt=dt)
    print(f'Saved: {out_path}  ({len(lags)} lags, d={d}, N={N})')


if __name__ == '__main__':
    print('cli/save_lags.py — use as library: from cli.save_lags import save_lags')
