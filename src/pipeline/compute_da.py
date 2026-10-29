"""
Pipeline step 2b: given saved C(τ), compute DA directions + significance.

Reads an NPZ file produced by save_lags and computes the DA
(Dominant Autocorrelation) directions: eigenvectors of the symmetrized
autocovariance C_sym(τ*) at the selected lag.

Lag selection (paper, Section "DA"): τ*(m) is the largest lag at which
  (i)   each of the top m eigenvalues of C_sym(τ) exceeds the null threshold
        (here the analytic edge λ₊(τ) of Theorem 2; da_significance gives the
        matched Monte Carlo null of Algorithm 2, Bonferroni over the m directions),
  (ii)  the 1D projection onto each of v_1..v_m rejects unimodality
        (Hartigan's dip test, Bonferroni over the m directions; needs Z_all), and
  (iii) ρ̂(τ) = tr Ĉ(τ)/tr Ĉ(0) < 1/e.
The number of metastable directions m̂ is the largest m for which τ*(m) exists.

Usage:
    from src.pipeline import compute_da, da_significance, select_tau_star

Command line: python -m cli.compute_da <C_tau.npz> [--significance B]
"""
import numpy as np

from ..noise_floor.free_conv_utils import edges
from .save_lags import compute_symmetrized_autocovariance


def dip_pvalues(Z_all, C_sym, m, max_points=20000, seed=0):
    """Hartigan dip-test p-values of the pooled samples projected on v_1..v_m."""
    import diptest
    X = Z_all.reshape(-1, Z_all.shape[-1])
    if len(X) > max_points:
        X = X[np.random.RandomState(seed).choice(len(X), max_points, replace=False)]
    evals, evecs = np.linalg.eigh(C_sym)
    V = evecs[:, np.argsort(evals)[::-1][:m]]
    return np.array([diptest.diptest(X @ V[:, j])[1] for j in range(m)])


def select_tau_star(eigenvalues, phys_times, sigma_f_sq, gamma, m=1,
                    C_tau_all=None, Z_all=None, alpha=0.05):
    """Largest lag satisfying the paper's three conditions for m directions.

    Args:
        eigenvalues: (n_lags, d) eigenvalues sorted descending (lag 0 first).
        phys_times: (n_lags,) physical times.
        sigma_f_sq: scalar diffusion coefficient.
        gamma: aspect ratio d/N.
        m: number of DA directions that must all pass.
        C_tau_all, Z_all: if both are given, condition (ii) (dip test at
            Bonferroni level alpha/m) is applied; otherwise only (i) and (iii).
        alpha: family-wise level for condition (ii).

    Returns:
        tau_star_idx, tau_star; (None, None) if no lag satisfies the conditions.
    """
    sigma = np.sqrt(sigma_f_sq)
    traces = eigenvalues.sum(axis=1)
    rho = traces / traces[0]
    for i in range(len(phys_times) - 1, 0, -1):
        _, lp = edges(phys_times[i], sigma, gamma=gamma)
        if not (np.all(eigenvalues[i, :m] > lp) and rho[i] < 1.0 / np.e):
            continue
        if C_tau_all is not None and Z_all is not None:
            if np.any(dip_pvalues(Z_all, C_tau_all[i], m) > alpha / m):
                continue
        return i, float(phys_times[i])
    return None, None


def n_metastable(C_tau_path, m_max=10, alpha=0.05):
    """Largest m for which τ*(m) exists, with that τ*(m)."""
    data = np.load(C_tau_path)
    Z_all = data['Z_all'] if 'Z_all' in data else None
    best = (0, None, None)
    for m in range(1, m_max + 1):
        idx, tau = select_tau_star(data['eigenvalues'], data['phys_times'],
                                   float(data['sigma_f_sq']), float(data['gamma']),
                                   m=m, C_tau_all=data['C_tau_all'], Z_all=Z_all,
                                   alpha=alpha)
        if idx is None:
            break
        best = (m, idx, tau)
    return {'m_hat': best[0], 'tau_star_idx': best[1], 'tau_star': best[2]}


def compute_da(C_tau_path, tau_idx=None, m=1, n_directions=6):
    """Compute DA directions from saved autocovariance.

    Args:
        C_tau_path: path to NPZ from save_lags.
        tau_idx: lag index to use. If None, τ*(m) from select_tau_star.
        m: number of directions the auto-selected lag must support.
        n_directions: number of eigenvectors to return.

    Returns:
        dict with keys: eigenvalues, eigenvectors, tau_star, tau_star_idx,
                         gap_ratio, sigma_f_sq, gamma, d.
    """
    data = np.load(C_tau_path)
    C_tau_all = data['C_tau_all']       # (n_lags, d, d)
    eigenvalues = data['eigenvalues']   # (n_lags, d)
    phys_times = data['phys_times']
    sigma_f_sq = float(data['sigma_f_sq'])
    gamma = float(data['gamma'])
    d = int(data['d'])

    # Select lag
    if tau_idx is None:
        tau_idx, tau_star = select_tau_star(
            eigenvalues, phys_times, sigma_f_sq, gamma, m=m, C_tau_all=C_tau_all,
            Z_all=data['Z_all'] if 'Z_all' in data else None)
        if tau_idx is None:
            raise ValueError(f"no lag satisfies the selection criterion for m={m}")
    else:
        tau_star = float(phys_times[tau_idx])

    # Eigendecompose C_sym(τ*)
    C_star = C_tau_all[tau_idx]
    evals, evecs = np.linalg.eigh(C_star)
    # Sort descending
    order = np.argsort(evals)[::-1].copy()
    evals = evals[order]
    evecs = evecs[:, order]

    # Truncate to n_directions
    evals = evals[:n_directions]
    evecs = evecs[:, :n_directions]

    # Gap ratio
    sigma = np.sqrt(sigma_f_sq)
    _, lp = edges(tau_star, sigma, gamma=gamma)
    gap_ratio = evals[0] / lp if lp > 0 else float('inf')

    return {
        'eigenvalues': evals,
        'eigenvectors': evecs,
        'tau_star': tau_star,
        'tau_star_idx': int(tau_idx),
        'gap_ratio': float(gap_ratio),
        'noise_floor': float(lp),
        'sigma_f_sq': sigma_f_sq,
        'gamma': gamma,
        'd': d,
    }


def da_significance(C_tau_path, B=100, tau_idx=None, m=1, alpha=0.05, seed=42):
    """Matched-covariance Gaussian Monte Carlo null (Algorithm 2).

    Each replicate draws N samples from N(0, Σ̂), with Σ̂ = Ĉ(0) the sample
    covariance, runs the canonical diffusion with the exact linear score
    -Σ̂⁻¹x (same N, Δt, T and σ_f² = tr Σ̂/d), and eigendecomposes C_sym(τ*).
    Observed λ_i is compared with the null distribution of the i-th eigenvalue;
    direction i passes if p_i ≤ alpha/m (Bonferroni over the top m).

    Returns:
        dict with keys: observed_evals, null_evals (B, n), p_values, passed, tau_star.
    """
    data = np.load(C_tau_path)
    result = compute_da(C_tau_path, tau_idx=tau_idx, m=m)
    observed = result['eigenvalues']
    n = len(observed)
    dt = float(data['dt'])
    N, d, T = int(data['N']), int(data['d']), int(data['T_steps'])
    lag = int(np.rint(result['tau_star'] / dt))

    Sigma = data['C_tau_all'][0]
    sigma_f_sq = np.trace(Sigma) / d
    L = np.linalg.cholesky(Sigma + 1e-12 * np.trace(Sigma) / d * np.eye(d))
    P = np.linalg.inv(Sigma)
    rng = np.random.RandomState(seed)
    null = np.zeros((B, n))
    for b in range(B):
        Z = np.empty((N, T + 1, d))
        Z[:, 0] = rng.randn(N, d) @ L.T
        for t in range(T):
            X = Z[:, t]
            Z[:, t + 1] = (X - 0.5 * sigma_f_sq * dt * X @ P
                           + np.sqrt(sigma_f_sq * dt) * rng.randn(N, d))
        Z -= Z.mean(axis=(0, 1))
        C = compute_symmetrized_autocovariance(Z, [lag])[0]
        null[b] = np.sort(np.linalg.eigvalsh(C))[::-1][:n]

    p_values = (1 + (null >= observed).sum(axis=0)) / (B + 1)
    return {
        'observed_evals': observed,
        'null_evals': null,
        'p_values': p_values,
        'passed': p_values[:m] <= alpha / m,
        'tau_star': result['tau_star'],
    }

