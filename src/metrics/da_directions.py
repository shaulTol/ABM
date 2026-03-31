"""
Dominant Autocorrelation (DA) Directions

Extracts the directions of maximal metastability from the lag-τ autocovariance
matrix C(τ). At τ=0, C(τ) is the data covariance (PCA directions). As τ grows,
fast intra-mode relaxations decay while slow inter-mode transitions persist,
so the leading eigenvector rotates from the max-variance direction toward the
max-metastability direction.

DA1 = leading eigenvector of C(τ) at large τ.

Two interfaces:
- GMM-based: pass (gmm_means, gmm_covs, gmm_weights) → uses LangevinSimulatorD
- Generic: pass a simulator object (GenericLangevinSimulator or any object with
  .step(), .sample_initial(), .mu_f attributes) → works with any score function
"""
import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Union
from ..simulation.langevin import LangevinSimulatorD
from .autocorrelation import compute_stationary_moments


def _make_simulator(
    simulator=None,
    gmm_means=None, gmm_covs=None, gmm_weights=None,
    dt=0.01, device=None,
):
    """
    Resolve simulator: use provided simulator object, or build one from GMM params.
    Returns (simulator, mu_f) tuple.
    """
    if simulator is not None:
        mu_f = simulator.mu_f
        if isinstance(mu_f, torch.Tensor):
            mu_f = mu_f.to(device=device or mu_f.device)
        return simulator, mu_f

    if gmm_means is None or gmm_covs is None or gmm_weights is None:
        raise ValueError(
            "Must provide either `simulator` or all of "
            "(gmm_means, gmm_covs, gmm_weights)"
        )
    sim = LangevinSimulatorD(gmm_means, gmm_covs, gmm_weights, dt=dt, device=device)
    mu_f, _ = compute_stationary_moments(gmm_means, gmm_covs, gmm_weights)
    mu_f = mu_f.to(device=device or gmm_means.device)
    return sim, mu_f


def _eigendecompose_C_tau(x_tau_centered, x0_centered, n):
    """Compute symmetrized C(τ), eigendecompose, return (evals_desc, evecs_desc, C_sym)."""
    C_tau = torch.matmul(x_tau_centered.t(), x0_centered) / n
    C_tau_sym = 0.5 * (C_tau + C_tau.t())
    evals, evecs = torch.linalg.eigh(C_tau_sym)  # ascending
    idx = torch.argsort(evals, descending=True)
    return evals[idx], evecs[:, idx], C_tau_sym


def extract_da_directions(
    gmm_means: Optional[torch.Tensor] = None,
    gmm_covs: Optional[torch.Tensor] = None,
    gmm_weights: Optional[torch.Tensor] = None,
    lag: int = 500,
    n_trajectories: int = 5000,
    T: Optional[int] = None,
    dt: float = 0.01,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    simulator=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract Dominant Autocorrelation directions from the lag-τ autocovariance matrix.

    Simulates the canonical diffusion forward τ steps from stationary samples X₀,
    computes C(τ) = (1/N) Σₙ (X_τ⁽ⁿ⁾ − μ_f)(X₀⁽ⁿ⁾ − μ_f)ᵀ, symmetrizes,
    and eigendecomposes to get DA directions sorted by decreasing eigenvalue.

    Two calling conventions:
        # GMM-based (original):
        extract_da_directions(gmm_means, gmm_covs, gmm_weights, lag=800)

        # Generic (score-function-based):
        extract_da_directions(simulator=my_simulator, lag=800)

    Args:
        gmm_means: (k, d) component means (GMM path)
        gmm_covs: (k, d, d) component covariance matrices (GMM path)
        gmm_weights: (k,) mixture weights (GMM path)
        lag: Number of diffusion steps τ
        n_trajectories: Number of samples N
        T: Total simulation length (defaults to lag)
        dt: Langevin time step
        device: Computation device
        seed: Random seed
        simulator: A simulator object with .step(x), .sample_initial(n), .mu_f.
                   If provided, GMM params are ignored.

    Returns:
        eigenvalues: (d,) eigenvalues sorted descending
        eigenvectors: (d, d) columns are DA directions, sorted by decreasing eigenvalue
        C_tau: (d, d) the symmetrized autocovariance matrix
    """
    device = device or torch.device('cpu')
    if T is None:
        T = lag
    assert T >= lag, f"T ({T}) must be >= lag ({lag})"

    if seed is not None:
        torch.manual_seed(seed)

    sim, mu_f = _make_simulator(
        simulator=simulator,
        gmm_means=gmm_means, gmm_covs=gmm_covs, gmm_weights=gmm_weights,
        dt=dt, device=device,
    )

    # Sample X₀ from stationary distribution
    x0 = sim.sample_initial(n_trajectories)  # (N, d)

    # Simulate forward τ steps (only need final state, not full trajectory)
    x = x0.clone()
    for t in range(lag):
        x = sim.step(x)
    x_tau = x  # (N, d)

    # Center
    x0_centered = x0 - mu_f.unsqueeze(0)
    x_tau_centered = x_tau - mu_f.unsqueeze(0)

    eigenvalues, eigenvectors, C_tau_sym = _eigendecompose_C_tau(
        x_tau_centered, x0_centered, n_trajectories
    )
    return eigenvalues, eigenvectors, C_tau_sym


def da_convergence_profile(
    gmm_means: Optional[torch.Tensor] = None,
    gmm_covs: Optional[torch.Tensor] = None,
    gmm_weights: Optional[torch.Tensor] = None,
    checkpoints: Optional[List[int]] = None,
    n_trajectories: int = 5000,
    dt: float = 0.01,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    reference_direction: Optional[torch.Tensor] = None,
    simulator=None,
) -> Dict[str, np.ndarray]:
    """
    Track how the leading eigenvector of C(τ) evolves with τ.

    Simulates out to max(checkpoints) steps, storing intermediate states at
    geometric checkpoints. At each checkpoint, computes C(τ) and its leading
    eigenvector. Returns angles, eigenvalues, and gap profile as a function of τ.

    Selects the optimal lag τ* as the checkpoint where the eigenvalue gap
    λ₁/λ₂ is maximized — the point where the slowest mode most dominates.

    Two calling conventions:
        # GMM-based:
        da_convergence_profile(gmm_means, gmm_covs, gmm_weights, ...)

        # Generic:
        da_convergence_profile(simulator=my_simulator, ...)

    Args:
        gmm_means: (k, d) component means (GMM path)
        gmm_covs: (k, d, d) component covariance matrices (GMM path)
        gmm_weights: (k,) mixture weights (GMM path)
        checkpoints: List of τ values to evaluate (default: geometric sequence)
        n_trajectories: Number of samples N
        dt: Langevin time step
        device: Computation device
        seed: Random seed
        reference_direction: (d,) unit vector to measure angles against.
                            If None, uses the first standard basis vector e₁.
        simulator: A simulator object (see extract_da_directions).

    Returns:
        Dictionary with:
            'checkpoints': (n_cp,) τ values
            'angles_deg': (n_cp,) angle of DA1 relative to reference, in degrees
            'leading_eigenvalues': (n_cp,) leading eigenvalue at each τ
            'all_eigenvalues': (n_cp, d) all eigenvalues at each τ
            'leading_eigenvectors': (n_cp, d) DA1 direction at each τ
            'gap_profile': (n_cp,) eigenvalue gap λ₁/λ₂ at each τ
            'optimal_idx': int, index into checkpoints where gap is maximized
            'optimal_eigenvalues': (d,) eigenvalues at τ*, descending
            'optimal_eigenvectors': (d, d) eigenvectors at τ*, columns = DA dirs
            'optimal_C_tau': (d, d) symmetrized C(τ) at τ*
    """
    device = device or torch.device('cpu')

    if checkpoints is None:
        checkpoints = [1, 2, 5, 10, 20, 50, 100, 200, 400, 800, 1600]
    checkpoints = sorted(checkpoints)
    T_max = checkpoints[-1]

    if seed is not None:
        torch.manual_seed(seed)

    sim, mu_f = _make_simulator(
        simulator=simulator,
        gmm_means=gmm_means, gmm_covs=gmm_covs, gmm_weights=gmm_weights,
        dt=dt, device=device,
    )

    d = mu_f.shape[0]

    if reference_direction is None:
        reference_direction = torch.zeros(d, device=device)
        reference_direction[0] = 1.0
    else:
        reference_direction = reference_direction.to(device=device)
        reference_direction = reference_direction / reference_direction.norm()

    # Sample X₀ from stationary distribution
    x0 = sim.sample_initial(n_trajectories)  # (N, d)

    # Center X₀
    x0_centered = x0 - mu_f.unsqueeze(0)  # (N, d)

    # Simulate forward, recording at checkpoints
    angles_deg = []
    leading_eigenvalues = []
    all_eigenvalues = []
    leading_eigenvectors = []
    top_k_eigenvectors = []  # top-k eigenvectors at each checkpoint
    gap_profile = []
    gap23_profile = []  # λ₂/λ₃ gap for DA2 optimal lag selection
    # Store full eigenvectors/eigenvalues/C_tau at each checkpoint for optimal selection
    all_evals_full = []
    all_evecs_full = []
    all_C_tau_full = []

    x = x0.clone()
    cp_idx = 0

    for t in range(1, T_max + 1):
        x = sim.step(x)

        if cp_idx < len(checkpoints) and t == checkpoints[cp_idx]:
            x_centered = x - mu_f.unsqueeze(0)
            C_tau = torch.matmul(x_centered.t(), x0_centered) / n_trajectories
            C_tau_sym = 0.5 * (C_tau + C_tau.t())

            evals, evecs = torch.linalg.eigh(C_tau_sym)
            # Sort descending
            idx_desc = torch.argsort(evals, descending=True)
            evals_desc = evals[idx_desc]
            evecs_desc = evecs[:, idx_desc]
            da1 = evecs_desc[:, 0]  # leading = largest eigenvalue

            # Angle between DA1 and reference direction
            cos_angle = torch.dot(da1, reference_direction).abs().clamp(-1, 1)
            angle = torch.acos(cos_angle) * 180.0 / np.pi

            # Eigenvalue gap λ₁/λ₂
            lam2 = evals_desc[1].item()
            gap = evals_desc[0].item() / lam2 if lam2 > 0 else float('inf')

            # Eigenvalue gap λ₂/λ₃ (for DA2 optimal lag)
            if len(evals_desc) > 2:
                lam3 = evals_desc[2].item()
                gap23 = lam2 / lam3 if lam3 > 0 else float('inf')
            else:
                gap23 = float('inf')

            angles_deg.append(angle.item())
            leading_eigenvalues.append(evals_desc[0].item())
            all_eigenvalues.append(evals_desc.cpu().numpy())
            leading_eigenvectors.append(da1.cpu().numpy())
            top_k_eigenvectors.append(evecs_desc[:, :min(5, d)].cpu().numpy())
            gap_profile.append(gap)
            gap23_profile.append(gap23)
            all_evals_full.append(evals_desc)
            all_evecs_full.append(evecs_desc)
            all_C_tau_full.append(C_tau_sym)

            cp_idx += 1

    # Select optimal lag: argmax of eigenvalue gap λ₁/λ₂
    gap_arr = np.array(gap_profile)
    optimal_idx = int(np.argmax(gap_arr))

    # Select DA2 optimal lag: argmax of eigenvalue gap λ₂/λ₃
    gap23_arr = np.array(gap23_profile)
    optimal_idx_da2 = int(np.argmax(gap23_arr))

    return {
        'checkpoints': np.array(checkpoints[:cp_idx]),
        'angles_deg': np.array(angles_deg),
        'leading_eigenvalues': np.array(leading_eigenvalues),
        'all_eigenvalues': np.array(all_eigenvalues),
        'leading_eigenvectors': np.array(leading_eigenvectors),
        'top_k_eigenvectors': np.array(top_k_eigenvectors),  # (n_cp, d, k)
        'gap_profile': gap_arr,
        'gap23_profile': gap23_arr,
        'optimal_idx': optimal_idx,
        'optimal_eigenvalues': all_evals_full[optimal_idx].cpu().numpy(),
        'optimal_eigenvectors': all_evecs_full[optimal_idx].cpu().numpy(),
        'optimal_C_tau': all_C_tau_full[optimal_idx].cpu().numpy(),
        'optimal_idx_da2': optimal_idx_da2,
        'optimal_eigenvalues_da2': all_evals_full[optimal_idx_da2].cpu().numpy(),
        'optimal_eigenvectors_da2': all_evecs_full[optimal_idx_da2].cpu().numpy(),
    }
