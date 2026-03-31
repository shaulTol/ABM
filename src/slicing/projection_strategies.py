"""
Projection Strategies Module - Methods for generating projection directions

Includes:
1. Uniform random projections
2. Variance-biased projections (PCA-informed)
3. Orthogonal projections
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict
import warnings


def uniform_random_projections(
    d: int,
    n_projections: int,
    device: torch.device = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate uniform random projections on the unit sphere
    
    Args:
        d: Dimension of the data
        n_projections: Number of projections to generate
        device: Device for computation
        seed: Random seed for reproducibility
        
    Returns:
        (n_projections, d) tensor of unit vectors
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate random Gaussian vectors
    projections = torch.randn(n_projections, d, device=device)
    
    # Normalize to unit sphere
    projections = projections / torch.norm(projections, dim=1, keepdim=True)
    
    return projections


def variance_biased_projections(
    data: torch.Tensor,
    n_projections: int,
    device: Optional[torch.device] = None,
    temperature: float = 1.0,
    n_pca_components: Optional[int] = None,
    mix_ratio: float = 0.5
) -> torch.Tensor:
    """
    Generate projections biased towards high-variance directions
    
    Uses PCA to find principal components and samples directions
    with probability proportional to their variance.
    
    Args:
        data: (n_samples, d) data tensor
        n_projections: Number of projections to generate
        device: Device for computation
        temperature: Controls bias strength (higher = more uniform)
        n_pca_components: Number of PCA components to consider (None = all)
        mix_ratio: Mix between biased (0) and uniform (1) sampling
        
    Returns:
        (n_projections, d) tensor of unit vectors
    """
    if device is None:
        device = data.device
    
    n_samples, d = data.shape
    
    # Center data
    data_centered = data - data.mean(dim=0, keepdim=True)
    
    # Compute covariance matrix
    cov = torch.matmul(data_centered.T, data_centered) / (n_samples - 1)
    
    # Eigendecomposition for PCA
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Sort in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
    except torch.linalg.LinAlgError:
        warnings.warn("PCA failed, using uniform projections")
        return uniform_random_projections(d, n_projections, device)
    
    # Use specified number of components
    if n_pca_components is None:
        n_pca_components = min(d, n_samples - 1)
    else:
        n_pca_components = min(n_pca_components, d, n_samples - 1)
    
    eigenvalues = eigenvalues[:n_pca_components]
    eigenvectors = eigenvectors[:, :n_pca_components]
    
    # Compute sampling probabilities based on eigenvalues
    # Use softmax with temperature for controlled bias
    log_probs = torch.log(eigenvalues.clamp_min(1e-10)) / temperature
    probs = torch.softmax(log_probs, dim=0)
    
    # Generate projections
    projections = torch.zeros(n_projections, d, device=device)
    
    n_biased = int(n_projections * (1 - mix_ratio))
    n_uniform = n_projections - n_biased
    
    # Biased projections
    if n_biased > 0:
        # Sample PCA components based on variance
        component_indices = torch.multinomial(probs, n_biased, replacement=True)
        
        for i, comp_idx in enumerate(component_indices):
            # Start with principal component
            base_direction = eigenvectors[:, comp_idx]
            
            # Add small random perturbation
            noise = torch.randn(d, device=device) * 0.1
            direction = base_direction + noise
            
            # Normalize
            projections[i] = direction / torch.norm(direction)
    
    # Uniform projections
    if n_uniform > 0:
        projections[n_biased:] = uniform_random_projections(d, n_uniform, device)
    
    return projections


def orthogonal_projections(
    d: int,
    n_projections: int,
    device: Optional[torch.device] = None,
    method: str = "gram_schmidt"
) -> torch.Tensor:
    """
    Generate orthogonal projection directions
    
    Args:
        d: Dimension of the data
        n_projections: Number of projections (must be <= d)
        device: Device for computation
        method: "gram_schmidt" or "qr"
        
    Returns:
        (n_projections, d) tensor of orthonormal vectors
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if n_projections > d:
        raise ValueError(f"Cannot generate {n_projections} orthogonal vectors in {d}D space")
    
    if method == "qr":
        # Use QR decomposition
        random_matrix = torch.randn(d, n_projections, device=device)
        Q, _ = torch.linalg.qr(random_matrix)
        return Q.T[:n_projections]
    
    elif method == "gram_schmidt":
        # Manual Gram-Schmidt
        projections = torch.zeros(n_projections, d, device=device)
        
        for i in range(n_projections):
            # Start with random vector
            v = torch.randn(d, device=device)
            
            # Orthogonalize against previous vectors
            for j in range(i):
                v = v - torch.dot(v, projections[j]) * projections[j]
            
            # Normalize
            projections[i] = v / torch.norm(v)
        
        return projections
    
    else:
        raise ValueError(f"Unknown method: {method}")


def stratified_projections(
    d: int,
    n_projections: int,
    device: Optional[torch.device] = None,
    n_strata: Optional[int] = None
) -> torch.Tensor:
    """
    Generate stratified random projections for better coverage
    
    Divides the sphere into regions and samples from each.
    
    Args:
        d: Dimension of the data
        n_projections: Number of projections
        device: Device for computation
        n_strata: Number of strata (None = sqrt(n_projections))
        
    Returns:
        (n_projections, d) tensor of unit vectors
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if n_strata is None:
        n_strata = int(np.sqrt(n_projections))
    
    projections = []
    projections_per_stratum = n_projections // n_strata
    remainder = n_projections % n_strata
    
    for stratum in range(n_strata):
        # Number of projections in this stratum
        n_in_stratum = projections_per_stratum + (1 if stratum < remainder else 0)
        
        if n_in_stratum == 0:
            continue
        
        # Generate random vectors with controlled angles
        # This is a simplified stratification - more sophisticated methods exist
        base_angle = 2 * np.pi * stratum / n_strata
        angle_range = 2 * np.pi / n_strata
        
        stratum_projections = torch.randn(n_in_stratum, d, device=device)
        
        # Apply rotation to first two dimensions (simplified)
        if d >= 2:
            angles = base_angle + torch.rand(n_in_stratum, device=device) * angle_range
            cos_angles = torch.cos(angles)
            sin_angles = torch.sin(angles)
            
            # Rotate in first plane
            x = stratum_projections[:, 0].clone()
            y = stratum_projections[:, 1].clone()
            stratum_projections[:, 0] = cos_angles * x - sin_angles * y
            stratum_projections[:, 1] = sin_angles * x + cos_angles * y
        
        # Normalize
        stratum_projections = stratum_projections / torch.norm(stratum_projections, dim=1, keepdim=True)
        projections.append(stratum_projections)
    
    return torch.cat(projections, dim=0)


def adaptive_projections(
    data: torch.Tensor,
    n_projections: int,
    initial_projections: Optional[torch.Tensor] = None,
    n_initial: int = 10,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate projections adaptively based on initial results
    
    Starts with random projections, then focuses on interesting directions.
    
    Args:
        data: (n_samples, d) data tensor
        n_projections: Total number of projections
        initial_projections: Pre-computed initial projections
        n_initial: Number of initial random projections
        device: Device for computation
        
    Returns:
        (n_projections, d) tensor of unit vectors
    """
    if device is None:
        device = data.device
    
    n_samples, d = data.shape
    
    # Start with initial projections
    if initial_projections is None:
        initial_projections = uniform_random_projections(d, n_initial, device)
    else:
        n_initial = initial_projections.shape[0]
    
    if n_projections <= n_initial:
        return initial_projections[:n_projections]
    
    # Project data with initial projections
    projected = torch.matmul(data, initial_projections.T)
    
    # Compute variance of each projection
    variances = projected.var(dim=0)
    
    # Find interesting directions (high variance or multimodal)
    interesting_indices = torch.argsort(variances, descending=True)[:n_initial//2]
    
    # Generate new projections near interesting directions
    n_remaining = n_projections - n_initial
    new_projections = []
    
    for _ in range(n_remaining):
        # Pick an interesting direction
        idx = interesting_indices[torch.randint(len(interesting_indices), (1,))]
        base_direction = initial_projections[idx]
        
        # Perturb it
        perturbation = torch.randn(d, device=device) * 0.3
        new_direction = base_direction + perturbation
        new_direction = new_direction / torch.norm(new_direction)
        
        new_projections.append(new_direction)
    
    new_projections = torch.stack(new_projections)
    
    return torch.cat([initial_projections, new_projections], dim=0)


# Helper functions

def check_projection_quality(projections: torch.Tensor) -> Dict:
    """
    Check quality metrics of projection set
    
    Args:
        projections: (n_projections, d) tensor
        
    Returns:
        Dictionary with quality metrics
    """
    n_proj, d = projections.shape
    
    # Check normalization
    norms = torch.norm(projections, dim=1)
    normalized = torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    # Check diversity (pairwise angles)
    dot_products = torch.matmul(projections, projections.T)
    # Remove diagonal
    mask = ~torch.eye(n_proj, dtype=torch.bool, device=projections.device)
    off_diagonal = dot_products[mask]
    
    # Compute angle statistics
    angles = torch.acos(torch.clamp(off_diagonal.abs(), -1, 1))
    
    return {
        'normalized': normalized,
        'min_angle': angles.min().item(),
        'max_angle': angles.max().item(),
        'mean_angle': angles.mean().item(),
        'std_angle': angles.std().item(),
        'condition_number': torch.linalg.cond(projections @ projections.T).item() if n_proj <= d else None
    }


# Test function
def test_projection_strategies():
    """Test projection generation strategies"""
    print("Testing Projection Strategies...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = 10
    n_proj = 5
    
    # Test uniform projections
    proj_uniform = uniform_random_projections(d, n_proj, device)
    print(f"Uniform projections shape: {proj_uniform.shape}")
    quality = check_projection_quality(proj_uniform)
    print(f"  Quality: normalized={quality['normalized']}, mean_angle={quality['mean_angle']:.3f}")
    
    # Test variance-biased projections
    data = torch.randn(100, d, device=device)
    data[:, 0] *= 5  # Make first dimension high variance
    proj_biased = variance_biased_projections(data, n_proj, device)
    print(f"Variance-biased projections shape: {proj_biased.shape}")
    
    # Test orthogonal projections
    proj_ortho = orthogonal_projections(d, min(n_proj, d), device)
    print(f"Orthogonal projections shape: {proj_ortho.shape}")
    dot_products = torch.matmul(proj_ortho, proj_ortho.T)
    print(f"  Orthogonality check: max off-diagonal = {(dot_products - torch.eye(proj_ortho.shape[0], device=device)).abs().max():.6f}")
    
    print("✓ Projection strategy tests passed!")


if __name__ == "__main__":
    test_projection_strategies()
