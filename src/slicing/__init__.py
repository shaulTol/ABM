"""
Slicing Module - Random Projection and Sliced Analysis
"""
from .sliced_autocorrelation import sliced_autocorrelation, full_autocorrelation
from .autocorsanitycheck import sliced_like_autocorrelation
from .projection_strategies import (
    uniform_random_projections,
    variance_biased_projections,
    orthogonal_projections
)

__all__ = [
    'sliced_autocorrelation',
    'full_autocorrelation',
    'sliced_like_autocorrelation',
    'uniform_random_projections',
    'variance_biased_projections',
    'orthogonal_projections'
]

