"""
KDE-based Autocorrelation Score Module

Provides functionality to compute the autocorrelation-based multimodality score
directly from raw samples using Kernel Density Estimation.

The main entry point is `autocorrelation_score()`.
"""

from .kde_estimator import KernelDensityEstimate
from .kde_score import KDEScore

__all__ = ['KernelDensityEstimate', 'KDEScore']

