"""
KDE Score Module

Provides a Gaussian-KDE score oracle ∇log f̂ for running the canonical
diffusion directly on raw samples.
"""

from .kde_score import KDEScore

__all__ = ['KDEScore']
