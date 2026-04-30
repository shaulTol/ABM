"""
Whitening Transform for the whitening-variant experiment suite.

Transforms data X to X_tilde = W (X - mu) so that Cov(X_tilde) ≈ I.
Optionally applies Ledoit-Wolf shrinkage before inversion (for empirical
covariance estimates in high-dimensional settings).

The whitened diffusion is mathematically equivalent to using A_0 = Cov_f(X)
in the Langevin SDE, but avoids per-step covariance multiplications:
just whiten once, run isotropic Langevin with var_f ≈ 1, then unwhiten.
"""
import numpy as np
from typing import Callable, Optional, Union


class WhiteningTransform:
    """
    Compute and apply a whitening transform W = Sigma^{-1/2}.

    Usage:
        # From analytic covariance (GMM experiments):
        wt = WhiteningTransform(analytic_cov=Sigma_f, mu=mu_f)

        # From samples with Ledoit-Wolf shrinkage (empirical, e.g. CLIP):
        wt = WhiteningTransform(samples, use_ledoit_wolf=True)

        X_tilde = wt.whiten(X)
        X_back  = wt.unwhiten(X_tilde)
        score_tilde = wt.transform_score(original_score_fn)
    """

    def __init__(
        self,
        samples: Optional[np.ndarray] = None,
        *,
        analytic_cov: Optional[np.ndarray] = None,
        mu: Optional[np.ndarray] = None,
        use_ledoit_wolf: bool = False,
    ):
        """
        Args:
            samples: (N, d) data samples. Used for mean/covariance estimation
                     unless analytic_cov is provided.
            analytic_cov: (d, d) known covariance matrix (skips estimation).
            mu: (d,) known mean. If None, estimated from samples.
            use_ledoit_wolf: Apply Ledoit-Wolf shrinkage to the empirical
                             covariance before inversion.
        """
        if analytic_cov is not None:
            Sigma = np.asarray(analytic_cov, dtype=np.float64)
            d = Sigma.shape[0]
            self.mu = np.asarray(mu, dtype=np.float64) if mu is not None else np.zeros(d)
            self.alpha_lw = None
            self._source = "analytic"
        elif samples is not None:
            samples = np.asarray(samples, dtype=np.float64)
            N, d = samples.shape
            self.mu = samples.mean(axis=0) if mu is None else np.asarray(mu, dtype=np.float64)

            if use_ledoit_wolf:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf().fit(samples)
                Sigma = lw.covariance_
                self.alpha_lw = float(lw.shrinkage_)
                self._source = "empirical_lw"
            else:
                centered = samples - self.mu
                Sigma = (centered.T @ centered) / (N - 1)
                self.alpha_lw = None
                self._source = "empirical"
        else:
            raise ValueError("Must provide either samples or analytic_cov")

        self.d = d
        self.cond_before = float(np.linalg.cond(Sigma))

        # Eigendecomposition: Sigma = Q diag(lam) Q^T
        lam, Q = np.linalg.eigh(Sigma)
        lam = np.maximum(lam, 1e-12)  # clamp for stability

        # W = Sigma^{-1/2} = Q diag(1/sqrt(lam)) Q^T
        inv_sqrt_lam = 1.0 / np.sqrt(lam)
        self.W = (Q * inv_sqrt_lam[None, :]) @ Q.T        # (d, d)

        # W_inv = Sigma^{1/2} = Q diag(sqrt(lam)) Q^T
        sqrt_lam = np.sqrt(lam)
        self.W_inv = (Q * sqrt_lam[None, :]) @ Q.T        # (d, d)

        self.cond_after = float(np.linalg.cond(self.W @ Sigma @ self.W.T))

        # sigma_tilde_sq should be ≈ 1 if whitening is correct
        # tr(W Sigma W^T) / d = tr(I) / d = 1
        self.sigma_tilde_sq = float(np.trace(self.W @ Sigma @ self.W.T) / d)

    def whiten(self, X: np.ndarray) -> np.ndarray:
        """Transform X to whitened coordinates: X_tilde = (X - mu) @ W^T."""
        return (np.asarray(X) - self.mu) @ self.W.T

    def unwhiten(self, X_tilde: np.ndarray) -> np.ndarray:
        """Inverse transform: X = X_tilde @ W_inv^T + mu."""
        return np.asarray(X_tilde) @ self.W_inv.T + self.mu

    def transform_score(
        self, score_fn: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Return a score function in whitened coordinates.

        If score_fn computes ∇log f(x), the whitened score is:
            ∇log f_tilde(x_tilde) = W_inv^T @ ∇log f(W_inv @ x_tilde + mu)

        This follows from the change-of-variables formula for the gradient.
        """
        W_inv = self.W_inv
        mu = self.mu

        def score_tilde(x_tilde: np.ndarray) -> np.ndarray:
            # Map back to original coordinates
            x = x_tilde @ W_inv.T + mu
            # Evaluate original score
            grad = score_fn(x)
            # Transform gradient: W_inv^T @ grad
            return grad @ W_inv  # = (W_inv^T @ grad^T)^T row-wise
        return score_tilde

    def transform_score_torch(self, score_fn):
        """
        Return a score function in whitened coordinates (torch version).

        score_fn: (N, d) tensor -> (N, d) tensor
        Returns: callable (N, d) tensor -> (N, d) tensor in whitened space
        """
        import torch
        W_inv_t = torch.tensor(self.W_inv, dtype=torch.float32)
        mu_t = torch.tensor(self.mu, dtype=torch.float32)

        def score_tilde(x_tilde):
            device = x_tilde.device
            W_inv_dev = W_inv_t.to(device)
            mu_dev = mu_t.to(device)
            x = x_tilde @ W_inv_dev.T + mu_dev
            grad = score_fn(x)
            return grad @ W_inv_dev
        return score_tilde

    def diagnostics(self) -> dict:
        """Return diagnostic information about the whitening transform."""
        return {
            "alpha_lw": self.alpha_lw,
            "cond_before": self.cond_before,
            "cond_after": self.cond_after,
            "sigma_tilde_sq": self.sigma_tilde_sq,
            "d": self.d,
            "source": self._source,
        }

    def save(self, path: str):
        """Save whitening matrices and metadata to .npz."""
        np.savez(
            path,
            W=self.W,
            W_inv=self.W_inv,
            mu=self.mu,
            alpha_lw=self.alpha_lw if self.alpha_lw is not None else np.nan,
            cond_before=self.cond_before,
            cond_after=self.cond_after,
            sigma_tilde_sq=self.sigma_tilde_sq,
            d=self.d,
            source=self._source,
        )

    @classmethod
    def load(cls, path: str) -> "WhiteningTransform":
        """Load a saved WhiteningTransform from .npz."""
        data = np.load(path, allow_pickle=True)
        # Reconstruct from W and W_inv directly
        obj = object.__new__(cls)
        obj.W = data["W"]
        obj.W_inv = data["W_inv"]
        obj.mu = data["mu"]
        alpha = float(data["alpha_lw"])
        obj.alpha_lw = None if np.isnan(alpha) else alpha
        obj.cond_before = float(data["cond_before"])
        obj.cond_after = float(data["cond_after"])
        obj.sigma_tilde_sq = float(data["sigma_tilde_sq"])
        obj.d = int(data["d"])
        obj._source = str(data["source"])
        return obj
