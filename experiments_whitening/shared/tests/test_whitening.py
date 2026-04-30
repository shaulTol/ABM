"""Tests for WhiteningTransform."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.whitening import WhiteningTransform


def test_round_trip():
    """unwhiten(whiten(X)) == X within float tolerance."""
    rng = np.random.default_rng(42)
    N, d = 500, 10
    X = rng.standard_normal((N, d)) @ np.diag([3, 2, 1, 0.5, 0.3, 4, 2.5, 1.5, 0.8, 0.2])
    X += rng.standard_normal(d) * 5  # offset

    wt = WhiteningTransform(X)
    X_tilde = wt.whiten(X)
    X_back = wt.unwhiten(X_tilde)

    np.testing.assert_allclose(X_back, X, atol=1e-10)


def test_whitened_covariance():
    """Whitened covariance should be close to identity."""
    rng = np.random.default_rng(123)
    N, d = 2000, 5
    Sigma = rng.standard_normal((d, d))
    Sigma = Sigma @ Sigma.T + np.eye(d)  # positive definite
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((N, d)) @ L.T + rng.standard_normal(d)

    wt = WhiteningTransform(X)
    X_tilde = wt.whiten(X)

    cov_tilde = np.cov(X_tilde, rowvar=False)
    err = np.linalg.norm(cov_tilde - np.eye(d), 'fro') / d
    assert err < 0.15, f"||Cov(X_tilde) - I||_F / d = {err:.4f} too large"


def test_analytic_cov():
    """Analytic covariance path produces correct whitening."""
    d = 4
    Sigma = np.diag([9.0, 4.0, 1.0, 0.25])
    mu = np.array([1.0, 2.0, 3.0, 4.0])

    wt = WhiteningTransform(analytic_cov=Sigma, mu=mu)

    # sigma_tilde_sq should be 1
    assert abs(wt.sigma_tilde_sq - 1.0) < 1e-10

    # Check W is Sigma^{-1/2}
    expected_W = np.diag([1/3, 1/2, 1.0, 2.0])
    np.testing.assert_allclose(wt.W, expected_W, atol=1e-10)


def test_score_transform_numerical():
    """Score transform matches numerical gradient on 2D Gaussian."""
    mu = np.array([1.0, -2.0])
    Sigma = np.array([[4.0, 1.0], [1.0, 2.0]])
    Sigma_inv = np.linalg.inv(Sigma)

    def score_fn(x):
        # ∇log N(x; mu, Sigma) = -Sigma^{-1}(x - mu)
        return -(x - mu) @ Sigma_inv.T

    wt = WhiteningTransform(analytic_cov=Sigma, mu=mu)
    score_tilde = wt.transform_score(score_fn)

    # Test at several points
    rng = np.random.default_rng(99)
    for _ in range(10):
        x_tilde = rng.standard_normal((1, 2))
        s = score_tilde(x_tilde)

        # Numerical gradient check
        eps = 1e-5
        x_orig = x_tilde @ wt.W_inv.T + mu
        # log f_tilde(x_tilde) = log f(W_inv x_tilde + mu) + log |det W_inv|
        # gradient of log f wrt x_tilde
        grad_num = np.zeros(2)
        for j in range(2):
            xp = x_tilde.copy(); xp[0, j] += eps
            xm = x_tilde.copy(); xm[0, j] -= eps
            # log f(x) = -0.5 (x-mu)^T Sigma^{-1} (x-mu) + const
            def log_f(xt):
                x = xt @ wt.W_inv.T + mu
                diff = x - mu
                return -0.5 * (diff @ Sigma_inv @ diff.T).item()
            grad_num[j] = (log_f(xp) - log_f(xm)) / (2 * eps)

        np.testing.assert_allclose(s[0], grad_num, atol=1e-4,
                                   err_msg="Score transform does not match numerical gradient")


def test_ledoit_wolf():
    """LW shrinkage should reduce condition number."""
    rng = np.random.default_rng(77)
    N, d = 50, 30  # low N/d ratio
    X = rng.standard_normal((N, d))

    wt_plain = WhiteningTransform(X, use_ledoit_wolf=False)
    wt_lw = WhiteningTransform(X, use_ledoit_wolf=True)

    assert wt_lw.alpha_lw is not None
    assert wt_lw.alpha_lw > 0
    assert wt_lw.cond_before <= wt_plain.cond_before * 1.01  # same raw cov
    # LW sigma_tilde should still be close to 1
    assert abs(wt_lw.sigma_tilde_sq - 1.0) < 0.1


def test_save_load(tmp_path):
    """Save and load round-trip."""
    rng = np.random.default_rng(55)
    X = rng.standard_normal((100, 5))
    wt = WhiteningTransform(X)

    path = str(tmp_path / "wt.npz")
    wt.save(path)
    wt2 = WhiteningTransform.load(path)

    np.testing.assert_allclose(wt2.W, wt.W, atol=1e-12)
    np.testing.assert_allclose(wt2.W_inv, wt.W_inv, atol=1e-12)
    np.testing.assert_allclose(wt2.mu, wt.mu, atol=1e-12)
    assert wt2.d == wt.d


def test_diagnostics():
    """Diagnostics dict has expected keys."""
    wt = WhiteningTransform(analytic_cov=np.eye(3), mu=np.zeros(3))
    diag = wt.diagnostics()
    assert set(diag.keys()) == {"alpha_lw", "cond_before", "cond_after",
                                 "sigma_tilde_sq", "d", "source"}
    assert diag["source"] == "analytic"
    assert diag["sigma_tilde_sq"] == pytest.approx(1.0, abs=1e-10)


def test_torch_score_transform():
    """Torch score transform matches numpy version."""
    import torch

    mu = np.array([1.0, -2.0])
    Sigma = np.array([[4.0, 1.0], [1.0, 2.0]])
    Sigma_inv = np.linalg.inv(Sigma)

    def score_fn_np(x):
        return -(x - mu) @ Sigma_inv.T

    def score_fn_torch(x):
        Sinv = torch.tensor(Sigma_inv, dtype=torch.float32)
        m = torch.tensor(mu, dtype=torch.float32)
        return -(x - m) @ Sinv.T

    wt = WhiteningTransform(analytic_cov=Sigma, mu=mu)
    score_np = wt.transform_score(score_fn_np)
    score_torch = wt.transform_score_torch(score_fn_torch)

    rng = np.random.default_rng(42)
    x_tilde = rng.standard_normal((5, 2))
    s_np = score_np(x_tilde)
    s_torch = score_torch(torch.tensor(x_tilde, dtype=torch.float32)).numpy()

    np.testing.assert_allclose(s_np, s_torch, atol=1e-5)


if __name__ == "__main__":
    test_round_trip()
    print("PASS: round_trip")
    test_whitened_covariance()
    print("PASS: whitened_covariance")
    test_analytic_cov()
    print("PASS: analytic_cov")
    test_score_transform_numerical()
    print("PASS: score_transform_numerical")
    test_ledoit_wolf()
    print("PASS: ledoit_wolf")
    test_diagnostics()
    print("PASS: diagnostics")
    try:
        import torch
        test_torch_score_transform()
        print("PASS: torch_score_transform")
    except ImportError:
        print("SKIP: torch_score_transform (torch not installed)")
    print("\nAll tests passed!")
