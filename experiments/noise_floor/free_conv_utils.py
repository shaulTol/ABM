"""
Free-convolution limit density and support edges for
    S(tau) = c_+(tau) W_+  -  c_-(tau) W_-
with W_+, W_- independent standard Wishart matrices of aspect gamma = d/N,
c_pm(tau) = (sigma^2 / 2)(1 +/- e^{-tau/2}).

Replaces the Bai-Yin / Frobenius upper-and-lower noise-floor bounds of
noise_floor_utils.py with the *exact* analytic support edges and the
*exact* limiting density. See MarchenkoPastourGeneralization.tex
Theorem~\\ref{thm:density} for the derivation.

Key identities used in the implementation:

    Stieltjes cubic (density):
        a(lam) G^3 + b(lam) G^2 + c(lam) G + d = 0
    with
        a(lam) = c_+ c_- gamma^2 * lam
        b(lam) = gamma * (c_+ c_- (2 - gamma) + lam (c_+ - c_-))
        c(lam) = (c_+ - c_-)(1 - gamma) - lam
        d      = 1
    Density = -Im[G(lam + i0^+)] / pi, where G is the root of the cubic
    lying in the lower half-plane (unique inside the support).

    Edge quartic (support endpoints):
        Delta_tau(lam) = 18 a b c d - 4 b^3 d + b^2 c^2 - 4 a c^3 - 27 a^2 d^2
    The two real roots of Delta_tau(lam) = 0 are (lam_-, lam_+).
    Degree 4, leading coefficient gamma^2 * (c_+ + c_-)^2 = gamma^2 * sigma^4.

Special cases:
  * tau = 0, c_- = 0 : cubic collapses to classical MP quadratic;
                       edges recover sigma^2 (1 +/- sqrt(gamma))^2.
  * tau = infinity, c_+ = c_- = sigma^2/2 : biquadratic in lam;
                       edges = +/- sigma^2/8 * (s+3)^{3/2} * sqrt(s-1)
                       with s = sqrt(1 + 4 gamma).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core coefficients
# ---------------------------------------------------------------------------

def c_plus_minus(tau: float, sigma: float = 1.0) -> tuple[float, float]:
    """Return (c_+(tau), c_-(tau)) = (sigma^2/2)(1 +/- e^{-tau/2})."""
    rho = float(np.exp(-float(tau) / 2.0))
    s2 = float(sigma) ** 2
    return 0.5 * s2 * (1.0 + rho), 0.5 * s2 * (1.0 - rho)


def stieltjes_cubic_coeffs(lam, c_plus: float, c_minus: float, gamma: float):
    """
    Coefficients (a, b, c, d) of the Stieltjes cubic

        a G^3 + b G^2 + c G + d = 0

    at scalar or array lam, for given (c_+, c_-, gamma).

    Supports vectorized lam (returns arrays).
    """
    lam = np.asarray(lam, dtype=float)
    p = c_plus * c_minus
    q = c_plus - c_minus
    a = p * gamma**2 * lam
    b = gamma * (p * (2.0 - gamma) + lam * q)
    c = q * (1.0 - gamma) - lam
    d = np.ones_like(lam)
    return a, b, c, d


# ---------------------------------------------------------------------------
# Support edges: real roots of the edge quartic
# ---------------------------------------------------------------------------

def _edge_quartic_as_poly1d(c_plus: float, c_minus: float, gamma: float) -> np.poly1d:
    """
    Return Delta_tau(lam) as a numpy.poly1d object (polynomial in lam).
    Built symbolically from a(lam), b(lam), c(lam), d using poly1d arithmetic.
    """
    # np.poly1d takes coefficients in highest-to-lowest order
    p = c_plus * c_minus
    q = c_plus - c_minus
    a = np.poly1d([p * gamma**2, 0.0])                    # p*gamma^2 * lam
    b = np.poly1d([gamma * q, gamma * p * (2.0 - gamma)]) # gamma*q*lam + gamma*p*(2-gamma)
    c = np.poly1d([-1.0, q * (1.0 - gamma)])              # -lam + q*(1-gamma)
    d = np.poly1d([1.0])

    Delta = 18 * a * b * c * d - 4 * b**3 * d + b**2 * c**2 - 4 * a * c**3 - 27 * a**2 * d**2
    return Delta


def edges(tau: float, sigma: float = 1.0, d: int = None, N: int = None,
          gamma: float = None) -> tuple[float, float]:
    """
    Return (lam_-, lam_+), the support endpoints of the limit distribution of
    S(tau), as the two real roots of the edge quartic Delta_tau(lam) = 0.

    Parameters
    ----------
    tau : float
        Lag.  Must be > 0 for the quartic to give clean edges; tau == 0 is
        handled by the MP closed form.
    sigma : float
        Marginal standard deviation of the OU stationary distribution.
    d, N : int
        Dimension and number of trajectories; only d/N = gamma is used.
    gamma : float, optional
        Aspect ratio d/N.  If provided, takes precedence over d, N.

    Returns
    -------
    (lam_minus, lam_plus) : tuple of floats
    """
    if gamma is None:
        if d is None or N is None:
            raise ValueError("Provide either gamma or (d, N).")
        gamma = float(d) / float(N)

    c_plus, c_minus = c_plus_minus(tau, sigma)

    # tau = 0 special case: cubic degenerates, closed-form MP edges
    if tau == 0.0 or c_minus == 0.0:
        mp_minus = sigma**2 * (1.0 - np.sqrt(gamma)) ** 2
        mp_plus = sigma**2 * (1.0 + np.sqrt(gamma)) ** 2
        return float(mp_minus), float(mp_plus)

    Delta = _edge_quartic_as_poly1d(c_plus, c_minus, gamma)
    roots = np.roots(Delta.coefficients)
    real_roots = roots[np.abs(roots.imag) < 1e-9].real

    if len(real_roots) < 2:
        raise RuntimeError(
            f"Edge quartic has {len(real_roots)} real roots at tau={tau}, "
            f"gamma={gamma}; expected 2."
        )
    # When there are >2 real roots (e.g. numerical artifacts near tau=0), take
    # the two outermost as the support edges.
    real_roots.sort()
    return float(real_roots[0]), float(real_roots[-1])


# ---------------------------------------------------------------------------
# Stieltjes transform and density
# ---------------------------------------------------------------------------

def stieltjes(lam: float, tau: float, sigma: float = 1.0,
              d: int = None, N: int = None, gamma: float = None) -> complex:
    """
    Compute G(lam + i0^+), the Stieltjes transform of the limit distribution
    of S(tau) evaluated just above the real axis at the point lam.

    Inside the support, returns a complex number with negative imaginary part
    (= -pi * density).  Outside the support, returns a real number (as complex
    with zero imaginary part).

    Implementation: solve the Stieltjes cubic and pick the root with the most
    negative imaginary part (or, if all roots are real, the one implied by the
    branch from infinity).
    """
    if gamma is None:
        if d is None or N is None:
            raise ValueError("Provide either gamma or (d, N).")
        gamma = float(d) / float(N)

    c_plus, c_minus = c_plus_minus(tau, sigma)
    a, b, c, dd = stieltjes_cubic_coeffs(lam, c_plus, c_minus, gamma)

    # tau = 0 special case: a == 0, cubic is actually a quadratic
    if abs(a) < 1e-15:
        # Solve b G^2 + c G + dd = 0
        if abs(b) < 1e-15:
            return complex(-dd / c, 0.0) if abs(c) > 1e-15 else complex(np.nan, np.nan)
        disc = c**2 - 4.0 * b * dd
        # Inside support: disc < 0 -> complex roots
        if disc < 0:
            root = (-c - 1j * np.sqrt(-disc)) / (2.0 * b)
        else:
            root = (-c - np.sqrt(disc)) / (2.0 * b)  # physical branch for MP
        return complex(root)

    # Solve the cubic a G^3 + b G^2 + c G + dd = 0
    cubic_roots = np.roots([a, b, c, dd])
    # Branch selection:
    #   - Inside the support: one real and two complex-conjugate roots.
    #     G(lam + i0^+) is the root with negative imaginary part.
    #   - Outside the support: three real roots; the physical G is the one that
    #     matches the large-|z| expansion G(z) ~ 1/z. We pick the real root
    #     whose sign matches -sign(1/lam) asymptotically, but for practical use
    #     any of them is consistent; density is 0 regardless.
    imag_parts = cubic_roots.imag
    if np.min(imag_parts) < -1e-10:
        # Pick root with most negative imaginary part
        return complex(cubic_roots[np.argmin(imag_parts)])
    # All real -> outside support; return the real root closest to 1/lam (asymptotic branch)
    # For density purposes, imaginary part is 0 so it doesn't matter much.
    cubic_roots_real = cubic_roots.real
    if lam > 0:
        target = 1.0 / lam if lam != 0 else 0.0
        return complex(cubic_roots_real[np.argmin(np.abs(cubic_roots_real - target))], 0.0)
    return complex(cubic_roots_real[0], 0.0)


def density(lam, tau: float, sigma: float = 1.0,
            d: int = None, N: int = None, gamma: float = None):
    """
    Evaluate the limiting density rho_tau^gamma(lam) of S(tau) at scalar or
    array lam.

    rho(lam) = -Im[G(lam + i0^+)] / pi,  G from the Stieltjes cubic.
    Zero outside the support interval [lam_-, lam_+].

    Vectorized over lam.
    """
    lam_arr = np.atleast_1d(np.asarray(lam, dtype=float))
    out = np.zeros_like(lam_arr)

    if gamma is None:
        if d is None or N is None:
            raise ValueError("Provide either gamma or (d, N).")
        gamma = float(d) / float(N)

    for i, li in enumerate(lam_arr):
        G = stieltjes(li, tau, sigma, gamma=gamma)
        out[i] = -G.imag / np.pi

    if np.isscalar(lam):
        return float(out[0])
    return out if out.shape != (1,) or np.ndim(lam) > 0 else float(out[0])


# ---------------------------------------------------------------------------
# Closed-form edges at the two endpoints (for unit tests / sanity checks)
# ---------------------------------------------------------------------------

def edge_tau0(sigma: float, gamma: float) -> tuple[float, float]:
    """Classical MP edges at tau = 0: sigma^2 (1 +/- sqrt(gamma))^2."""
    return (sigma**2 * (1.0 - np.sqrt(gamma)) ** 2,
            sigma**2 * (1.0 + np.sqrt(gamma)) ** 2)


def edge_tauinf(sigma: float, gamma: float) -> tuple[float, float]:
    """
    Closed-form symmetric edges at tau = infinity:
        lam_+(infinity) = sigma^2 / 8 * (s+3)^{3/2} * sqrt(s-1),  s = sqrt(1+4*gamma)
        lam_-(infinity) = -lam_+(infinity)
    """
    s = float(np.sqrt(1.0 + 4.0 * gamma))
    lam_plus = sigma**2 / 8.0 * (s + 3.0) ** 1.5 * np.sqrt(s - 1.0)
    return -lam_plus, lam_plus


# ---------------------------------------------------------------------------
# Optional: quick self-test / demo against simulation
# ---------------------------------------------------------------------------

def _demo(tau: float = 5.0, d: int = 200, N: int = 1000, sigma: float = 1.0,
          n_trials: int = 20, seed: int = 0, save_path: str | None = None):
    """
    Overlay the analytic density on an eigenvalue histogram from direct
    Wishart-difference sampling of S(tau).  Prints edge comparison.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    gamma = d / N

    eigs = []
    c_plus, c_minus = c_plus_minus(tau, sigma)
    for _ in range(n_trials):
        U = rng.standard_normal((d, N))
        V = rng.standard_normal((d, N))
        Wp = U @ U.T / N
        Wm = V @ V.T / N
        S = c_plus * Wp - c_minus * Wm
        eigs.append(np.linalg.eigvalsh(S))
    eigs = np.concatenate(eigs)

    lm, lp = edges(tau, sigma, gamma=gamma)
    print(f"tau = {tau}, gamma = {gamma}, sigma = {sigma}")
    print(f"  analytic edges:    ({lm:.4f}, {lp:.4f})")
    print(f"  empirical min/max: ({eigs.min():.4f}, {eigs.max():.4f})")

    lam_grid = np.linspace(lm - 0.05 * (lp - lm), lp + 0.05 * (lp - lm), 600)
    rho_grid = density(lam_grid, tau, sigma, gamma=gamma)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(eigs, bins=80, density=True, alpha=0.45,
            label=f"simulated eigenvalues (n_trials={n_trials}, d={d}, N={N})")
    ax.plot(lam_grid, rho_grid, "r-", lw=2, label="analytic density (free-conv.)")
    ax.axvline(lm, color="k", ls="--", lw=1, label=f"edges [{lm:.3f}, {lp:.3f}]")
    ax.axvline(lp, color="k", ls="--", lw=1)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\rho_\tau^\gamma(\lambda)$")
    ax.set_title(fr"$S(\tau)$ spectrum: $\tau={tau}$, $\gamma={gamma}$, $\sigma={sigma}$")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"  saved plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Demo: overlay analytic free-conv. density on simulated "
                    "eigenvalues of S(tau).")
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--d", type=int, default=200)
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None,
                        help="path to save plot (optional; otherwise shows)")
    args = parser.parse_args()

    # Sanity checks first
    sigma = args.sigma
    gamma = args.d / args.N
    print("Sanity checks:")
    e0 = edges(0.0, sigma, gamma=gamma)
    c0 = edge_tau0(sigma, gamma)
    print(f"  tau=0:    analytic={e0}, closed-form MP={c0}")
    e_inf = edges(100.0, sigma, gamma=gamma)  # tau=100 ~= infinity
    c_inf = edge_tauinf(sigma, gamma)
    print(f"  tau=100:  analytic={e_inf}, closed-form limit={c_inf}")
    print()

    _demo(tau=args.tau, d=args.d, N=args.N, sigma=args.sigma,
          n_trials=args.n_trials, seed=args.seed, save_path=args.save)
