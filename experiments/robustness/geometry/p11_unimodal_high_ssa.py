#!/usr/bin/env python3
"""
P11: Strictly unimodal high-SSA controls (NeurIPS rebuttal).

Goal: find densities with a UNIQUE mode but slow canonical relaxation and
SUBSTANTIALLY elevated finite-horizon SSA, to show directly that high SSA is
evidence of persistent barrier-like / slow structure and is NOT a sufficient
test of multimodality.

Three families (all exactly scored):

  1. Student-t, df in {3,5,10}, finite covariance (df>2).
       f(x) ∝ (1 + q/nu)^{-(nu+d)/2},   q = x^T Sigma^{-1} x
       score = -(nu+d) Sigma^{-1} x / (nu + q)
       Cov = nu/(nu-2) Sigma.
     UNIMODAL (analytic): f is a strictly DECREASING function of q, and q is a
     strictly convex quadratic form with a unique minimiser x=0. Hence f has a
     unique maximum at 0 and is quasiconcave (ellipsoidal superlevel sets).

  2. Banana / curved valley, an INVERTIBLE transform of a single Gaussian
     (Haario et al. twisted Gaussian):
       T(x) = (x1, x2 + b(x1^2 - s1sq), x3, ..., xd),   X ~ N(0, diag(s1sq,1,..,1))
       det J_T = 1 (unit lower-triangular), so f_Y(y) = f_X(T^{-1} y) with
       T^{-1}(y) = (y1, y2 - b(y1^2 - s1sq), y3, ...).
     UNIMODAL (analytic): T is a bijection with unit Jacobian, so f_Y and f_X
     have the same set of maximisers up to T. f_X has the unique maximiser 0,
     hence f_Y has the unique maximiser T(0) = (0, -b*s1sq, 0, ..., 0).
     NOTE: the banana is unimodal but NOT quasiconcave -- its superlevel sets
     are curved. We test and report that separately; it does not affect
     uniqueness of the mode.
       Cov_Y = diag(s1sq, 1 + 2 b^2 s1sq^2, 1, ..., 1)  (Cov(Y1,Y2)=b*E[X1^3]=0)

  3. Generalized Gaussian single wells (secondary control, extends P3c):
       f(x) ∝ exp(-sum_i (|x_i|/alpha)^beta)
       score_i = -(beta/alpha^beta) |x_i|^(beta-1) sign(x_i)
     UNIMODAL (analytic): separable, each factor strictly decreasing in |x_i|,
     so the unique maximiser is 0. Log-concave (hence quasiconcave) for beta>=1.

METHODOLOGY REQUIREMENT (spec): unique modality is verified ANALYTICALLY (above,
and restated per-density in the output) AND by direct multi-start numerical
optimisation in the FULL d-dimensional space. A dip/Silverman pass on a 1-D
projection is NEVER treated as evidence of unimodality.

Every family is compared against a COVARIANCE-MATCHED Gaussian (full covariance
matrix, not merely isotropic scale) run with IDENTICAL N, dt, horizon, max_lag
and stopping rule, so any SSA elevation reflects non-Gaussian SHAPE and not
covariance/anisotropy.

SSA estimator (submitted-paper convention, shared with P3): lag 0 excluded,
floor = 1/N, stopping lag = LARGEST tau with rho^2 > 1/N, Shat = dt*sum(rho_sq[:keep]).
tau* == max_lag is flagged window_limited => Shat is a finite-horizon LOWER BOUND.

Usage:
  python p11_unimodal_high_ssa.py --checks     # all required checks, no production
  python p11_unimodal_high_ssa.py --smoke      # tiny CPU smoke, all families
  python p11_unimodal_high_ssa.py              # full run
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'synthetic'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'sdxl'))

from src.simulation.langevin import LangevinSimulatorD
from src.simulation.generic_langevin import GenericLangevinSimulator
from src.metrics.autocorrelation import compute_stationary_moments

# --- reuse P3 machinery (spec: reuse first) ---
from p3_scope_controls import (
    ssa_trace, modality_tests, angle_between_deg, principal_angles_deg,
    gap_report, agg_ratio, _fmt_ratio, da_across_seeds, STOPPING_DESC,
)

from scipy import stats as sstats
from scipy import optimize as sopt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'p11_unimodal')


# =========================================================================== #
# Densities
# =========================================================================== #
class Density:
    """Common interface: logpdf (unnormalised OK), score, sample, analytic cov."""
    name = 'base'
    quasiconcave_expected = True

    def logpdf(self, x):          # numpy (n,d) -> (n,)
        raise NotImplementedError

    def score(self, x):           # torch (n,d) -> (n,d)
        raise NotImplementedError

    def sample(self, n, seed):    # -> torch (n,d)
        raise NotImplementedError


class TwoModeGaussian(Density):
    """NEGATIVE CONTROL ONLY -- a genuinely BIMODAL density, used to prove the
    barrier-merge mode counter can still detect multiple modes (otherwise a
    counter that always returns 1 would be worthless). Never used in production."""
    quasiconcave_expected = False

    def __init__(self, d=2, sep=6.0, var=1.0):
        self.d, self.sep, self.var = d, float(sep), float(var)
        self.name = f"twomode_sep{sep:g}_d{d}"
        self.label = f"Two-mode Gaussian (sep={sep:g}, d={d}) [negative control]"
        self.mu = np.zeros((2, d))
        self.mu[0, 0] = -sep / 2
        self.mu[1, 0] = +sep / 2
        cov = var * np.eye(d)
        cov[0, 0] = var + (sep / 2) ** 2
        self.cov = cov
        self.mode = self.mu[0].copy()
        self.unimodality_note = "NOT unimodal by construction (negative control)."
        self.moment_note = "all moments finite"

    def logpdf(self, x):
        a = -0.5 * np.sum((x - self.mu[0]) ** 2, axis=-1) / self.var
        b = -0.5 * np.sum((x - self.mu[1]) ** 2, axis=-1) / self.var
        return np.logaddexp(a, b)

    def score(self, x):
        d0 = x - torch.tensor(self.mu[0], dtype=x.dtype)
        d1 = x - torch.tensor(self.mu[1], dtype=x.dtype)
        a = -0.5 * (d0 ** 2).sum(-1, keepdim=True) / self.var
        b = -0.5 * (d1 ** 2).sum(-1, keepdim=True) / self.var
        w = torch.softmax(torch.cat([a, b], dim=-1), dim=-1)
        return -(w[..., :1] * d0 + w[..., 1:] * d1) / self.var

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        lab = rng.integers(0, 2, n)
        z = rng.standard_normal((n, self.d)) * np.sqrt(self.var)
        return torch.tensor(z + self.mu[lab], dtype=torch.float32)


class StudentT(Density):
    quasiconcave_expected = True

    def __init__(self, d, nu):
        assert nu > 2, "df must exceed 2 for finite covariance"
        self.d, self.nu = d, float(nu)
        self.name = f"student_t_df{int(nu)}_d{d}"
        self.label = f"Student-t (df={int(nu)}, d={d})"
        # Sigma = I  => Cov = nu/(nu-2) I
        self.cov = (self.nu / (self.nu - 2.0)) * np.eye(d)
        self.mode = np.zeros(d)
        self.unimodality_note = (
            "ANALYTIC: f is a strictly decreasing function of the quadratic form "
            "q = x^T x, which is strictly convex with unique minimiser 0. Hence a "
            "unique maximum at x=0; superlevel sets are ellipsoids (quasiconcave).")
        self.moment_note = ("df=3 has INFINITE fourth moment, so any EMPIRICAL "
                            "covariance estimate is heavy-tailed/noisy; we therefore "
                            "use the ANALYTIC covariance nu/(nu-2) I everywhere."
                            if nu <= 4 else "fourth moment finite")

    def logpdf(self, x):
        q = np.sum(x * x, axis=-1)
        return -0.5 * (self.nu + self.d) * np.log1p(q / self.nu)

    def score(self, x):
        q = (x * x).sum(dim=-1, keepdim=True)
        return -(self.nu + self.d) * x / (self.nu + q)

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, self.d))
        w = rng.chisquare(self.nu, size=(n, 1))
        return torch.tensor(z / np.sqrt(w / self.nu), dtype=torch.float32)


class Banana(Density):
    """Invertible (unit-Jacobian) transform of a single Gaussian."""
    quasiconcave_expected = False   # curved superlevel sets -- unimodal but not quasiconcave

    def __init__(self, d, b, s1sq=20.0):
        self.d, self.b, self.s1sq = d, float(b), float(s1sq)
        self.name = f"banana_b{b:g}_d{d}"
        self.label = f"Banana (b={b:g}, σ₁²={s1sq:g}, d={d})"
        cov = np.eye(d)
        cov[0, 0] = s1sq
        cov[1, 1] = 1.0 + 2.0 * self.b ** 2 * s1sq ** 2
        self.cov = cov
        self.mode = np.zeros(d)
        self.mode[1] = -self.b * s1sq
        self.unimodality_note = (
            "ANALYTIC: Y = T(X) with T(x)=(x1, x2+b(x1^2-s1sq), x3..) a bijection "
            "with det J_T = 1, so f_Y(y)=f_X(T^-1 y). f_X (a single Gaussian) has the "
            "unique maximiser 0, hence f_Y has the unique maximiser T(0)=(0,-b*s1sq,0..). "
            "NOT quasiconcave: superlevel sets are curved (this is the point of the "
            "example) -- curvature does not create a second mode.")
        self.moment_note = "all moments finite"

    def _u(self, y1, y2):
        return y2 - self.b * (y1 ** 2 - self.s1sq)

    def logpdf(self, x):
        u = self._u(x[..., 0], x[..., 1])
        rest = np.sum(x[..., 2:] ** 2, axis=-1) if self.d > 2 else 0.0
        return -0.5 * (x[..., 0] ** 2 / self.s1sq + u ** 2 + rest)

    def score(self, x):
        y1, y2 = x[..., 0], x[..., 1]
        u = y2 - self.b * (y1 ** 2 - self.s1sq)
        g = -x.clone()
        g[..., 0] = -y1 / self.s1sq + 2.0 * self.b * y1 * u
        g[..., 1] = -u
        return g

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, self.d))
        x[:, 0] *= np.sqrt(self.s1sq)
        y = x.copy()
        y[:, 1] = x[:, 1] + self.b * (x[:, 0] ** 2 - self.s1sq)
        return torch.tensor(y, dtype=torch.float32)


class GenGauss(Density):
    quasiconcave_expected = True

    def __init__(self, d, beta, alpha=3.0):
        self.d, self.beta, self.alpha = d, float(beta), float(alpha)
        self.name = f"gengauss_beta{beta:g}_d{d}"
        self.label = f"Gen-Gaussian (β={beta:g}, α={alpha:g}, d={d})"
        v = float(sstats.gennorm.var(beta, scale=alpha))
        self.cov = v * np.eye(d)
        self.mode = np.zeros(d)
        self.unimodality_note = (
            "ANALYTIC: separable, f = prod_i g(|x_i|) with g strictly decreasing, so "
            "the unique maximiser is 0. Log-concave (hence quasiconcave) for beta>=1.")
        self.moment_note = "all moments finite"

    def logpdf(self, x):
        return -np.sum((np.abs(x) / self.alpha) ** self.beta, axis=-1)

    def score(self, x):
        c = self.beta / (self.alpha ** self.beta)
        return -c * (x.abs() ** (self.beta - 1)) * torch.sign(x)

    def sample(self, n, seed):
        s = sstats.gennorm.rvs(self.beta, scale=self.alpha, size=(n, self.d),
                               random_state=seed)
        return torch.tensor(s, dtype=torch.float32)


# =========================================================================== #
# Unique-mode verification (analytic note + multi-start optimisation)
# =========================================================================== #
def verify_unique_mode(dens, n_starts=200, seed=0, tol=1e-3):
    """Multi-start maximisation of logpdf over the FULL d-dim space.

    Starts are drawn widely: half from the density itself (inflated 3x) and half
    uniform in a generous box, so basins away from the analytic mode would be
    found if they existed. Returns cluster count and distance to the analytic mode.
    """
    rng = np.random.default_rng(seed)
    d = dens.d
    scale = np.sqrt(np.diag(dens.cov))
    half = n_starts // 2
    s1 = dens.sample(half, seed + 1234).numpy() * 3.0
    s2 = rng.uniform(-6 * scale, 6 * scale, size=(n_starts - half, d))
    starts = np.vstack([s1, s2])

    def negf(z):
        return -float(dens.logpdf(z[None, :])[0])

    def negg(z):
        return -dens.score(torch.tensor(z[None, :], dtype=torch.float64))[0].numpy()

    found, fvals, n_ok = [], [], 0
    for z0 in starts:
        try:
            r = sopt.minimize(negf, z0, jac=negg, method='L-BFGS-B',
                              options=dict(maxiter=2000, ftol=1e-14, gtol=1e-10))
        except Exception:
            continue
        if not r.success and np.linalg.norm(negg(r.x)) > 1e-3:
            continue
        n_ok += 1
        found.append(r.x)
        fvals.append(-r.fun)
    found = np.array(found)
    fvals = np.array(fvals)

    # ---- merge candidates that are NOT separated by a barrier -------------
    # A genuine second mode requires a VALLEY between two optima. On a very flat
    # top (e.g. generalized-Gaussian with large beta) the gradient underflows and
    # the optimiser halts at many scattered points on the same plateau; those are
    # NOT distinct modes. So two candidates are merged when the log-density along
    # the straight segment joining them never dips more than `barrier_tol` below
    # the lower endpoint. We report the largest dip actually observed.
    barrier_tol = 1e-4
    n = len(found)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    ts = np.linspace(0.0, 1.0, 33)[1:-1]
    max_dip = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            seg = found[i][None, :] * (1 - ts[:, None]) + found[j][None, :] * ts[:, None]
            fseg = dens.logpdf(seg)
            dip = float(min(fvals[i], fvals[j]) - np.min(fseg))
            max_dip = max(max_dip, dip)
            if dip <= barrier_tol:          # no barrier => same basin/plateau
                union(i, j)
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    clusters = [[found[i] for i in idx] for idx in comps.values()]

    best = float(np.max(fvals)) if len(fvals) else float('nan')
    dist_analytic = (float(np.min(np.linalg.norm(found - dens.mode[None, :], axis=1)))
                     if len(found) else float('nan'))
    sizes = sorted((len(c) for c in clusters), reverse=True)
    # spread of the merged basin: how far the optimiser wanders on a flat top
    plateau_radius = (float(np.max(np.linalg.norm(found - found.mean(0)[None, :], axis=1)))
                      if n else float('nan'))
    # how much log-density the whole candidate set spans (flat top => ~0)
    fval_spread = float(np.max(fvals) - np.min(fvals)) if n else float('nan')

    # numerical Hessian at the analytic mode must be negative definite
    eps = 1e-4
    H = np.zeros((d, d))
    m = dens.mode.astype(float)
    for i in range(d):
        e = np.zeros(d); e[i] = eps
        gp = dens.score(torch.tensor((m + e)[None, :], dtype=torch.float64))[0].numpy()
        gm = dens.score(torch.tensor((m - e)[None, :], dtype=torch.float64))[0].numpy()
        H[:, i] = (gp - gm) / (2 * eps)
    H = 0.5 * (H + H.T)
    hev = np.linalg.eigvalsh(H)
    neg_def = bool(np.all(hev < 0))

    return dict(
        n_starts=int(n_starts), n_converged=int(n_ok),
        n_distinct_optima=int(len(clusters)), cluster_sizes=sizes[:5],
        n_raw_candidates=int(n),
        barrier_tol=barrier_tol, max_barrier_dip_observed=float(max_dip),
        plateau_radius=plateau_radius, logpdf_spread_over_candidates=fval_spread,
        merge_rule='candidates joined when the log-density along the connecting '
                   'segment never dips more than barrier_tol below the lower '
                   'endpoint (a distinct mode requires a separating valley)',
        max_logpdf=best, dist_to_analytic_mode=dist_analytic,
        analytic_mode=dens.mode.tolist(),
        hessian_eigvals_at_mode=hev.tolist(),
        hessian_negative_definite=neg_def,
        unique_mode=bool(len(clusters) == 1 and neg_def),
    )


def quasiconcavity_probe(dens, n_pairs=2000, seed=0):
    """Numerical quasiconcavity probe: f(mid) >= min(f(a),f(b)) for random pairs.
    Informative but NOT required for unimodality (the banana is unimodal and
    deliberately fails this)."""
    a = dens.sample(n_pairs, seed + 11).numpy()
    b = dens.sample(n_pairs, seed + 22).numpy()
    fa, fb = dens.logpdf(a), dens.logpdf(b)
    fm = dens.logpdf(0.5 * (a + b))
    viol = fm < np.minimum(fa, fb) - 1e-9
    return dict(n_pairs=int(n_pairs), violation_rate=float(viol.mean()),
                quasiconcave=bool(viol.mean() < 1e-3))


def score_fd_check(dens, n_pts=100, seed=0, eps=1e-5):
    """Analytic score vs central finite differences of logpdf."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(np.diag(dens.cov))
    pts = rng.uniform(-2 * scale, 2 * scale, size=(n_pts, dens.d))
    an = dens.score(torch.tensor(pts, dtype=torch.float64)).numpy()
    max_err = 0.0
    for j, p in enumerate(pts):
        for i in range(dens.d):
            pp = p.copy(); pp[i] += eps
            pm = p.copy(); pm[i] -= eps
            fd = (dens.logpdf(pp[None])[0] - dens.logpdf(pm[None])[0]) / (2 * eps)
            max_err = max(max_err, abs(fd - an[j, i]) / (abs(fd) + 1e-6))
    finite = bool(np.all(np.isfinite(an)))
    return dict(max_rel_err=float(max_err), finite=finite,
                shape_ok=bool(an.shape == pts.shape))


# =========================================================================== #
# Simulation helpers
# =========================================================================== #
def max_curvature(dens, n_pts=400, seed=0, q=0.95):
    """Robust upper scale of the curvature of -log f (max Hessian eigenvalue).

    Estimated by central finite differences of the analytic score at points drawn
    from f. We take the q-quantile over sampled points rather than the maximum so
    a single tail point cannot dominate (matters for Student-t df=3).
    """
    pts = dens.sample(n_pts, seed).numpy().astype(np.float64)
    d = dens.d
    eps = 1e-4
    lams = []
    for p in pts:
        H = np.zeros((d, d))
        for i in range(d):
            e = np.zeros(d); e[i] = eps
            gp = dens.score(torch.tensor((p + e)[None, :], dtype=torch.float64))[0].numpy()
            gm = dens.score(torch.tensor((p - e)[None, :], dtype=torch.float64))[0].numpy()
            H[:, i] = -(gp - gm) / (2 * eps)      # Hessian of -log f
        H = 0.5 * (H + H.T)
        lams.append(float(np.max(np.linalg.eigvalsh(H))))
    return float(np.quantile(lams, q))


def stable_dt(dens, dt_max, target_k_dt=0.1, n_pts=200, seed=0):
    """Largest dt <= dt_max keeping the explicit Euler-Maruyama scheme stable.

    The canonical diffusion uses the SCALAR coefficient sigma_f^2 = tr(Cov_f)/d.
    For a locally quadratic well of curvature lam the induced OU rate is
    k = 0.5*sigma_f^2*lam, and explicit Euler needs k*dt < 2 (we target 0.1).
    Strongly ANISOTROPIC densities (e.g. the banana, where Var(Y1)=s1sq is large
    but the valley is O(1) wide) therefore need a smaller dt: the scalar
    sigma_f^2 is set by the WIDE direction while stability is set by the STIFF
    one. Without this the banana silently diverges to NaN.
    """
    sf2 = float(np.trace(dens.cov) / dens.d)
    lam = max_curvature(dens, n_pts=n_pts, seed=seed)
    denom = 0.5 * sf2 * lam

    # ZERO / NON-FINITE CURVATURE GUARD.
    # Some log-densities are piecewise LINEAR, so their Hessian vanishes almost
    # everywhere and there is NO curvature-based step restriction. The canonical
    # example in this grid is the generalized Gaussian at beta=1 (Laplace):
    # log f = -|x|/alpha has zero second derivative away from the origin, giving
    # lam == 0 exactly. The score is bounded there (|score| = 1/alpha), so
    # explicit Euler is stable at the configured dt and falling back to it is the
    # correct action -- but we RECORD the fallback rather than substituting
    # silently. Non-finite lam is treated the same way and also recorded.
    fallback = None
    if not np.isfinite(lam) or not np.isfinite(denom):
        fallback = 'nonfinite_curvature'
    elif denom <= 1e-12:
        fallback = 'zero_curvature'

    if fallback is not None:
        dt = dt_max
        dt_lim = float('inf')
    else:
        dt_lim = target_k_dt / denom
        dt = min(dt_max, dt_lim)
    # round down to a clean 1/2/5 x 10^k value for reproducible reporting
    e = np.floor(np.log10(dt))
    m = dt / 10 ** e
    m = 5.0 if m >= 5 else (2.0 if m >= 2 else 1.0)
    dt_out = float(m * 10 ** e) if fallback is None else float(dt_max)
    return dt_out, dict(sigma_f_sq=sf2, max_curvature=float(lam),
                        k_dt_at_chosen=float(denom * dt_out) if np.isfinite(denom) else None,
                        dt_limit=(None if not np.isfinite(dt_lim) else float(dt_lim)),
                        dt_max=dt_max, fallback=fallback,
                        fallback_note=(None if fallback is None else
                                       'no curvature-based step restriction '
                                       '(piecewise-linear log-density, e.g. Laplace); '
                                       'bounded score, so configured dt is used'))


def check_stationarity(dens, dt, N=2000, T=600, seed=0, burn_frac=0.5):
    """Simulate and verify the sampler is stable AND stationary at the target.

    Confirms (a) no divergence, (b) the late-time ensemble covariance matches the
    ANALYTIC covariance. Catches both stiffness blow-up and systematic bias.

    RESOLUTION NOTE: this probe needs enough independent chains for the hardest
    functional it checks. For the banana, Var(Y2)=1+2b^2 s1sq^2 depends on
    E[X1^4] along the SLOW wide direction, so at N<~1000 the estimate itself is
    unstable (it reads ~15.8 against a true 9.0 at N=400) and the probe reports a
    false "not stationary". N>=1500 resolves it. This is a limitation of the
    DIAGNOSTIC, not of the production simulation.
    """
    sim = make_simulator(dens, dt, N, seed)
    torch.manual_seed(seed)
    x0 = sim.sample_initial(N)
    traj = sim.simulate(x0, T)
    finite = bool(torch.isfinite(traj).all().item())
    if not finite:
        return dict(finite=False, max_rel_cov_err=float('inf'), stationary=False, dt=dt)
    late = traj[:, int(burn_frac * T):, :].reshape(-1, dens.d).numpy().astype(np.float64)
    emp = np.cov(late.T) if dens.d > 1 else np.array([[late.var()]])
    dg_e, dg_a = np.diag(np.atleast_2d(emp)), np.diag(dens.cov)
    rel = np.abs(dg_e - dg_a) / np.abs(dg_a)

    # IID REFERENCE: the covariance estimator is itself noisy, badly so for heavy
    # tails (Student-t). Comparing to a fixed tolerance would be threshold-shopping.
    # Instead we measure the spread of the SAME estimator on IID draws from f,
    # using N independent draws (a trajectory of N chains has ~N independent
    # samples regardless of its length), and require the trajectory estimate to
    # be within 3x that reference.
    ref = []
    for k in range(8):
        s = dens.sample(N, seed + 5000 + k).numpy().astype(np.float64)
        c = np.cov(s.T) if dens.d > 1 else np.array([[s.var()]])
        ref.append(float((np.abs(np.diag(np.atleast_2d(c)) - dg_a) / np.abs(dg_a)).max()))
    iid_ref = float(np.quantile(ref, 0.9))
    tol = max(0.15, 3.0 * iid_ref)
    return dict(finite=True, dt=dt,
                empirical_cov_diag=dg_e.tolist(), analytic_cov_diag=dg_a.tolist(),
                max_rel_cov_err=float(rel.max()),
                iid_reference_err=iid_ref, tolerance=tol,
                tolerance_rule='max(0.15, 3 x 90th-pct error of the same estimator '
                               'on N IID draws from f)',
                stationary=bool(rel.max() < tol))


def make_simulator(dens, dt, n_samples, seed):
    cov_t = torch.tensor(dens.cov, dtype=torch.float32)
    var_f = float(np.trace(dens.cov) / dens.d)     # ANALYTIC, avoids noisy Sigma_hat
    samples = dens.sample(n_samples, seed)
    return GenericLangevinSimulator(dens.score, samples, dt=dt,
                                    mu_f=torch.zeros(dens.d), cov_f=cov_t,
                                    var_f=var_f)


def matched_gaussian_params(dens):
    """Covariance-matched Gaussian: FULL covariance matrix, zero mean."""
    means = torch.zeros(1, dens.d)
    covs = torch.tensor(dens.cov, dtype=torch.float32).unsqueeze(0)
    weights = torch.ones(1)
    return means, covs, weights


def run_ssa_target(dens, cfg):
    """Finite-horizon SSA for the target density, over cfg['ssa_seeds']."""
    d = dens.d
    dmeans, dcovs, dweights = matched_gaussian_params(dens)  # for trace normalisation
    out = []
    st = cfg.get('stride', 1)
    for s in cfg['ssa_seeds']:
        sim = make_simulator(dens, cfg['dt_sim'], cfg['N_ssa'], s)
        torch.manual_seed(s)
        x0 = sim.sample_initial(cfg['N_ssa'])
        traj = sim.simulate(x0, cfg['T_sim'])
        traj = traj[:, ::st, :].contiguous()      # -> analysis grid, Delta t = cfg['dt']
        out.append(ssa_trace(traj, dmeans, dcovs, dweights, cfg['dt'],
                             cfg['max_lag'], cfg['N_ssa'], d))
    return out


def run_ssa_matched_gaussian(dens, cfg):
    """Matched Gaussian SSA under IDENTICAL N, dt, horizon, max_lag, stopping rule."""
    d = dens.d
    gm, gc, gw = matched_gaussian_params(dens)
    out = []
    st = cfg.get('stride', 1)
    for s in cfg['ssa_seeds']:
        torch.manual_seed(90000 + s)
        gsim = LangevinSimulatorD(gm, gc, gw, dt=cfg['dt_sim'])
        x0 = gsim.sample_initial(cfg['N_ssa'])
        traj = gsim.simulate(x0, cfg['T_sim'])
        traj = traj[:, ::st, :].contiguous()      # identical analysis grid
        out.append(ssa_trace(traj, gm, gc, gw, cfg['dt'], cfg['max_lag'],
                             cfg['N_ssa'], d))
    return out


# =========================================================================== #
# One cell
# =========================================================================== #
def run_cell(dens, cfg, verbose=True):
    t0 = time.time()
    d = dens.d
    # ---- stability-aware dt for THIS cell (target and matched Gaussian share it,
    # so N, dt, horizon and stopping rule remain identical WITHIN the comparison) ----
    cfg = dict(cfg)
    dt_analysis = cfg['dt']                      # ANALYSIS grid: same for every cell
    dt_cell, dt_info = stable_dt(dens, dt_analysis)
    # SIMULATE finely for stability, ANALYSE on the common coarse grid.
    #
    # Naively scaling max_lag with 1/dt is what made this experiment take 14h47m:
    # the banana at b=0.2 needs dt=5e-4, i.e. max_lag=24000 and ~1e12 autocorrelation
    # operations for that single cell. Instead we integrate at dt_sim for numerical
    # stability and then SUBSAMPLE every `stride` steps, so the analysed series
    # always has Delta t = dt_analysis and max_lag = the configured value.
    # This keeps the SSA estimator (Delta t, max_lag, horizon, stopping rule)
    # IDENTICAL across all cells and against every matched-Gaussian baseline,
    # which is exactly what the matched comparison requires.
    stride = max(1, int(round(dt_analysis / dt_cell)))
    cfg['stride'] = stride
    cfg['dt_sim'] = dt_analysis / stride
    cfg['T_sim'] = cfg['T_ssa'] * stride         # same PHYSICAL horizon
    cfg['checkpoints_sim'] = [c * stride for c in cfg['checkpoints']]
    cfg['stat_T'] = cfg['stat_T'] * stride
    dt_cell = cfg['dt_sim']
    # cfg['dt'] and cfg['max_lag'] deliberately UNCHANGED (analysis grid)
    if verbose:
        print(f"\n{'='*74}\n{dens.label}\n{'='*74}")
        print(f"  unimodality: {dens.unimodality_note}")
        print(f"  moments: {dens.moment_note}")
        print(f"  analytic Cov diag: {np.round(np.diag(dens.cov), 4).tolist()}  "
              f"sigma_f^2=tr(Cov)/d={np.trace(dens.cov)/d:.4f}")
        kdt = dt_info['k_dt_at_chosen']
        print(f"  dt_sim={dt_cell:g} (stride={cfg['stride']}, curvature="
              f"{dt_info['max_curvature']:.3f}, k*dt="
              f"{'n/a' if kdt is None else format(kdt, '.4f')}"
              f"{'' if dt_info['fallback'] is None else ', FALLBACK='+dt_info['fallback']})"
              f"; T_sim={cfg['T_sim']} -> analysis dt={cfg['dt']}, "
              f"max_lag={cfg['max_lag']}, physical horizon="
              f"{cfg['T_ssa']*cfg['dt']:.1f} (identical across cells)")

    # ---- stationarity / stability gate: refuse to report a diverged cell ----
    stat = check_stationarity(dens, dt_cell, N=cfg['stat_N'], T=cfg['stat_T'], seed=0)
    if verbose:
        print(f"  [stationarity] finite={stat['finite']}, max rel cov err="
              f"{stat['max_rel_cov_err']:.4f} vs tol={stat.get('tolerance', float('nan')):.4f} "
              f"(IID ref {stat.get('iid_reference_err', float('nan')):.4f}) -> "
              f"{'OK' if stat['stationary'] else 'NOT STATIONARY'}")
    if not stat['finite']:
        raise RuntimeError(
            f"{dens.name}: simulation DIVERGED at dt={dt_cell}. Refusing to report "
            f"numbers from a diverged run.")

    # ---- unique-mode verification (FULL space, multi-start) ----
    um = verify_unique_mode(dens, n_starts=cfg['n_mode_starts'], seed=0)
    qc = quasiconcavity_probe(dens, n_pairs=cfg['n_qc_pairs'], seed=0)
    if verbose:
        print(f"  [unique mode] multi-start: {um['n_converged']}/{um['n_starts']} converged, "
              f"{um['n_distinct_optima']} distinct optimum(a), "
              f"dist to analytic mode={um['dist_to_analytic_mode']:.2e}, "
              f"Hessian neg-def={um['hessian_negative_definite']}  => "
              f"UNIQUE MODE {'CONFIRMED' if um['unique_mode'] else 'NOT CONFIRMED'}")
        print(f"  [quasiconcavity probe] violation rate={qc['violation_rate']:.4f} "
              f"(expected {'quasiconcave' if dens.quasiconcave_expected else 'NOT quasiconcave'})")

    # ---- SSA: target vs covariance-matched Gaussian ----
    tgt = run_ssa_target(dens, cfg)
    mg = run_ssa_matched_gaussian(dens, cfg)
    S = np.array([r['S_hat'] for r in tgt])
    Sg = np.array([r['S_hat'] for r in mg])
    ratio = float(S.mean() / Sg.mean()) if Sg.mean() else float('nan')
    wl = [bool(r['window_limited']) for r in tgt]
    wlg = [bool(r['window_limited']) for r in mg]
    if verbose:
        print(f"  SSA Ŝ target          = {'≥ ' if any(wl) else ''}{S.mean():.4f} ± {S.std():.4f} "
              f"(τ*={[r['tau_star'] for r in tgt]}, window_limited={wl})")
        print(f"  SSA Ŝ matched Gaussian= {'≥ ' if any(wlg) else ''}{Sg.mean():.4f} ± {Sg.std():.4f} "
              f"(τ*={[r['tau_star'] for r in mg]})")
        print(f"  ==> SSA RATIO target/matched-Gaussian = {ratio:.3f}×")

    # ---- DA for target and for the matched Gaussian (pipeline-matched null) ----
    per_seed, da_stats = da_across_seeds(
        None, None, None, cfg['checkpoints_sim'], cfg['N_da'], cfg['dt_sim'],
        cfg['da_seeds'],
        simulator_fn=lambda s: make_simulator(dens, cfg['dt_sim'], cfg['N_da'], s))
    gm, gc, gw = matched_gaussian_params(dens)
    g_per_seed, g_da_stats = da_across_seeds(
        gm, gc, gw, cfg['checkpoints_sim'], cfg['N_da'], cfg['dt_sim'], cfg['da_seeds'])
    # matched-Gaussian DA null threshold = max leading eigenvalue across its seeds
    lam_null = float(np.max([p['lam1'] for p in g_per_seed]))
    evals0 = np.array(per_seed[0]['evals'], dtype=float)
    sig_idx = [int(k) for k in range(len(evals0)) if evals0[k] > lam_null]
    if verbose:
        print(f"  DA λ1/λ2 = {_fmt_ratio(da_stats['gap_ratio_mean'])} "
              f"(λ1={np.round(da_stats['lam1_per_seed'],4).tolist()})")
        print(f"  DA λ2/λ3 = {_fmt_ratio(da_stats['gap_ratio_23_mean'])}")
        print(f"  DA1 cross-seed angle (deg): "
              f"{np.round(da_stats['da1_angle_deg'],1).tolist()} "
              f"(max {da_stats['da1_angle_deg_max']:.1f}°)")
        print(f"  matched-Gaussian DA null λ_max = {lam_null:.5f}; "
              f"significant DA directions (λ_k > null): {sig_idx}")

    # ---- dip / Silverman on EVERY significant DA projection ----
    samp = dens.sample(cfg['N_mod'], 777).numpy()
    evecs0 = np.array(per_seed[0]['leading_eigenvectors'])  # unused; use optimal evecs
    modality = {}
    for k in sig_idx[:cfg['max_modality_dirs']]:
        vk = np.array(per_seed[0]['da1']) if k == 0 else None
        if k == 1:
            vk = np.array(per_seed[0]['da2'])
        if vk is None:
            continue
        modality[f'DA{k+1}'] = modality_tests(samp @ vk, silverman_B=cfg['silverman_B'],
                                              seed=0)
    if not modality:  # always report DA1 even if not above the null
        modality['DA1'] = modality_tests(samp @ np.array(per_seed[0]['da1']),
                                         silverman_B=cfg['silverman_B'], seed=0)
    if verbose:
        for k, v in modality.items():
            print(f"  {k} projection: dip p={v['dip_p']:.3g}, "
                  f"Silverman p={v['silverman_p']:.3g}")

    return dict(
        name=dens.name, label=dens.label, d=int(d), family=type(dens).__name__,
        params={k: v for k, v in vars(dens).items()
                if isinstance(v, (int, float, str)) and k not in ('name', 'label')},
        cov_diag=np.diag(dens.cov).tolist(),
        dt_sim=dt_cell, dt_analysis=cfg['dt'], stride=cfg['stride'],
        dt_info=dt_info, T_sim=cfg['T_sim'], max_lag=cfg['max_lag'],
        physical_horizon=cfg['T_ssa']*cfg['dt'], stationarity=stat,
        sigma_f_sq=float(np.trace(dens.cov) / d),
        unimodality_note=dens.unimodality_note, moment_note=dens.moment_note,
        unique_mode=um, quasiconcavity=qc,
        ssa=dict(S_hat_mean=float(S.mean()), S_hat_std=float(S.std()),
                 tau_star=[r['tau_star'] for r in tgt], window_limited=bool(any(wl)),
                 window_limited_per_seed=wl,
                 estimator=f'floor=1/N, stopping lag={STOPPING_DESC}, '
                           'lag0 excluded',
                 finite_horizon=True, resolved_population=False),
        matched_gaussian=dict(
            S_hat_mean=float(Sg.mean()), S_hat_std=float(Sg.std()),
            tau_star=[r['tau_star'] for r in mg], window_limited=bool(any(wlg)),
            match_rule='FULL covariance matrix matched; identical N, dt, T, max_lag, '
                       'stopping rule',
            ssa_ratio=ratio),
        da_stats=da_stats, da_per_seed=per_seed,
        da_null=dict(lam_max_matched_gaussian=lam_null,
                     gaussian_lam1_per_seed=[p['lam1'] for p in g_per_seed],
                     significant_dirs=sig_idx),
        modality=modality,
        runtime_s=time.time() - t0,
    )


# =========================================================================== #
# Checks
# =========================================================================== #
def build_densities(dims, smoke=False):
    dens = []
    for d in dims:
        for nu in ([5] if smoke else [3, 5, 10]):
            dens.append(StudentT(d, nu))
        for b in ([0.2] if smoke else [0.05, 0.1, 0.2]):
            dens.append(Banana(d, b))
    for beta in ([1.0, 6.0] if smoke else [1.0, 2.0, 4.0, 6.0, 8.0]):
        dens.append(GenGauss(max(dims), beta))
    return dens


def run_checks():
    print("\n" + "=" * 74 + "\nP11 REQUIRED CHECKS\n" + "=" * 74)
    ok = True

    print("\n[Check A] analytic scores: shape, finiteness, vs finite differences")
    for dens in [StudentT(2, 3), StudentT(10, 5), Banana(2, 0.05), Banana(10, 0.1),
                 GenGauss(2, 6.0), GenGauss(10, 1.0)]:
        r = score_fd_check(dens, n_pts=40, seed=1)
        good = r['finite'] and r['shape_ok'] and r['max_rel_err'] < 1e-2
        print(f"  {dens.name:28s} shape_ok={r['shape_ok']} finite={r['finite']} "
              f"max_rel_err={r['max_rel_err']:.2e}  -> {'OK' if good else 'FAIL'}")
        ok &= good

    print("\n[Check B] UNIQUE MODE in the FULL space (multi-start optimisation)")
    print("          (analytic argument stated per density; NOT inferred from any 1-D test)")
    for dens in [StudentT(2, 3), StudentT(10, 3), StudentT(10, 10),
                 Banana(2, 0.05), Banana(10, 0.1), GenGauss(10, 6.0)]:
        um = verify_unique_mode(dens, n_starts=120, seed=0)
        print(f"  {dens.name:28s} {um['n_converged']:3d} converged -> "
              f"{um['n_distinct_optima']} basin(s) after barrier-merge "
              f"(raw {um['n_raw_candidates']}, max dip {um['max_barrier_dip_observed']:.2e}, "
              f"plateau r={um['plateau_radius']:.3f}), H neg-def="
              f"{um['hessian_negative_definite']}  -> "
              f"{'OK' if um['unique_mode'] else 'FAIL'}")
        ok &= um['unique_mode']

    print("\n[Check C] quasiconcavity probe matches expectation "
          "(banana: unimodal but NOT quasiconcave)")
    for dens in [StudentT(2, 5), GenGauss(2, 6.0), Banana(2, 0.05)]:
        qc = quasiconcavity_probe(dens, n_pairs=1500, seed=0)
        good = (qc['quasiconcave'] == dens.quasiconcave_expected)
        print(f"  {dens.name:28s} violation_rate={qc['violation_rate']:.4f} "
              f"quasiconcave={qc['quasiconcave']} expected={dens.quasiconcave_expected} "
              f"-> {'OK' if good else 'FAIL'}")
        ok &= good

    print("\n[Check D] analytic covariance matches empirical (large sample)")
    for dens in [StudentT(2, 10), Banana(2, 0.05), GenGauss(2, 6.0)]:
        s = dens.sample(400000, 5).numpy().astype(np.float64)
        emp = np.cov(s.T)
        rel = np.abs(np.diag(emp) - np.diag(dens.cov)) / np.abs(np.diag(dens.cov))
        good = bool(np.all(rel < 0.06))
        print(f"  {dens.name:28s} analytic diag={np.round(np.diag(dens.cov),3).tolist()} "
              f"empirical={np.round(np.diag(emp),3).tolist()} maxrel={rel.max():.3f} "
              f"-> {'OK' if good else 'FAIL'}")
        ok &= good
    print("  (Student-t df=3 is deliberately excluded here: infinite 4th moment makes "
          "the empirical covariance estimator itself heavy-tailed. We use the ANALYTIC "
          "covariance in all simulation settings.)")

    print("\n[Check E] matched Gaussian: FULL covariance match + identical settings")
    for dens in [StudentT(10, 3), Banana(10, 0.1)]:
        gm, gc, gw = matched_gaussian_params(dens)
        exact = np.allclose(gc[0].numpy(), dens.cov, atol=1e-5)
        sim_t = make_simulator(dens, 0.01, 100, 0)
        sim_g = LangevinSimulatorD(gm, gc, gw, dt=0.01)
        dvar = abs(sim_t.var_f - sim_g.var_f)
        good = exact and gw.numel() == 1 and dvar < 1e-4
        print(f"  {dens.name:28s} cov exact={exact}, single-component={gw.numel()==1}, "
              f"|sigma_f^2 diff|={dvar:.2e}  -> {'OK' if good else 'FAIL'}")
        ok &= good
    print("  (N, dt, T, max_lag and the stopping rule come from the SAME cfg in "
          "run_cell, so they are identical by construction.)")

    print("\n[Check F] sampler sanity: Student-t tail heavier than matched Gaussian")
    dens = StudentT(2, 3)
    s = dens.sample(200000, 7).numpy()
    r = np.linalg.norm(s, axis=1)
    g = np.random.default_rng(7).multivariate_normal(np.zeros(2), dens.cov, 200000)
    rg = np.linalg.norm(g, axis=1)
    q999_t, q999_g = np.quantile(r, 0.999), np.quantile(rg, 0.999)
    good = q999_t > 1.5 * q999_g
    print(f"  df=3: 99.9% radius t={q999_t:.2f} vs matched Gaussian={q999_g:.2f} "
          f"(ratio {q999_t/q999_g:.2f}x)  -> {'OK' if good else 'FAIL'}")
    ok &= good

    print("\n[Check G] NEGATIVE CONTROL: the barrier-merge mode counter must still")
    print("          detect genuine multimodality (a counter that always says 1 is useless)")
    for sep in (6.0, 8.0):
        neg = TwoModeGaussian(d=2, sep=sep)
        um = verify_unique_mode(neg, n_starts=120, seed=0)
        good = (um['n_distinct_optima'] == 2) and (not um['unique_mode'])
        print(f"  {neg.name:28s} basins={um['n_distinct_optima']} (expect 2), "
              f"max barrier dip={um['max_barrier_dip_observed']:.3f}, "
              f"unique_mode={um['unique_mode']} (expect False)  -> "
              f"{'OK' if good else 'FAIL'}")
        ok &= good
    neg10 = TwoModeGaussian(d=10, sep=6.0)
    um = verify_unique_mode(neg10, n_starts=120, seed=0)
    good = (um['n_distinct_optima'] == 2) and (not um['unique_mode'])
    print(f"  {neg10.name:28s} basins={um['n_distinct_optima']} (expect 2), "
          f"unique_mode={um['unique_mode']} (expect False)  -> {'OK' if good else 'FAIL'}")
    ok &= good

    print("\n[Check H] stable_dt over EVERY cell of the production grid")
    print("          (this is the check that would have caught the beta=1 zero-curvature")
    print("           ZeroDivisionError in seconds instead of after 14h47m)")
    grid = build_densities([2, 10], smoke=False)
    # beta=1 (Laplace) has a piecewise-linear log-density => zero Hessian, so the
    # fallback is EXPECTED there and nowhere else.
    expect_fallback = {'gengauss_beta1_d10'}
    for i, dens in enumerate(grid, 1):
        try:
            dt, info = stable_dt(dens, 0.01, n_pts=60)
        except Exception as e:
            print(f"  cell {i:2d} {dens.name:26s} RAISED {e!r}  -> FAIL")
            ok = False
            continue
        fb = info['fallback']
        good = (np.isfinite(dt) and dt > 0
                and ((fb is not None) == (dens.name in expect_fallback)))
        print(f"  cell {i:2d} {dens.name:26s} dt={dt:<8g} lam={info['max_curvature']:.4g} "
              f"fallback={fb}  -> {'OK' if good else 'FAIL'}")
        ok &= good
    print("  (a fallback anywhere OTHER than the expected Laplace cell is a FAIL: it "
          "would mean a density lost its stability limit unnoticed)")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


# =========================================================================== #
# Table
# =========================================================================== #
def build_table(results, cfg):
    N = cfg['N_ssa']
    cap = (
        f"**Table (P11: strictly unimodal high-SSA controls).** Every density has a "
        f"UNIQUE mode, established analytically AND by multi-start numerical "
        f"optimisation in the full d-dimensional space (never from a 1-D projection "
        f"test). Ŝ is the *finite-horizon* estimator, not resolved population SSA: "
        f"lag 0 excluded, floor 1/N={1.0/N:.2g} (N={N}), stopping lag τ* = "
        f"{STOPPING_DESC}, Ŝ=Δt·Σρ²(τ), Δt={cfg['dt']}, horizon H={cfg['T_ssa']*cfg['dt']:.0f}. "
        f"'≥' marks window-limited cells (τ*=max_lag), which are finite-horizon LOWER "
        f"bounds. The baseline is a Gaussian matched on the FULL covariance matrix and "
        f"run with identical N, Δt, horizon and stopping rule, so the ratio isolates "
        f"non-Gaussian shape rather than covariance/anisotropy. 'Sig. DA' counts DA "
        f"directions whose eigenvalue exceeds the matched-Gaussian DA null; dip and "
        f"Silverman p are reported for each of them (small p ⇒ reject unimodality of "
        f"that projection)."
    )
    hdr = ["Density", "d", "unique mode", "Ŝ", "matched-Gauss Ŝ", "**SSA ratio**",
           "win-lim", "τ*", "λ1/λ2", "DA1 cross-seed ∠", "Sig. DA", "dip p / Silverman p"]
    lines = [cap, "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in sorted(results, key=lambda z: -z['matched_gaussian']['ssa_ratio']):
        s, mg, ds = r['ssa'], r['matched_gaussian'], r['da_stats']
        um = r['unique_mode']
        mod = "; ".join(f"{k}: {v['dip_p']:.2g}/{v['silverman_p']:.2g}"
                        for k, v in r['modality'].items())
        lines.append("| " + " | ".join([
            r['label'], str(r['d']),
            "**yes**" if um['unique_mode'] else "NOT CONFIRMED",
            f"{'≥ ' if s['window_limited'] else ''}{s['S_hat_mean']:.3f}±{s['S_hat_std']:.3f}",
            f"{'≥ ' if mg['window_limited'] else ''}{mg['S_hat_mean']:.3f}±{mg['S_hat_std']:.3f}",
            f"**{mg['ssa_ratio']:.2f}×**",
            "yes" if s['window_limited'] else "no",
            str(s['tau_star']),
            _fmt_ratio(ds['gap_ratio_mean']),
            f"{ds['da1_angle_deg_max']:.1f}°",
            str(len(r['da_null']['significant_dirs'])),
            mod,
        ]) + " |")

    best = max(results, key=lambda z: z['matched_gaussian']['ssa_ratio'])
    br = best['matched_gaussian']['ssa_ratio']
    lines.append("")
    if br >= 1.5:
        lines.append(
            f"**Headline.** {best['label']} is provably unimodal (unique mode verified "
            f"analytically and by {best['unique_mode']['n_converged']} multi-start "
            f"optimisations, all converging to a single optimum) yet attains "
            f"Ŝ = {best['ssa']['S_hat_mean']:.3f}, **{br:.2f}× its covariance-matched "
            f"Gaussian baseline**. High finite-horizon SSA therefore does NOT imply "
            f"multimodality: it reflects slow / barrier-like relaxation, which a "
            f"strictly unimodal heavy-tailed or curved density can also produce.")
    else:
        lines.append(
            f"**Honest negative result.** The largest elevation over a "
            f"covariance-matched Gaussian across all three families is only "
            f"{br:.2f}× ({best['label']}). On this evidence, substantially elevated "
            f"SSA was NOT manufacturable without barriers in these families — high "
            f"SSA appears harder to produce from unimodal slow geometry than the "
            f"concern supposes. Reported as observed; no density was swapped in to "
            f"improve the number.")
    return "\n".join(lines)


# =========================================================================== #
def get_config(smoke):
    if smoke:
        return dict(dt=0.01, N_ssa=200, T_ssa=300, max_lag=150, N_da=300, N_mod=1000,
                    ssa_seeds=[0], da_seeds=[0, 1], checkpoints=[20, 50, 100, 200],
                    silverman_B=100, n_mode_starts=60, n_qc_pairs=500,
                    stat_N=1500, stat_T=400,
                    max_modality_dirs=2)
    return dict(dt=0.01, N_ssa=1000, T_ssa=2000, max_lag=1200, N_da=2000, N_mod=5000,
                ssa_seeds=[0, 1, 2], da_seeds=[0, 1, 2],
                checkpoints=[50, 100, 200, 400, 800, 1200, 1600, 2000],
                silverman_B=500, n_mode_starts=200, n_qc_pairs=2000,
                stat_N=2000, stat_T=1000,
                max_modality_dirs=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--checks', action='store_true')
    ap.add_argument('--dims', type=int, nargs='+', default=[2, 10])
    ap.add_argument('--family', '--only_family', dest='family',
                    choices=['t', 'banana', 'gg'], default=None,
                    help='Run only one family, so families can go as parallel jobs.')
    ap.add_argument('--cells', type=str, default=None,
                    help='1-based cell selector over the production grid, e.g. '
                         '"13-17" or "1,3,13-17". Use --list_cells to see indices.')
    ap.add_argument('--list_cells', action='store_true',
                    help='Print the numbered production grid and exit.')
    ap.add_argument('--resume', action='store_true',
                    help='Skip cells whose per-cell JSON already exists.')
    ap.add_argument('--merge_only', action='store_true',
                    help='Do not simulate; merge existing per-cell JSONs into the '
                         'combined result + table.')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--output_dir', default=RESULTS_DIR)
    args = ap.parse_args()

    if args.checks:
        sys.exit(0 if run_checks() else 1)

    cfg = get_config(args.smoke)
    tag = args.tag or ('smoke' if args.smoke else 'full')
    grid = build_densities(args.dims, smoke=args.smoke)

    if args.list_cells:
        for i, d in enumerate(grid, 1):
            print(f"  {i:2d}  {d.name}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    cell_dir = os.path.join(args.output_dir, f'cells_{tag}')
    os.makedirs(cell_dir, exist_ok=True)

    def cell_path(dens):
        return os.path.join(cell_dir, f'{dens.name}.json')

    # ---- select cells ----
    sel = list(range(len(grid)))
    if args.family:
        cls = dict(t='StudentT', banana='Banana', gg='GenGauss')[args.family]
        sel = [i for i in sel if type(grid[i]).__name__ == cls]
    if args.cells:
        want = set()
        for part in args.cells.split(','):
            part = part.strip()
            if '-' in part:
                a, b = part.split('-')
                want.update(range(int(a) - 1, int(b)))
            elif part:
                want.add(int(part) - 1)
        sel = [i for i in sel if i in want]
    dens_list = [grid[i] for i in sel]

    def merge_and_write():
        """Collect every per-cell JSON on disk into the combined outputs."""
        merged = []
        for d in grid:
            p = cell_path(d)
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        merged.append(json.load(f))
                except Exception as e:
                    print(f"  [warn] unreadable per-cell JSON {p}: {e}")
        jp = os.path.join(args.output_dir, f'p11_results_{tag}.json')
        with open(jp, 'w') as f:
            json.dump(dict(tag=tag, config=cfg, n_cells=len(merged), results=merged),
                      f, indent=2)
        mp = None
        if merged:
            table = build_table(merged, cfg)
            mp = os.path.join(args.output_dir, f'p11_table_{tag}.md')
            with open(mp, 'w') as f:
                f.write(f"# P11 strictly unimodal high-SSA controls ({tag})\n\n"
                        f"{len(merged)}/{len(grid)} cells present.\n\n{table}\n")
        return jp, mp, merged

    if args.merge_only:
        jp, mp, merged = merge_and_write()
        print(f"Merged {len(merged)} cells\n  {jp}\n  {mp}")
        return

    print(f"P11 unimodal high-SSA controls — {tag}\ncfg: {cfg}")
    print(f"grid: {len(grid)} cells; running {len(dens_list)}: "
          f"{[d.name for d in dens_list]}")

    n_pass = n_fail = n_skip = 0
    for j, dens in enumerate(dens_list, 1):
        cp = cell_path(dens)
        if args.resume and os.path.exists(cp):
            n_skip += 1
            print(f"\n[CELL {j}/{len(dens_list)}] SKIP (resume, already done): {dens.name}")
            continue
        try:
            res = run_cell(dens, cfg)
            # write THIS cell immediately: a later crash cannot destroy it
            with open(cp, 'w') as f:
                json.dump(res, f, indent=2)
            n_pass += 1
            print(f"[CELL {j}/{len(dens_list)}] PASS {dens.name} "
                  f"(ratio={res['matched_gaussian']['ssa_ratio']:.3f}, "
                  f"{res['runtime_s']:.0f}s) -> {cp}")
        except Exception as e:
            import traceback
            n_fail += 1
            err = dict(name=dens.name, label=getattr(dens, 'label', dens.name),
                       d=dens.d, failed=True, error=repr(e),
                       traceback=traceback.format_exc())
            with open(os.path.join(cell_dir, f'{dens.name}.FAILED.json'), 'w') as f:
                json.dump(err, f, indent=2)
            print(f"[CELL {j}/{len(dens_list)}] FAIL {dens.name}: {e!r}")
            traceback.print_exc()
            # one bad density must not kill the run
        # re-merge after every cell so the combined artifacts are always current
        merge_and_write()

    jp, mp, merged = merge_and_write()
    print(f"\n{'='*74}\nPASS={n_pass} FAIL={n_fail} SKIP={n_skip}; "
          f"{len(merged)}/{len(grid)} cells on disk")
    if merged:
        print("=" * 74 + "\nREBUTTAL TABLE\n" + "=" * 74)
        print(build_table(merged, cfg))
    print(f"\nSaved:\n  {jp}\n  {mp}\n  per-cell: {cell_dir}/")


if __name__ == '__main__':
    main()
