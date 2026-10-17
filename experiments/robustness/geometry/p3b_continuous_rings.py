#!/usr/bin/env python3
"""
P3b (REDESIGN): EXACT continuous concentric rings.

Replaces the inconclusive finite-GMM ring approximation (which created extra
ANGULAR modes and was therefore ruled INCONCLUSIVE; that result is retained
only as a failed calibration and must NOT be used as rebuttal evidence).

CONSTRUCTION (d = 2, EXACTLY rotationally symmetric).  With q = ||x||^2,

    p(x) = (1/pi) * sum_i w_i g_i(q),
    g_i(q) = Z_i^{-1} exp[ -(q - m_i)^2 / (2 s_q^2) ],   q >= 0,
    Z_i = int_0^inf exp[-(q-m_i)^2/(2 s_q^2)] dq = s_q sqrt(2 pi) Phi(m_i / s_q),

with w_1 = w_2 = 1/2 and m_1 < m_2.  Both shells are mean-zero (E[X | shell i]
= 0 by rotational symmetry), so their separation is PURELY RADIAL and NO signed
linear direction points from one shell to the other.

EXACT SAMPLING.  In 2-D, dx = pi dq after integrating the uniform angle, so the
law of q is EXACTLY the (truncated) 1-D mixture sum_i w_i g_i(q).  Therefore:
sample the shell label, sample q from its truncated Gaussian, sample
theta ~ Unif[0, 2pi), and return x = sqrt(q) (cos theta, sin theta).

EXACT SCORE.
    grad_x log p(x) = -(2 x / s_q^2) sum_i r_i(q) (q - m_i),
    r_i(q) = w_i g_i(q) / sum_j w_j g_j(q).
(The 1/pi and the Z_i are constants in x except through r_i, which uses the
NORMALIZED g_i; both are handled exactly below.)

WHAT THIS TESTS.  The slow shell-switching eigenfunction of the generator is
rotationally invariant, psi_2(x) = h(||x||), hence

    E[X psi_2(X)] = 0   EXACTLY, by symmetry.

So the shell-switching mode should be VISIBLE to the radial observable
q = ||x||^2 but ABSENT from the population linear-coordinate autocovariance
that ambient-space SSA and DA use.  Rotational symmetry also forces
C(tau) = c(tau) I, so any ambient DA direction is non-unique.

PREDECLARED BARRIER GRID (all points reported; no favourable-case selection):
    m_1 = 2.25,  m_2 = 9,  (m_2 - m_1)/s_q in {3, 4, 5}.

READOUTS AND CONTROLS.  The SAME canonical trajectories are pushed through two
readouts: (1) the ambient features x = (x_1, x_2); (2) the radial feature
q = ||x||^2 (standardization is irrelevant: the trace-normalized rho is scale
invariant).  Exact single-INNER-shell and single-OUTER-shell controls and a
covariance-matched isotropic Gaussian are run under IDENTICAL N, dt, physical
horizon, lag grid and stopping convention.  The single-shell controls are what
separate slow ANGULAR motion on a ring from genuine radial shell-switching
metastability.

ESTIMATOR CONVENTION (identical to the rest of the rebuttal): threshold 1/N,
stopping lag tau* = the LARGEST lag with rho^2 > 1/N, integrate THROUGH tau*,
lag 0 excluded, S_hat = dt * sum_{tau=1}^{tau*} rho^2(tau).  Cells with
tau* == max_lag are WINDOW-LIMITED and are finite-horizon LOWER BOUNDS, never
resolved population values.

Usage:
    python p3b_continuous_rings.py --checks     # verification only, no simulation
    python p3b_continuous_rings.py --smoke      # tiny CPU smoke test
    python p3b_continuous_rings.py --tag full   # production run
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

from src.simulation.langevin import LangevinSimulatorD
from src.simulation.generic_langevin import GenericLangevinSimulator
from src.metrics.autocorrelation import (
    compute_vector_autocorrelation,
    compute_stationary_moments,
    aggregate_ssa,
)
from src.metrics.da_directions import da_convergence_profile

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'sdxl'))
from silverman_test import silverman_test

import diptest
from scipy import stats as sstats
from scipy.stats import norm as _norm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'p3b_rings')

# Predeclared barrier grid (do NOT select one favourable case).
M1_DEFAULT = 2.25
M2_DEFAULT = 9.0
BARRIER_GRID_DEFAULT = (3.0, 4.0, 5.0)


# =========================================================================== #
# 1. The exact density
# =========================================================================== #
class RingShellDensity:
    """Exact rotationally symmetric shell density in d=2, parameterised in q.

    p(x) = (1/pi) sum_i w_i g_i(q),  q = ||x||^2,
    g_i = truncated-to-[0,inf) Gaussian(m_i, s_q^2) density in q.

    The law of q is EXACTLY sum_i w_i g_i(q) (because dx = pi dq), which makes
    both sampling and every radial moment analytic.
    """

    def __init__(self, m, w, s_q):
        self.m = np.asarray(m, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.w = self.w / self.w.sum()
        self.s_q = float(s_q)
        self.k = len(self.m)
        # alpha_i = (0 - m_i)/s : lower-truncation point in standard units
        self.alpha = (0.0 - self.m) / self.s_q
        # Z_i = int_0^inf exp[-(q-m_i)^2/(2 s^2)] dq = s sqrt(2pi) Phi(m_i/s)
        self.log_Z = (np.log(self.s_q) + 0.5 * np.log(2.0 * np.pi)
                      + _norm.logcdf(-self.alpha))
        self.d = 2

    # ------------------------------------------------------------------ #
    # density in q (this IS the exact law of q = ||x||^2)
    # ------------------------------------------------------------------ #
    def log_comp_q(self, q):
        """(k, ...) log g_i(q), zero-truncated and normalized."""
        q = np.asarray(q, dtype=float)
        out = (-0.5 * ((q[None, ...] - self.m[:, None]) / self.s_q) ** 2
               - self.log_Z[:, None])
        out = np.where(q[None, ...] >= 0.0, out, -np.inf)
        return out

    def pdf_q(self, q):
        """Exact density of q = ||x||^2."""
        lg = self.log_comp_q(np.atleast_1d(q))
        mx = np.max(lg, axis=0)
        val = np.exp(mx) * np.sum(self.w[:, None] * np.exp(lg - mx[None, :]), axis=0)
        return val

    def pdf_r(self, r):
        """Exact density of r = ||x|| (Jacobian dq = 2r dr)."""
        r = np.atleast_1d(np.asarray(r, dtype=float))
        return 2.0 * r * self.pdf_q(r ** 2)

    def logpdf_x(self, x):
        """Exact log p(x) up to the additive constant -log(pi)."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        q = (x ** 2).sum(axis=1)
        return np.log(self.pdf_q(q)) - np.log(np.pi)

    # ------------------------------------------------------------------ #
    # exact score
    # ------------------------------------------------------------------ #
    def score_np(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        q = (x ** 2).sum(axis=1)
        lg = self.log_comp_q(q) + np.log(self.w)[:, None]      # (k, n)
        lg = lg - lg.max(axis=0, keepdims=True)
        r = np.exp(lg)
        r = r / r.sum(axis=0, keepdims=True)                   # responsibilities
        coef = (r * (q[None, :] - self.m[:, None])).sum(axis=0)  # (n,)
        return -(2.0 / self.s_q ** 2) * x * coef[:, None]

    def score_torch_fn(self):
        """Batched torch score, numerically stable via log-softmax."""
        m = torch.tensor(self.m, dtype=torch.float32)
        logw_minus_logZ = torch.tensor(np.log(self.w) - self.log_Z,
                                       dtype=torch.float32)
        s2 = float(self.s_q ** 2)

        def score(x):
            q = (x ** 2).sum(dim=-1, keepdim=True)             # (n,1)
            logits = (-0.5 * (q - m.unsqueeze(0)) ** 2 / s2
                      + logw_minus_logZ.unsqueeze(0))          # (n,k)
            r = torch.softmax(logits, dim=-1)
            coef = (r * (q - m.unsqueeze(0))).sum(dim=-1, keepdim=True)  # (n,1)
            return -(2.0 / s2) * x * coef
        return score

    # ------------------------------------------------------------------ #
    # exact sampling
    # ------------------------------------------------------------------ #
    def sample_q(self, n, rng):
        lab = rng.choice(self.k, size=n, p=self.w)
        q = np.empty(n, dtype=float)
        for i in range(self.k):
            sel = lab == i
            ni = int(sel.sum())
            if ni:
                q[sel] = sstats.truncnorm.rvs(
                    self.alpha[i], np.inf, loc=self.m[i], scale=self.s_q,
                    size=ni, random_state=rng)
        return q, lab

    def sample(self, n, rng):
        q, lab = self.sample_q(n, rng)
        th = rng.uniform(0.0, 2.0 * np.pi, size=n)
        r = np.sqrt(q)
        x = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
        return x, q, lab

    def cdf_q(self, q):
        q = np.atleast_1d(np.asarray(q, dtype=float))
        out = np.zeros_like(q)
        for i in range(self.k):
            out += self.w[i] * sstats.truncnorm.cdf(
                q, self.alpha[i], np.inf, loc=self.m[i], scale=self.s_q)
        return out

    # ------------------------------------------------------------------ #
    # exact moments
    # ------------------------------------------------------------------ #
    def moments(self):
        """Exact E[q], Var[q] and Cov_f = (E[q]/2) I  (2-D, rotationally symmetric)."""
        Eq_i, Vq_i = [], []
        for i in range(self.k):
            a = self.alpha[i]
            lam = _norm.pdf(a) / _norm.sf(a)          # inverse Mills ratio
            Eq_i.append(self.m[i] + self.s_q * lam)
            Vq_i.append(self.s_q ** 2 * (1.0 + a * lam - lam ** 2))
        Eq_i = np.array(Eq_i)
        Vq_i = np.array(Vq_i)
        Eq = float((self.w * Eq_i).sum())
        Eq2 = float((self.w * (Vq_i + Eq_i ** 2)).sum())
        Vq = Eq2 - Eq ** 2
        # E[x1^2] = E[q cos^2 th] = E[q]/2  => Cov_f = (E[q]/2) I
        cov_f = (Eq / 2.0) * np.eye(2)
        return dict(E_q=Eq, Var_q=Vq, E_q_per_comp=Eq_i.tolist(),
                    cov_f=cov_f, sigma_f_sq=float(np.trace(cov_f) / 2.0))

    # ------------------------------------------------------------------ #
    # radial mode structure
    # ------------------------------------------------------------------ #
    def mode_structure(self, q_max=None, n_grid=200001):
        """Interior local maxima/minima of the exact q-density on q >= 0."""
        if q_max is None:
            q_max = float(self.m.max() + 8.0 * self.s_q)
        q = np.linspace(0.0, q_max, n_grid)
        f = self.pdf_q(q)
        up = (f[1:-1] > f[:-2]) & (f[1:-1] > f[2:])
        dn = (f[1:-1] < f[:-2]) & (f[1:-1] < f[2:])
        maxima = q[1:-1][up]
        minima = q[1:-1][dn]
        # boundary behaviour at q = 0 (a truncation-induced edge peak would
        # break the "exactly two maxima, one minimum" requirement)
        boundary_max = bool(f[0] > f[1])
        return dict(q_grid_max=float(q_max), maxima_q=maxima.tolist(),
                    minima_q=minima.tolist(), n_maxima=int(len(maxima)),
                    n_minima=int(len(minima)), boundary_max_at_0=boundary_max,
                    f_at_maxima=self.pdf_q(maxima).tolist() if len(maxima) else [],
                    f_at_minima=self.pdf_q(minima).tolist() if len(minima) else [])

    def marginal_x1(self, x1, n_th=20001):
        """EXACT 1-D marginal p(x_1) = int p(x_1, x_2) dx_2, by quadrature.

        Every linear projection of this density has the SAME law (rotational
        symmetry), so this is simultaneously the law of x . u for every unit u,
        including DA1 and DA2.  Its mode structure is a RING-GEOMETRY artefact
        at +/- r and does NOT separate the shells -- that is what the
        classifier / coupling tests establish.
        """
        x1 = np.atleast_1d(np.asarray(x1, float))
        # p(x1) = (1/pi) int_{-inf}^{inf} pq(x1^2 + x2^2) dx2 ; use a broad grid
        x2max = np.sqrt(self.m.max() + 8.0 * self.s_q) + 1.0
        x2 = np.linspace(-x2max, x2max, n_th)
        Q = x1[:, None] ** 2 + x2[None, :] ** 2
        vals = self.pdf_q(Q.ravel()).reshape(Q.shape) / np.pi
        # numpy compat: `trapezoid` is numpy>=2.0 only; the cluster env is 1.26.4,
        # where the same function is `trapz`. Same pattern as src/metrics/autocorrelation.py.
        _trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
        return _trapz(vals, x2, axis=1)

    def mode_structure_marginal(self, n_grid=2001):
        """Interior local maxima/minima of the EXACT linear marginal p(x_1)."""
        xm = np.sqrt(self.m.max() + 6.0 * self.s_q)
        x = np.linspace(-xm, xm, n_grid)
        f = self.marginal_x1(x)
        up = (f[1:-1] > f[:-2]) & (f[1:-1] > f[2:])
        dn = (f[1:-1] < f[:-2]) & (f[1:-1] < f[2:])
        return dict(maxima_x1=x[1:-1][up].tolist(), minima_x1=x[1:-1][dn].tolist(),
                    n_maxima=int(up.sum()), n_minima=int(dn.sum()),
                    note='law of x.u for EVERY unit direction u, by rotational '
                         'symmetry; modes sit at +/- radii and are symmetric '
                         'about 0, so they do NOT separate inner from outer')

    def mode_structure_r(self, n_grid=200001):
        """Same, for the density of r = ||x|| (includes the 2r Jacobian)."""
        r_max = float(np.sqrt(self.m.max() + 8.0 * self.s_q))
        r = np.linspace(1e-9, r_max, n_grid)
        f = self.pdf_r(r)
        up = (f[1:-1] > f[:-2]) & (f[1:-1] > f[2:])
        dn = (f[1:-1] < f[:-2]) & (f[1:-1] < f[2:])
        return dict(maxima_r=r[1:-1][up].tolist(), minima_r=r[1:-1][dn].tolist(),
                    n_maxima=int(up.sum()), n_minima=int(dn.sum()))

    def barrier_depth(self):
        """log-density drop from the shallower maximum to the intervening minimum."""
        ms = self.mode_structure()
        if ms['n_maxima'] != 2 or ms['n_minima'] != 1:
            return None
        fmax = min(ms['f_at_maxima'])
        fmin = ms['f_at_minima'][0]
        return dict(q_min=float(ms['minima_q'][0]),
                    q_max_inner=float(ms['maxima_q'][0]),
                    q_max_outer=float(ms['maxima_q'][1]),
                    log_depth_nats=float(np.log(fmax / fmin)),
                    density_ratio=float(fmax / fmin))


def two_shell(m1, m2, ratio):
    return RingShellDensity([m1, m2], [0.5, 0.5], (m2 - m1) / ratio)


def one_shell(m, s_q):
    return RingShellDensity([m], [1.0], s_q)


# =========================================================================== #
# 2. SSA drivers (shared estimator convention)
# =========================================================================== #
def _stopping_lag(rho_sq, floor):
    """tau* = LARGEST lag with rho^2 > floor; returns the number of leading lags
    to keep so that rho_sq[:keep] integrates THROUGH tau*.  NOT first-crossing."""
    above = np.where(rho_sq > floor)[0]
    if len(above) == 0:
        return 0
    return int(above[-1]) + 1


def ssa_feature(feat, mu_feat, dt, max_lag, N):
    """Finite-horizon trace-normalized SSA for an arbitrary feature trajectory.

    feat:     (N, T+1, p) feature trajectories.
    mu_feat:  (p,) EXACT stationary mean of the feature (analytic, not empirical).

    Reuses compute_vector_autocorrelation(..., normalization='trace').  In the
    'trace' branch the GMM arguments enter ONLY through the stationary mean
    mu_f, so a single-component dummy carrying the exact analytic mean gives
    exactly the intended global-mean centering; the dummy covariance is unused.
    """
    p = feat.shape[-1]
    dummy_means = torch.tensor(np.asarray(mu_feat, float).reshape(1, p),
                               dtype=feat.dtype)
    dummy_covs = torch.eye(p, dtype=feat.dtype).unsqueeze(0)
    dummy_w = torch.ones(1, dtype=feat.dtype)
    rho, _, _ = compute_vector_autocorrelation(
        feat, max_lag, dummy_means, dummy_covs, dummy_w,
        demean=True, normalization='trace', method='fft')
    rho_sq = (rho[1:] ** 2).double().numpy()
    floor = 1.0 / N
    keep = _stopping_lag(rho_sq, floor)
    S_hat = aggregate_ssa(rho_sq[:keep], dt=dt)
    S_full = aggregate_ssa(rho_sq, dt=dt)
    floor_1Nd = 1.0 / (N * p)
    keep_1Nd = _stopping_lag(rho_sq, floor_1Nd)
    return dict(S_hat=float(S_hat), S_full=float(S_full), tau_star=int(keep),
                floor=float(floor), floor_name='1/N',
                window_limited=bool(keep >= len(rho_sq)),
                max_lag=int(len(rho_sq)),
                S_hat_1Nd=float(aggregate_ssa(rho_sq[:keep_1Nd], dt=dt)),
                tau_star_1Nd=int(keep_1Nd), floor_1Nd=float(floor_1Nd),
                rho_sq=rho_sq)


def make_simulator(dens, dt, n_pool, seed):
    """GenericLangevinSimulator with EXACT moments (mu_f = 0, sigma_f^2 = E[q]/2)."""
    rng = np.random.default_rng(seed)
    pool, _, _ = dens.sample(n_pool, rng)
    mom = dens.moments()
    sim = GenericLangevinSimulator(
        dens.score_torch_fn(),
        torch.tensor(pool, dtype=torch.float32),
        dt=dt, mu_f=torch.zeros(2), var_f=mom['sigma_f_sq'])
    return sim, mom


def simulate_density(dens, cfg, seed):
    """Exact-initial-condition trajectories: x0 are EXACT iid draws from p."""
    rng = np.random.default_rng(100000 + seed)
    x0_np, q0, lab0 = dens.sample(cfg['N_ssa'], rng)
    sim, mom = make_simulator(dens, cfg['dt'], cfg['n_pool'], seed)
    torch.manual_seed(seed)
    traj = sim.simulate(torch.tensor(x0_np, dtype=torch.float32), cfg['T_ssa'])
    return traj, mom, lab0


def readouts(traj, mom, cfg):
    """Ambient x-readout and radial q-readout from the SAME trajectories."""
    N = cfg['N_ssa']
    amb = ssa_feature(traj, np.zeros(2), cfg['dt'], cfg['max_lag'], N)
    q = (traj ** 2).sum(dim=-1, keepdim=True)          # (N, T+1, 1)
    rad = ssa_feature(q, np.array([mom['E_q']]), cfg['dt'], cfg['max_lag'], N)
    return amb, rad, q


def matched_gaussian_density_params(mom):
    """Isotropic Gaussian matched EXACTLY to Cov_f = (E[q]/2) I."""
    sigma2 = mom['sigma_f_sq']
    means = torch.zeros(1, 2)
    covs = (sigma2 * torch.eye(2)).unsqueeze(0)
    weights = torch.ones(1)
    return means, covs, weights, sigma2


def run_matched_gaussian(mom, cfg, seed):
    """Matched isotropic Gaussian under IDENTICAL N, dt, horizon, lag grid, rule.

    Its diffusion coefficient sigma_f^2 = tr(Cov_f)/d equals that of the
    two-shell density by construction, so the time scale is identical too."""
    means, covs, weights, sigma2 = matched_gaussian_density_params(mom)
    torch.manual_seed(20000 + seed)
    sim = LangevinSimulatorD(means, covs, weights, dt=cfg['dt'])
    x0 = sim.sample_initial(cfg['N_ssa'])
    traj = sim.simulate(x0, cfg['T_ssa'])
    amb = ssa_feature(traj, np.zeros(2), cfg['dt'], cfg['max_lag'], cfg['N_ssa'])
    q = (traj ** 2).sum(dim=-1, keepdim=True)
    # E[q] = 2 sigma^2 for an isotropic 2-D Gaussian
    rad = ssa_feature(q, np.array([2.0 * sigma2]), cfg['dt'], cfg['max_lag'],
                      cfg['N_ssa'])
    return amb, rad, sigma2, traj


# =========================================================================== #
# 3. Shell-transition / residence statistics
# =========================================================================== #
def transition_stats(q_traj, q_min, q_in, q_out, dt):
    """Radial shell-transition counts and residence / first-passage summaries.

    q_traj: (N, T+1) radial feature.
    Raw crossings use the density minimum q_min.  COMMITTED transitions use
    hysteresis: commit to 'inner' on q <= q_in (the inner mode location) and to
    'outer' on q >= q_out (the outer mode location); this removes barrier-top
    recrossing noise, which otherwise inflates counts arbitrarily as dt -> 0.
    """
    q = np.asarray(q_traj, dtype=float)
    N, Tp1 = q.shape
    T_phys = (Tp1 - 1) * dt

    raw = int(np.sum(np.diff((q > q_min).astype(np.int8), axis=1) != 0))

    committed_counts = np.zeros(N, dtype=int)
    res_inner, res_outer, first_passage = [], [], []
    n_censored = 0
    for n in range(N):
        qn = q[n]
        # keep only the time points that COMMIT to a shell, then collapse runs
        st = np.full(Tp1, -1, dtype=np.int8)
        st[qn <= q_in] = 0
        st[qn >= q_out] = 1
        idx = np.flatnonzero(st >= 0)
        if len(idx) == 0:
            n_censored += 1
            continue
        s = st[idx]
        chg = np.flatnonzero(s[1:] != s[:-1])       # positions in `idx` before a switch
        committed_counts[n] = len(chg)
        if len(chg):
            t_switch = idx[chg + 1]                  # step index of each switch
            t_enter = np.concatenate([[idx[0]], t_switch[:-1]])
            durs = (t_switch - t_enter) * dt
            from_state = s[chg]
            res_inner.extend(durs[from_state == 0].tolist())
            res_outer.extend(durs[from_state == 1].tolist())
            first_passage.append(float(t_switch[0] * dt))
        else:
            n_censored += 1

    tot = int(committed_counts.sum())
    rate = tot / (N * T_phys) if T_phys > 0 else float('nan')

    def _summ(v):
        v = np.asarray(v, float)
        if len(v) == 0:
            return dict(n=0, mean=None, median=None, p10=None, p90=None)
        return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)),
                    p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)))

    return dict(
        thresholds=dict(q_min=float(q_min), q_commit_inner=float(q_in),
                        q_commit_outer=float(q_out)),
        horizon_phys=float(T_phys), n_traj=int(N),
        raw_crossings_of_q_min=raw,
        raw_crossings_per_traj=float(raw / N),
        committed_transitions_total=tot,
        committed_transitions_per_traj=float(tot / N),
        committed_transition_rate_per_unit_time=float(rate),
        implied_mean_dwell_time=(float(1.0 / rate) if rate > 0 else None),
        frac_traj_with_ge1_transition=float(np.mean(committed_counts >= 1)),
        residence_inner=_summ(res_inner), residence_outer=_summ(res_outer),
        first_passage_to_other_shell=_summ(first_passage),
        first_passage_censored_frac=float(n_censored / N),
        note=('residence times exclude the right-censored final sojourn; '
              'first-passage is censored for trajectories that never commit '
              'to the other shell within the horizon'))


# =========================================================================== #
# 4. Linear-observability tests (the core of the P3b claim)
# =========================================================================== #
def _auc(score, label):
    """Mann-Whitney AUC of a 1-D score for a binary label."""
    score = np.asarray(score, float)
    label = np.asarray(label).astype(int)
    n1 = int(label.sum())
    n0 = len(label) - n1
    if n0 == 0 or n1 == 0:
        return float('nan')
    r = sstats.rankdata(score)
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def _bal_acc(pred, label):
    pred = np.asarray(pred).astype(int)
    label = np.asarray(label).astype(int)
    tpr = float(np.mean(pred[label == 1] == 1)) if (label == 1).any() else np.nan
    tnr = float(np.mean(pred[label == 0] == 0)) if (label == 0).any() else np.nan
    return 0.5 * (tpr + tnr)


def _lda_fit(X, y):
    """Fisher LDA: w = Sw^{-1}(mu1 - mu0), threshold at the midpoint."""
    X0, X1 = X[y == 0], X[y == 1]
    mu0, mu1 = X0.mean(0), X1.mean(0)
    Sw = np.cov(X0.T, bias=False) * (len(X0) - 1) + np.cov(X1.T, bias=False) * (len(X1) - 1)
    Sw = Sw / (len(X) - 2) + 1e-9 * np.eye(X.shape[1])
    w = np.linalg.solve(Sw, mu1 - mu0)
    b = -0.5 * w @ (mu0 + mu1)
    return w, float(b)


def _logreg_fit(X, y, l2=1e-3, n_iter=100):
    """Newton-IRLS logistic regression with intercept (no sklearn dependency)."""
    Xa = np.hstack([X, np.ones((len(X), 1))])
    p = Xa.shape[1]
    beta = np.zeros(p)
    for _ in range(n_iter):
        z = Xa @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        W = np.clip(mu * (1 - mu), 1e-8, None)
        g = Xa.T @ (y - mu) - l2 * beta
        H = (Xa * W[:, None]).T @ Xa + l2 * np.eye(p)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def _logreg_predict(beta, X):
    Xa = np.hstack([X, np.ones((len(X), 1))])
    return (Xa @ beta > 0).astype(int)


def shell_label_observability(X, q, label, da1, da2, n_dir=720, seed=0):
    """Everything that decides whether the AMBIENT LINEAR readout can see the shell.

    Returns coupling of the shell label to each SIGNED linear DA projection,
    the best AUC over ALL signed unit directions, linear classifiers on x
    (which must stay at chance by symmetry) versus a classifier on q, and a
    direct numerical check of E[X psi_2(X)] = 0.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]

    out = {}

    # --- coupling of the label to each SIGNED DA projection ---
    for nm, v in (('da1', da1), ('da2', da2)):
        pr = X @ np.asarray(v, float)
        rpb, ppb = sstats.pointbiserialr(label, pr)
        out[f'coupling_{nm}'] = dict(
            direction=list(map(float, np.asarray(v, float))),
            point_biserial_r=float(rpb), p_value=float(ppb),
            auc=_auc(pr, label),
            mean_proj_inner=float(pr[label == 0].mean()),
            mean_proj_outer=float(pr[label == 1].mean()),
            std_proj_inner=float(pr[label == 0].std()),
            std_proj_outer=float(pr[label == 1].std()),
            note='SIGNED projection (no absolute value): a shell-separating '
                 'linear direction would give |r| >> 0 and AUC far from 0.5')

    # --- best SIGNED linear direction over the whole circle, with a PERMUTATION
    #     null for the max-over-directions statistic (a max over 720 correlated
    #     directions is NOT calibrated by the single-direction chance SE) ---
    th = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
    U = np.stack([np.cos(th), np.sin(th)], axis=1)
    # The rank matrix is (n_sub, n_dir); cap n_sub so it stays a few hundred MB.
    # At n_sub = 50000 the single-direction chance SE is ~0.0016, far finer than
    # any effect that would matter.
    n_sub = min(n, 50000)
    sub = rng.choice(n, size=n_sub, replace=False) if n_sub < n else np.arange(n)
    Xs, lab_s = X[sub], label[sub]
    P = Xs @ U.T                                       # (n_sub, n_dir)
    R = np.apply_along_axis(sstats.rankdata, 0, P)     # (n_sub, n_dir) ranks
    n1 = int(lab_s.sum()); n0 = n_sub - n1

    def _aucs_from_ranks(Y):
        """Y: (n, B) 0/1 label matrix -> (n_dir, B) AUCs."""
        S = R.T @ Y                                    # rank sums of class 1
        return (S - n1 * (n1 + 1) / 2.0) / (n0 * n1)

    aucs = _aucs_from_ranks(lab_s.reshape(-1, 1).astype(float))[:, 0]
    j = int(np.argmax(np.abs(aucs - 0.5)))
    obs_stat = float(np.max(np.abs(aucs - 0.5)))
    B = 500
    Y = np.empty((n_sub, B))
    for b in range(B):
        Y[:, b] = rng.permutation(lab_s).astype(float)
    null_stat = np.max(np.abs(_aucs_from_ranks(Y) - 0.5), axis=0)
    p_perm = float((1 + np.sum(null_stat >= obs_stat)) / (B + 1))
    out['best_signed_direction'] = dict(
        n_directions=int(n_dir), n_samples_used=int(n_sub),
        best_angle_deg=float(np.degrees(th[j])),
        best_auc=float(aucs[j]),
        best_abs_auc_minus_half=obs_stat,
        mean_abs_auc_minus_half=float(np.mean(np.abs(aucs - 0.5))),
        single_direction_chance_se=float(np.sqrt(1.0 / (12.0 * n0) + 1.0 / (12.0 * n1))),
        perm_null_B=B,
        perm_null_mean=float(null_stat.mean()),
        perm_null_q95=float(np.quantile(null_stat, 0.95)),
        perm_null_max=float(null_stat.max()),
        perm_p_value=p_perm,
        note='exhaustive search over ALL signed unit directions (0.5 = chance), '
             'calibrated by a LABEL-PERMUTATION null of the SAME max-over-'
             'directions statistic; a large p means no signed linear direction '
             'separates the shells better than chance')

    # --- linear classifiers on x (must be at chance) vs a classifier on q ---
    y = label.astype(int)
    w, b = _lda_fit(X[tr], y[tr])
    lda_acc = float(_bal_acc((X[te] @ w + b > 0).astype(int), y[te]))
    # permutation null for the LDA balanced accuracy (refit on each permutation)
    n_perm = 200
    null_acc = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        wp, bp = _lda_fit(X[tr], yp[tr])
        null_acc[i] = _bal_acc((X[te] @ wp + bp > 0).astype(int), yp[te])
    out['classifier_linear_x_lda'] = dict(
        balanced_accuracy_test=lda_acc, w=list(map(float, w)), b=float(b),
        perm_null_B=n_perm, perm_null_mean=float(null_acc.mean()),
        perm_null_q95=float(np.quantile(null_acc, 0.95)),
        perm_p_value=float((1 + np.sum(null_acc >= lda_acc)) / (n_perm + 1)),
        note='LDA refit inside every permutation replicate, so the null '
             'absorbs any fitting optimism')
    beta = _logreg_fit(X[tr], y[tr])
    out['classifier_linear_x_logreg'] = dict(
        balanced_accuracy_test=float(_bal_acc(_logreg_predict(beta, X[te]), y[te])),
        beta=list(map(float, beta)))
    # --- BEST LINEAR HALF-PLANE, threshold swept (the strongest linear rule) ---
    # AUC = 0.5 says no signed linear score ORDERS the shells.  It does NOT say
    # every linear RULE is at chance: the two shells share a mean but differ in
    # SPREAD, so a far-out half-plane {x.u > c} is almost purely outer shell.
    # That rule uses no preferred direction (by symmetry every u is equivalent),
    # but it is above chance, so we measure exactly how far above.
    n_dir_hp, n_hp = 72, min(n, 20000)
    sub2 = rng.choice(n, size=n_hp, replace=False) if n_hp < n else np.arange(n)
    th2 = np.linspace(0, np.pi, n_dir_hp, endpoint=False)
    U2 = np.stack([np.cos(th2), np.sin(th2)], axis=1)
    P2 = X[sub2] @ U2.T                                # (n_hp, n_dir_hp)
    ORD = np.argsort(P2, axis=0).T                     # (n_dir_hp, n_hp)
    y_hp = label[sub2].astype(float)

    def _max_halfplane_bal(yv):
        Y2 = yv[ORD]                                   # (n_dir_hp, n_hp)
        c1 = np.cumsum(Y2, axis=1)
        c0 = np.cumsum(1.0 - Y2, axis=1)
        t1, t0 = c1[:, -1], c0[:, -1]
        bal = 0.5 * ((t1[:, None] - c1) / t1[:, None] + c0 / t0[:, None])
        bal = np.maximum(bal, 1.0 - bal)
        return float(bal.max()), int(np.argmax(bal.max(axis=1)))

    hp_stat, hp_dir = _max_halfplane_bal(y_hp)
    B2 = 200
    hp_null = np.empty(B2)
    for b in range(B2):
        hp_null[b] = _max_halfplane_bal(rng.permutation(y_hp))[0]
    out['classifier_linear_x_best_halfplane'] = dict(
        balanced_accuracy=hp_stat,
        n_directions=int(n_dir_hp), n_samples_used=int(n_hp),
        best_direction_deg=float(np.degrees(th2[hp_dir])),
        perm_null_B=B2, perm_null_mean=float(hp_null.mean()),
        perm_null_q95=float(np.quantile(hp_null, 0.95)),
        perm_p_value=float((1 + np.sum(hp_null >= hp_stat)) / (B2 + 1)),
        note='max over ALL directions AND all thresholds, in-sample (an upper '
             'bound on any linear rule).  ABOVE chance: the shells differ in '
             'SPREAD, so an extreme half-plane is almost purely outer shell. '
             'This exploits no preferred DIRECTION -- by rotational symmetry '
             'every u gives the identical statistic, so the maximising '
             'direction is arbitrary -- and it does not affect C(tau), which '
             'is annihilated by the exact symmetry E[X psi_2(X)] = 0.')

    qq = np.asarray(q, float).reshape(-1, 1)
    beta_q = _logreg_fit(qq[tr], y[tr])
    out['classifier_q_logreg'] = dict(
        balanced_accuracy_test=float(_bal_acc(_logreg_predict(beta_q, qq[te]), y[te])),
        beta=list(map(float, beta_q)))
    out['classifier_q_auc'] = _auc(qq[:, 0], label)
    out['chance_level'] = 0.5
    out['classifier_note'] = (
        'the SAME labels and the SAME points are used for all classifiers; only '
        'the feature map differs.  Both shells have mean zero and are isotropic, '
        'so NO signed linear score orders them (AUC = 0.5 exactly) and LDA is at '
        'chance.  They differ in the SECOND moment, which is the quadratic '
        'feature q -- and a thresholded half-plane is a partial, direction-free '
        'proxy for it, which is why best_halfplane is above 0.5.  The unthresholded '
        'logistic fit is reported as-is and is NOT a calibrated chance test: with '
        'a near-zero slope it predicts almost one class, and balanced accuracy is '
        'unstable in that regime.  Use AUC, LDA and best_halfplane instead.')

    # --- direct check that E[X psi_2(X)] = 0 with psi_2 taken as the centered
    #     shell indicator (a rotationally invariant function of ||x||) ---
    psi = label.astype(float) - label.mean()
    v = (X * psi[:, None]).mean(axis=0)
    se = (X * psi[:, None]).std(axis=0) / np.sqrt(n)
    scale = np.sqrt((X ** 2).mean())
    out['E_X_psi2'] = dict(
        vector=list(map(float, v)), se=list(map(float, se)),
        norm=float(np.linalg.norm(v)),
        norm_over_rms_scale=float(np.linalg.norm(v) / scale),
        z_scores=list(map(float, v / se)),
        n=int(n),
        note='psi_2 proxy = centered shell indicator, a function of ||x|| only; '
             'population value is EXACTLY 0 by rotational symmetry')
    return out


def modality_tests(v, silverman_B=500, seed=0):
    v = np.asarray(v, float)
    dip_stat, dip_p = diptest.diptest(v)
    sv = silverman_test(v, n_modes=2, B=silverman_B, seed=seed)
    return dict(dip_stat=float(dip_stat), dip_p=float(dip_p),
                silverman_hcrit=sv['h_crit'], silverman_p=sv['p_value'],
                silverman_nmodes=sv['n_modes_observed'])


# =========================================================================== #
# 5. DA / rotational degeneracy
# =========================================================================== #
def angle_deg(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    c = abs(float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def signed_angle_deg(v):
    v = np.asarray(v, float)
    a = np.degrees(np.arctan2(v[1], v[0])) % 180.0
    return float(a)


def da_across_seeds(sim_factory, cfg, seeds, gmm=None):
    """DA profile across seeds.

    sim_factory: seed -> simulator object (GenericLangevinSimulator path), or
    gmm: (means, covs, weights) to use the GMM path of da_convergence_profile
    (LangevinSimulatorD has no .mu_f attribute, so it cannot be passed as a
    generic simulator object).
    """
    per_seed = []
    for s in seeds:
        if gmm is not None:
            prof = da_convergence_profile(gmm[0], gmm[1], gmm[2],
                                          checkpoints=cfg['checkpoints'],
                                          n_trajectories=cfg['N_da'], dt=cfg['dt'],
                                          seed=s)
        else:
            prof = da_convergence_profile(simulator=sim_factory(s),
                                          checkpoints=cfg['checkpoints'],
                                          n_trajectories=cfg['N_da'], dt=cfg['dt'],
                                          seed=s)
        ev = np.asarray(prof['optimal_eigenvalues'], float)
        vec = np.asarray(prof['optimal_eigenvectors'], float)
        C = np.asarray(prof['optimal_C_tau'], float)
        lam1, lam2 = float(ev[0]), float(ev[1])
        per_seed.append(dict(
            seed=int(s), opt_lag=int(prof['checkpoints'][prof['optimal_idx']]),
            lam1=lam1, lam2=lam2,
            gap_ratio=(float(lam1 / lam2) if lam2 > 0 else None),
            split=float(abs(lam1 - lam2) / (abs(lam1) + abs(lam2) + 1e-300)),
            da1=vec[:, 0].tolist(), da2=vec[:, 1].tolist(),
            da1_orientation_deg=signed_angle_deg(vec[:, 0]),
            C_tau=C.tolist(),
            offdiag_over_diag=float(abs(C[0, 1]) / (0.5 * abs(C[0, 0] + C[1, 1]) + 1e-300)),
            diag_asym=float(abs(C[0, 0] - C[1, 1]) / (abs(C[0, 0]) + abs(C[1, 1]) + 1e-300)),
            gap_profile=np.asarray(prof['gap_profile'], float).tolist(),
            checkpoints=np.asarray(prof['checkpoints']).tolist(),
            leading_eigenvectors=np.asarray(prof['leading_eigenvectors']).tolist(),
            all_eigenvalues=np.asarray(prof['all_eigenvalues']).tolist(),
            # (n_cp, d, k) with k=d=2 here: the FULL eigenbasis at every
            # checkpoint, which lets C(tau) be reconstructed per checkpoint.
            top_k_eigenvectors=np.asarray(prof['top_k_eigenvectors']).tolist(),
        ))
    ref = np.asarray(per_seed[0]['da1'])
    ang = [angle_deg(ref, np.asarray(p['da1'])) for p in per_seed]
    ratios = [p['gap_ratio'] for p in per_seed if p['gap_ratio'] is not None]
    orients = [p['da1_orientation_deg'] for p in per_seed]
    # rotation of DA1 across NEARBY accepted lags within seed 0
    lev = np.asarray(per_seed[0]['leading_eigenvectors'])
    cps = per_seed[0]['checkpoints']
    ref0 = np.asarray(per_seed[0]['da1'])
    lag_rot = {int(c): angle_deg(ref0, lev[i]) for i, c in enumerate(cps)}
    stats = dict(
        gap_ratio_per_seed=[p['gap_ratio'] for p in per_seed],
        gap_ratio_mean=(float(np.mean(ratios)) if ratios else None),
        gap_ratio_max=(float(np.max(ratios)) if ratios else None),
        eig_split_per_seed=[p['split'] for p in per_seed],
        eig_split_mean=float(np.mean([p['split'] for p in per_seed])),
        da1_angle_vs_seed0_deg=ang,
        da1_angle_vs_seed0_deg_max=float(np.max(ang)),
        da1_orientation_deg_per_seed=orients,
        da1_orientation_spread_deg=float(np.max(orients) - np.min(orients)),
        da1_rotation_nearby_lags_deg=lag_rot,
        offdiag_over_diag_per_seed=[p['offdiag_over_diag'] for p in per_seed],
        diag_asym_per_seed=[p['diag_asym'] for p in per_seed],
        lam1_per_seed=[p['lam1'] for p in per_seed],
        lam2_per_seed=[p['lam2'] for p in per_seed],
        auto_lag_note=('the AUTO-selected lag is argmax_tau lambda1/lambda2, '
                       'which is a MAXIMUM over checkpoints and is therefore '
                       'upward biased under exact degeneracy, and it can land '
                       'on different lags for different systems.  Use the '
                       'FIXED-LAG profile below for any cross-system comparison.'),
        note=('C(tau) = c(tau) I exactly in the population, so lambda1/lambda2 '
              '-> 1, the DA basis is arbitrary, and any observed splitting is '
              'sampling noise.  Compare against the matched isotropic Gaussian, '
              'whose C(tau) is EXACTLY isotropic, run at the same N.'))

    # ---- FIXED-LAG profile: the fair cross-system comparison ----
    cps = per_seed[0]['checkpoints']
    EV = np.array([p['all_eigenvalues'] for p in per_seed])          # (S, n_cp, 2)
    LV = np.array([p['leading_eigenvectors'] for p in per_seed])     # (S, n_cp, 2)
    VK = np.array([p['top_k_eigenvectors'] for p in per_seed])       # (S, n_cp, d, k)
    prof = {}
    for i, c in enumerate(cps):
        l1 = EV[:, i, 0]
        l2 = EV[:, i, 1]
        ratio = np.where(l2 > 0, l1 / np.where(l2 > 0, l2, 1.0), np.nan)
        orients = [signed_angle_deg(LV[s, i]) for s in range(LV.shape[0])]
        ang = [angle_deg(LV[0, i], LV[s, i]) for s in range(LV.shape[0])]
        # reconstruct C(tau) = V diag(lambda) V^T per seed to report isotropy
        offd, asym = [], []
        for s in range(VK.shape[0]):
            V = VK[s, i]                                  # (d, k=d)
            C = V @ np.diag(EV[s, i]) @ V.T
            offd.append(abs(C[0, 1]) / (0.5 * abs(C[0, 0] + C[1, 1]) + 1e-300))
            asym.append(abs(C[0, 0] - C[1, 1]) / (abs(C[0, 0]) + abs(C[1, 1]) + 1e-300))
        prof[int(c)] = dict(
            lag_phys=float(c * cfg['dt']),
            lam1_mean=float(l1.mean()), lam2_mean=float(l2.mean()),
            gap_ratio_per_seed=[None if not np.isfinite(r) else float(r)
                                for r in ratio],
            gap_ratio_mean=(float(np.nanmean(ratio)) if np.any(np.isfinite(ratio))
                            else None),
            n_seeds_lam2_nonpositive=int(np.sum(l2 <= 0)),
            eig_split_mean=float(np.mean(np.abs(l1 - l2)
                                         / (np.abs(l1) + np.abs(l2) + 1e-300))),
            da1_orientation_deg=orients,
            da1_max_angle_vs_seed0_deg=float(np.max(ang)),
            offdiag_over_diag_mean=float(np.mean(offd)),
            diag_asym_mean=float(np.mean(asym)))
    stats['fixed_lag_profile'] = prof
    n_bad = int(np.sum([p['lam2'] <= 0 for p in per_seed]))
    stats['auto_lag_lam2_nonpositive_seeds'] = n_bad
    if n_bad:
        stats['auto_lag_warning'] = (
            f'lambda2 <= 0 at the auto-selected lag in {n_bad}/{len(per_seed)} '
            f'seeds.  Expected under EXACT degeneracy: da_convergence_profile '
            f'selects argmax lambda1/lambda2 and treats lambda2 <= 0 as an '
            f'infinite gap, so it lands on the noisiest checkpoint.  The '
            f'auto-lag lambda1/lambda2 and C(tau) isotropy numbers are '
            f'MEANINGLESS here; use the fixed-lag profile.')
    for p in per_seed:
        p.pop('leading_eigenvectors', None)
        p.pop('all_eigenvalues', None)
        p.pop('top_k_eigenvectors', None)
    return per_seed, stats


# =========================================================================== #
# 6. Verification checks (run BEFORE any simulation)
# =========================================================================== #
def run_checks(m1, m2, grid, dt=0.002, verbose=True):
    ok = True
    lines = []

    def say(s):
        lines.append(s)
        if verbose:
            print(s)

    say('=' * 78)
    say('P3b VERIFICATION CHECKS (all must pass before simulating)')
    say('=' * 78)

    checks = {}
    for ratio in grid:
        dens = two_shell(m1, m2, ratio)
        tag = f'(m2-m1)/s_q={ratio:g}  s_q={dens.s_q:.6f}'
        say(f'\n--- grid point {tag} ---')
        rec = {}

        # C1: score vs finite differences AND torch autograd
        rng = np.random.default_rng(7)
        Xc, _, _ = dens.sample(400, rng)
        Xc = np.vstack([Xc, np.zeros((1, 2)), np.array([[1e-3, -1e-3]])])
        h = 1e-5
        fd = np.zeros_like(Xc)
        for j in range(2):
            e = np.zeros(2); e[j] = h
            fd[:, j] = (dens.logpdf_x(Xc + e) - dens.logpdf_x(Xc - e)) / (2 * h)
        an = dens.score_np(Xc)
        err_fd = np.max(np.abs(an - fd) / (1.0 + np.abs(fd)))
        # torch autograd on the same points, using the torch score's own logits
        xt = torch.tensor(Xc, dtype=torch.float64, requires_grad=True)
        mt = torch.tensor(dens.m, dtype=torch.float64)
        lwz = torch.tensor(np.log(dens.w) - dens.log_Z, dtype=torch.float64)
        qt = (xt ** 2).sum(-1, keepdim=True)
        lp = torch.logsumexp(-0.5 * (qt - mt.unsqueeze(0)) ** 2 / dens.s_q ** 2
                             + lwz.unsqueeze(0), dim=-1).sum()
        lp.backward()
        ad = xt.grad.detach().numpy()
        err_ad = np.max(np.abs(an - ad) / (1.0 + np.abs(ad)))
        st = dens.score_torch_fn()(torch.tensor(Xc, dtype=torch.float32)).numpy()
        err_torch = np.max(np.abs(an - st) / (1.0 + np.abs(an)))
        c1 = bool(err_fd < 1e-4 and err_ad < 1e-8 and err_torch < 1e-4
                  and np.all(np.isfinite(an)))
        ok &= c1
        say(f'  C1 score: max rel err vs finite-diff = {err_fd:.3e}, '
            f'vs autograd = {err_ad:.3e}, torch-vs-numpy = {err_torch:.3e}, '
            f'all finite = {bool(np.all(np.isfinite(an)))}  -> {"PASS" if c1 else "FAIL"}')
        say(f'     score at origin = {an[-2].tolist()} (must be exactly 0 by symmetry)')
        rec['C1_score'] = dict(passed=c1, max_rel_err_fd=float(err_fd),
                               max_rel_err_autograd=float(err_ad),
                               max_rel_err_torch=float(err_torch),
                               score_at_origin=an[-2].tolist())

        # C2: EXACTLY two maxima and one intervening minimum in the radial density
        ms = dens.mode_structure()
        msr = dens.mode_structure_r()
        interleaved = (ms['n_maxima'] == 2 and ms['n_minima'] == 1
                       and ms['maxima_q'][0] < ms['minima_q'][0] < ms['maxima_q'][1])
        c2 = bool(interleaved and not ms['boundary_max_at_0'])
        ok &= c2
        bd = dens.barrier_depth()
        say(f'  C2 radial density in q: n_maxima={ms["n_maxima"]} at '
            f'{np.round(ms["maxima_q"],4).tolist()}, n_minima={ms["n_minima"]} at '
            f'{np.round(ms["minima_q"],4).tolist()}, boundary max at q=0: '
            f'{ms["boundary_max_at_0"]}  -> {"PASS" if c2 else "FAIL"}')
        say(f'     barrier depth = {bd["log_depth_nats"]:.4f} nats '
            f'(density ratio {bd["density_ratio"]:.3f})')
        say(f'     density of r=|x| (with 2r Jacobian): n_maxima={msr["n_maxima"]} at '
            f'{np.round(msr["maxima_r"],4).tolist()}, n_minima={msr["n_minima"]}')
        # INFORMATIONAL (not pass/fail): the exact linear marginal.  Ring
        # geometry can make EVERY linear projection multimodal at +/- r; those
        # modes are symmetric about 0 and do NOT separate the shells.
        msm = dens.mode_structure_marginal()
        say(f'     [informational] EXACT linear marginal p(x.u), same for every '
            f'unit u: n_maxima={msm["n_maxima"]} at '
            f'{np.round(msm["maxima_x1"],3).tolist()}, n_minima={msm["n_minima"]} '
            f'at {np.round(msm["minima_x1"],3).tolist()}')
        rec['C2_modes'] = dict(passed=c2, q=ms, r=msr, marginal=msm, barrier=bd)

        # C3: exact sampler agrees with the exact law of q
        rng = np.random.default_rng(11)
        xs, qs, labs = dens.sample(200000, rng)
        ks = sstats.kstest(qs, dens.cdf_q)
        mom = dens.moments()
        eq_err = abs(qs.mean() - mom['E_q']) / mom['E_q']
        cov_emp = np.cov(xs.T)
        cov_err = np.max(np.abs(cov_emp - mom['cov_f'])) / mom['cov_f'][0, 0]
        # isotropy of the sampled angle
        thh = np.arctan2(xs[:, 1], xs[:, 0])
        ks_th = sstats.kstest((thh + np.pi) / (2 * np.pi), 'uniform')
        c3 = bool(ks.pvalue > 1e-3 and eq_err < 5e-3 and cov_err < 2e-2
                  and ks_th.pvalue > 1e-3)
        ok &= c3
        say(f'  C3 exact sampling: KS(q vs exact CDF) p={ks.pvalue:.3f}, '
            f'|E[q]_emp-E[q]_exact|/E[q]={eq_err:.2e}, max|Cov_emp-Cov_exact|/'
            f'Cov_11={cov_err:.2e}, KS(theta uniform) p={ks_th.pvalue:.3f}  '
            f'-> {"PASS" if c3 else "FAIL"}')
        say(f'     E[q]={mom["E_q"]:.6f}  Var[q]={mom["Var_q"]:.6f}  '
            f'sigma_f^2=tr(Cov_f)/d={mom["sigma_f_sq"]:.6f}')
        rec['C3_sampling'] = dict(passed=c3, ks_q_p=float(ks.pvalue),
                                  Eq_rel_err=float(eq_err),
                                  cov_rel_err=float(cov_err),
                                  ks_theta_p=float(ks_th.pvalue), moments=dict(
                                      E_q=mom['E_q'], Var_q=mom['Var_q'],
                                      sigma_f_sq=mom['sigma_f_sq'],
                                      cov_f=mom['cov_f'].tolist()))

        # C4: the Langevin discretization at THE dt ACTUALLY USED preserves p.
        # Starting from EXACT iid draws, any drift of the law of q after a
        # physical time of 4.0 is pure Euler-Maruyama discretization bias.
        # (dt=0.01 FAILS this at (m2-m1)/s_q = 4 and 5; that is why the
        # canonical dt is 0.002.)
        n_steps_c4 = int(round(4.0 / dt))
        sim, _ = make_simulator(dens, dt, 40000, 3)
        x0np, _, _ = dens.sample(20000, np.random.default_rng(5))
        torch.manual_seed(5)
        x = torch.tensor(x0np, dtype=torch.float32)
        for _ in range(n_steps_c4):
            x = sim.step(x)
        q_end = (x ** 2).sum(-1).numpy()
        eq_drift = abs(q_end.mean() - mom['E_q']) / mom['E_q']
        ks_end = sstats.kstest(q_end, dens.cdf_q)
        p_out_exact = 1.0 - float(dens.cdf_q(bd['q_min'])[0])
        p_out_emp = float(np.mean(q_end > bd['q_min']))
        p_drift = abs(p_out_emp - p_out_exact)
        c4 = bool(eq_drift < 0.01 and ks_end.pvalue > 1e-4 and p_drift < 0.01)
        ok &= c4
        say(f'  C4 stationarity of the simulated chain (dt={dt:g}, {n_steps_c4} steps '
            f'= 4.0 physical): E[q] drift={eq_drift:.2e}, KS(q_T vs exact CDF) '
            f'p={ks_end.pvalue:.3f}, P(outer) {p_out_emp:.4f} vs exact '
            f'{p_out_exact:.4f} (drift {p_drift:.2e})  -> {"PASS" if c4 else "FAIL"}')
        rec['C4_stationarity'] = dict(passed=c4, dt=float(dt),
                                      Eq_rel_drift=float(eq_drift),
                                      ks_p=float(ks_end.pvalue),
                                      P_outer_exact=p_out_exact,
                                      P_outer_empirical=p_out_emp,
                                      P_outer_drift=float(p_drift))

        # C5: population symmetry E[X h(||X||)] = 0 for the shell indicator
        lab = (qs > bd['q_min']).astype(float)
        psi = lab - lab.mean()
        v = (xs * psi[:, None]).mean(axis=0)
        se = (xs * psi[:, None]).std(axis=0) / np.sqrt(len(xs))
        z = np.abs(v / se)
        c5 = bool(np.all(z < 4.0))
        ok &= c5
        say(f'  C5 E[X psi_2(X)] on 2e5 EXACT samples: vector={np.round(v,5).tolist()}, '
            f'|z|={np.round(z,2).tolist()} (must be O(1))  -> {"PASS" if c5 else "FAIL"}')
        rec['C5_symmetry'] = dict(passed=c5, vector=v.tolist(), z=z.tolist())

        # C6: single-shell controls are exact and unimodal in q
        for nm, mm in (('inner', m1), ('outer', m2)):
            d1 = one_shell(mm, dens.s_q)
            m1s = d1.mode_structure()
            good = bool(m1s['n_maxima'] == 1 and m1s['n_minima'] == 0)
            ok &= good
            say(f'  C6 single-{nm} shell control (m={mm}, s_q={dens.s_q:.4f}): '
                f'n_maxima={m1s["n_maxima"]}, n_minima={m1s["n_minima"]}, '
                f'E[q]={d1.moments()["E_q"]:.4f}, sigma_f^2='
                f'{d1.moments()["sigma_f_sq"]:.4f}  -> {"PASS" if good else "FAIL"}')
            rec[f'C6_single_{nm}'] = dict(passed=good, n_maxima=m1s['n_maxima'],
                                          n_minima=m1s['n_minima'],
                                          moments={k: (v.tolist() if hasattr(v, 'tolist') else v)
                                                   for k, v in d1.moments().items()})

        # C7: matched Gaussian really has the intended covariance
        mg_means, mg_covs, mg_w, s2 = matched_gaussian_density_params(mom)
        mu_g, cov_g = compute_stationary_moments(mg_means, mg_covs, mg_w)
        cerr = float(np.max(np.abs(cov_g.numpy() - mom['cov_f'])))
        c7 = bool(cerr < 1e-5 and float(mu_g.abs().max()) < 1e-12)
        ok &= c7
        say(f'  C7 matched isotropic Gaussian: sigma^2={s2:.6f}, '
            f'max|Cov_G - Cov_f|={cerr:.2e}, |mu_G|={float(mu_g.abs().max()):.1e}  '
            f'-> {"PASS" if c7 else "FAIL"}')
        rec['C7_matched_gaussian'] = dict(passed=c7, sigma2=float(s2),
                                          max_cov_err=cerr)

        # C8: angle helpers are sign- and degeneracy-aware
        u = np.array([1.0, 0.0])
        c8 = bool(abs(angle_deg(u, -u)) < 1e-9 and abs(angle_deg(u, [0, 1]) - 90) < 1e-9)
        ok &= c8
        say(f'  C8 angle helper sign-invariance: angle(u,-u)=0, angle(e1,e2)=90 '
            f'-> {"PASS" if c8 else "FAIL"}')
        rec['C8_angle_helper'] = dict(passed=c8)

        checks[f'ratio_{ratio:g}'] = rec

    say('\n' + '=' * 78)
    say(f'ALL CHECKS {"PASSED" if ok else "FAILED"}')
    say('=' * 78)
    return ok, checks, lines


# =========================================================================== #
# 7. Main per-grid-point experiment
# =========================================================================== #
def run_grid_point(m1, m2, ratio, cfg):
    t0 = time.time()
    dens = two_shell(m1, m2, ratio)
    bd = dens.barrier_depth()
    mom = dens.moments()
    print(f"\n{'='*78}\nP3b EXACT RINGS  (m2-m1)/s_q = {ratio:g}   s_q = {dens.s_q:.6f}"
          f"\n{'='*78}")
    print(f"  m=({m1}, {m2}), w=(1/2, 1/2); E[q]={mom['E_q']:.4f}, "
          f"sigma_f^2=tr(Cov_f)/d={mom['sigma_f_sq']:.4f}")
    print(f"  radial modes at q={np.round(bd['q_max_inner'],3)}, "
          f"{np.round(bd['q_max_outer'],3)}; barrier minimum at q="
          f"{np.round(bd['q_min'],3)}; depth {bd['log_depth_nats']:.3f} nats")

    ssa_seeds = cfg['seeds'][:cfg['n_ssa_seeds']]
    out = dict(ratio=float(ratio), m1=float(m1), m2=float(m2), s_q=float(dens.s_q),
               barrier=bd, checks_marginal=dens.mode_structure_marginal(),
               radial_modes_q=dens.mode_structure(),
               moments=dict(E_q=mom['E_q'], Var_q=mom['Var_q'],
                                        sigma_f_sq=mom['sigma_f_sq'],
                                        cov_f=mom['cov_f'].tolist()),
               config=dict(cfg))
    arrays = {}

    # ---------------- two-shell: both readouts from the SAME trajectories ----
    amb_l, rad_l, tr_l = [], [], []
    for s in ssa_seeds:
        traj, mom_s, _ = simulate_density(dens, cfg, s)
        amb, rad, qtraj = readouts(traj, mom_s, cfg)
        amb_l.append(amb); rad_l.append(rad)
        tr_l.append(transition_stats(qtraj[..., 0].numpy(), bd['q_min'],
                                     bd['q_max_inner'], bd['q_max_outer'], cfg['dt']))
        if 'rho_sq_ambient' not in arrays:
            arrays['rho_sq_ambient'] = amb['rho_sq']
            arrays['rho_sq_radial'] = rad['rho_sq']
        del traj, qtraj

    def _agg(lst, label):
        S = np.array([r['S_hat'] for r in lst])
        wl = [bool(r['window_limited']) for r in lst]
        rec = dict(S_hat_mean=float(S.mean()), S_hat_std=float(S.std()),
                   S_hat_per_seed=S.tolist(),
                   tau_star=[r['tau_star'] for r in lst],
                   window_limited=bool(any(wl)), window_limited_per_seed=wl,
                   S_full_mean=float(np.mean([r['S_full'] for r in lst])),
                   S_hat_1Nd_mean=float(np.mean([r['S_hat_1Nd'] for r in lst])),
                   tau_star_1Nd=[r['tau_star_1Nd'] for r in lst],
                   floor=lst[0]['floor'], max_lag=lst[0]['max_lag'])
        print(f"  {label:34s} S_hat = {'>= ' if any(wl) else '   '}"
              f"{S.mean():.4f} +/- {S.std():.4f}   tau*={rec['tau_star']}"
              f"{'  [WINDOW-LIMITED: finite-horizon LOWER BOUND]' if any(wl) else ''}")
        return rec

    out['two_shell'] = dict(ambient=_agg(amb_l, 'two-shell AMBIENT x'),
                            radial=_agg(rad_l, 'two-shell RADIAL q=|x|^2'))
    out['transitions'] = tr_l
    ts = tr_l[0]
    print(f"  radial transitions: {ts['committed_transitions_per_traj']:.2f} committed "
          f"per trajectory over H={ts['horizon_phys']:.1f} "
          f"(rate {ts['committed_transition_rate_per_unit_time']:.4f} /unit time; "
          f"implied mean dwell {ts['implied_mean_dwell_time']:.2f})"
          if ts['implied_mean_dwell_time'] else '  radial transitions: none')
    print(f"    residence inner mean={ts['residence_inner']['mean']}, "
          f"outer mean={ts['residence_outer']['mean']}, "
          f"first-passage median={ts['first_passage_to_other_shell']['median']} "
          f"(censored {ts['first_passage_censored_frac']*100:.1f}%)")

    # ---------------- single-shell controls (identical settings) ------------
    for nm, mm in (('inner', m1), ('outer', m2)):
        d1 = one_shell(mm, dens.s_q)
        a_l, r_l = [], []
        for s in ssa_seeds:
            traj, mom1, _ = simulate_density(d1, cfg, 1000 + s)
            a, r, _ = readouts(traj, mom1, cfg)
            a_l.append(a); r_l.append(r)
            del traj
        out[f'single_{nm}'] = dict(
            m=float(mm), s_q=float(dens.s_q),
            moments=dict(E_q=d1.moments()['E_q'],
                         sigma_f_sq=d1.moments()['sigma_f_sq']),
            ambient=_agg(a_l, f'single-{nm} AMBIENT x'),
            radial=_agg(r_l, f'single-{nm} RADIAL q=|x|^2'))

    # ---------------- covariance-matched isotropic Gaussian -----------------
    ga_l, gr_l = [], []
    for s in ssa_seeds:
        a, r, sigma2, gtraj = run_matched_gaussian(mom, cfg, s)
        ga_l.append(a); gr_l.append(r)
        del gtraj
    out['matched_gaussian'] = dict(
        sigma2=float(mom['sigma_f_sq']),
        match_rule='isotropic, sigma^2 = tr(Cov_f)/d -> Cov EXACTLY equal to '
                   'the two-shell Cov_f, and the same diffusion coefficient',
        ambient=_agg(ga_l, 'matched-Gaussian AMBIENT x'),
        radial=_agg(gr_l, 'matched-Gaussian RADIAL q=|x|^2'))

    # ---------------- ratios that decide the claim --------------------------
    def rr(a, b):
        return float(a / b) if b else float('nan')
    R = dict(
        radial_two_shell_over_single_inner=rr(
            out['two_shell']['radial']['S_hat_mean'],
            out['single_inner']['radial']['S_hat_mean']),
        radial_two_shell_over_single_outer=rr(
            out['two_shell']['radial']['S_hat_mean'],
            out['single_outer']['radial']['S_hat_mean']),
        radial_two_shell_over_gaussian=rr(
            out['two_shell']['radial']['S_hat_mean'],
            out['matched_gaussian']['radial']['S_hat_mean']),
        ambient_two_shell_over_single_inner=rr(
            out['two_shell']['ambient']['S_hat_mean'],
            out['single_inner']['ambient']['S_hat_mean']),
        ambient_two_shell_over_single_outer=rr(
            out['two_shell']['ambient']['S_hat_mean'],
            out['single_outer']['ambient']['S_hat_mean']),
        ambient_two_shell_over_gaussian=rr(
            out['two_shell']['ambient']['S_hat_mean'],
            out['matched_gaussian']['ambient']['S_hat_mean']),
        radial_over_ambient_two_shell=rr(
            out['two_shell']['radial']['S_hat_mean'],
            out['two_shell']['ambient']['S_hat_mean']))
    # The DECISIVE comparisons: two-shell vs the WORST-CASE (largest) single-shell
    # control.  A single shell has no radial metastability at all, so any radial
    # excess over BOTH single shells is shell-switching, not within-shell
    # relaxation; and any ambient excess over BOTH single shells would be the
    # ambient readout acquiring a shell signal.
    R['radial_two_shell_over_worst_single'] = rr(
        out['two_shell']['radial']['S_hat_mean'],
        max(out['single_inner']['radial']['S_hat_mean'],
            out['single_outer']['radial']['S_hat_mean']))
    R['ambient_two_shell_over_worst_single'] = rr(
        out['two_shell']['ambient']['S_hat_mean'],
        max(out['single_inner']['ambient']['S_hat_mean'],
            out['single_outer']['ambient']['S_hat_mean']))
    out['ratios'] = R
    print(f"  RATIOS  radial(2-shell)/radial(inner)={R['radial_two_shell_over_single_inner']:.2f}x, "
          f"/radial(outer)={R['radial_two_shell_over_single_outer']:.2f}x, "
          f"/radial(Gauss)={R['radial_two_shell_over_gaussian']:.2f}x")
    print(f"          ambient(2-shell)/ambient(inner)={R['ambient_two_shell_over_single_inner']:.2f}x, "
          f"/ambient(outer)={R['ambient_two_shell_over_single_outer']:.2f}x, "
          f"/ambient(Gauss)={R['ambient_two_shell_over_gaussian']:.2f}x")
    print(f"  DECISIVE: radial 2-shell / WORST single-shell = "
          f"{R['radial_two_shell_over_worst_single']:.2f}x   |   "
          f"ambient 2-shell / WORST single-shell = "
          f"{R['ambient_two_shell_over_worst_single']:.2f}x")

    # ---------------- DA, rotational degeneracy -----------------------------
    def sim_factory(seed):
        return make_simulator(dens, cfg['dt'], cfg['n_pool'], seed)[0]

    per_seed, da_stats = da_across_seeds(sim_factory, cfg, cfg['seeds'])
    out['da_per_seed'] = per_seed
    out['da_stats'] = da_stats
    print(f"  DA lambda1/lambda2 per seed: "
          f"{[None if r is None else round(r,4) for r in da_stats['gap_ratio_per_seed']]}")
    print(f"  DA1 orientation per seed (deg mod 180): "
          f"{np.round(da_stats['da1_orientation_deg_per_seed'],1).tolist()} "
          f"(spread {da_stats['da1_orientation_spread_deg']:.1f} deg, "
          f"max angle vs seed 0 = {da_stats['da1_angle_vs_seed0_deg_max']:.1f} deg)")
    if da_stats.get('auto_lag_warning'):
        print(f"  !! {da_stats['auto_lag_warning']}")

    # NULL for the degeneracy check: an EXACTLY isotropic Gaussian at the same N
    mg_means, mg_covs, mg_w, s2 = matched_gaussian_density_params(mom)
    g_per_seed, g_da_stats = da_across_seeds(None, cfg, cfg['seeds'],
                                             gmm=(mg_means, mg_covs, mg_w))
    out['da_stats_matched_gaussian_null'] = g_da_stats
    print(f"  [null] matched-Gaussian DA lambda1/lambda2 per seed: "
          f"{[None if r is None else round(r,4) for r in g_da_stats['gap_ratio_per_seed']]}"
          f"  (exactly isotropic in population; this is the sampling-noise scale)")
    print(f"    NOTE: the auto lag is argmax_tau lambda1/lambda2, a MAXIMUM over "
          f"checkpoints, and it can land on different lags for the two systems. "
          f"Cross-system comparison uses the FIXED lags below.")

    # ---- FIXED-LAG rotational-degeneracy comparison (the fair one) ----
    fixed = {}
    for t_ref in cfg['fixed_ref_lags_phys']:
        c = min(cfg['checkpoints'], key=lambda c: abs(c * cfg['dt'] - t_ref))
        a = da_stats['fixed_lag_profile'][int(c)]
        b = g_da_stats['fixed_lag_profile'][int(c)]
        fixed[f'{t_ref:g}'] = dict(lag_steps=int(c), lag_phys=float(c * cfg['dt']),
                                   two_shell=a, iso_gaussian_null=b)
        print(f"  [fixed lag tau={c*cfg['dt']:g} ({c} steps)]  two-shell "
              f"lambda1/lambda2={a['gap_ratio_mean']:.3f}, |C12|/diag="
              f"{a['offdiag_over_diag_mean']:.4f}, |C11-C22|/(C11+C22)="
              f"{a['diag_asym_mean']:.4f}, DA1 orientations="
              f"{np.round(a['da1_orientation_deg'],1).tolist()}")
        print(f"                            iso-Gaussian NULL "
              f"lambda1/lambda2={b['gap_ratio_mean']:.3f}, |C12|/diag="
              f"{b['offdiag_over_diag_mean']:.4f}, |C11-C22|/(C11+C22)="
              f"{b['diag_asym_mean']:.4f}, DA1 orientations="
              f"{np.round(b['da1_orientation_deg'],1).tolist()}")
    out['da_fixed_lag_comparison'] = fixed

    # ---------------- observability / classifiers / modality ----------------
    da1 = np.asarray(per_seed[0]['da1'], float)
    da2 = np.asarray(per_seed[0]['da2'], float)
    # Observability / classifiers use a LARGE exact sample (cheap: no bootstrap),
    # so the symmetry statements are tested at high precision.  Modality tests
    # use the smaller N_mod because Silverman bootstraps B critical bandwidths.
    rng = np.random.default_rng(999)
    Xo, qo, _ = dens.sample(cfg['N_obs'], rng)
    shell_o = (qo > bd['q_min']).astype(int)
    obs = shell_label_observability(Xo, qo, shell_o, da1, da2, seed=0)
    obs['n_samples'] = int(cfg['N_obs'])
    rng_m = np.random.default_rng(1999)
    Xm, qm, _ = dens.sample(cfg['N_mod'], rng_m)
    out['observability'] = obs
    print(f"  shell-label coupling to SIGNED DA1 proj: r={obs['coupling_da1']['point_biserial_r']:+.4f} "
          f"(p={obs['coupling_da1']['p_value']:.3g}), AUC={obs['coupling_da1']['auc']:.4f}")
    print(f"  shell-label coupling to SIGNED DA2 proj: r={obs['coupling_da2']['point_biserial_r']:+.4f} "
          f"(p={obs['coupling_da2']['p_value']:.3g}), AUC={obs['coupling_da2']['auc']:.4f}")
    bsd = obs['best_signed_direction']
    print(f"  BEST signed linear direction over 720 angles (N={cfg['N_obs']}): "
          f"AUC={bsd['best_auc']:.4f} (|AUC-0.5|={bsd['best_abs_auc_minus_half']:.4f}); "
          f"permutation null mean={bsd['perm_null_mean']:.4f}, q95={bsd['perm_null_q95']:.4f}, "
          f"p={bsd['perm_p_value']:.3f}")
    lda = obs['classifier_linear_x_lda']
    print(f"  LINEAR classifier on x: LDA bal.acc={lda['balanced_accuracy_test']:.4f} "
          f"(perm null mean={lda['perm_null_mean']:.4f}, q95={lda['perm_null_q95']:.4f}, "
          f"p={lda['perm_p_value']:.3f}), "
          f"logreg bal.acc={obs['classifier_linear_x_logreg']['balanced_accuracy_test']:.4f}"
          f"  (chance 0.5)")
    hp = obs['classifier_linear_x_best_halfplane']
    print(f"  BEST linear HALF-PLANE on x (max over all directions AND thresholds): "
          f"bal.acc={hp['balanced_accuracy']:.4f} (perm null mean={hp['perm_null_mean']:.4f}, "
          f"q95={hp['perm_null_q95']:.4f}, p={hp['perm_p_value']:.3f}) "
          f"<- ABOVE chance: a spread effect, direction arbitrary")
    print(f"  classifier on q:        logreg bal.acc={obs['classifier_q_logreg']['balanced_accuracy_test']:.4f}, "
          f"AUC={obs['classifier_q_auc']:.4f}")
    print(f"  E[X psi_2(X)] = {np.round(obs['E_X_psi2']['vector'],5).tolist()} "
          f"(|z|={np.round(np.abs(obs['E_X_psi2']['z_scores']),2).tolist()}, "
          f"norm/rms-scale={obs['E_X_psi2']['norm_over_rms_scale']:.5f})")

    mod = dict(
        da1_proj=modality_tests(Xm @ da1, cfg['silverman_B'], seed=0),
        da2_proj=modality_tests(Xm @ da2, cfg['silverman_B'], seed=0),
        radial_q=modality_tests(qm, cfg['silverman_B'], seed=0),
        radial_r=modality_tests(np.sqrt(qm), cfg['silverman_B'], seed=0),
        x1_marginal=modality_tests(Xm[:, 0], cfg['silverman_B'], seed=0))
    out['modality'] = mod
    for k, v in mod.items():
        print(f"  modality {k:12s}: dip p={v['dip_p']:.3g}, "
              f"Silverman p={v['silverman_p']:.3g} (nmodes={v['silverman_nmodes']})")

    arrays['da1_seed0'] = da1
    arrays['da2_seed0'] = da2
    out['runtime_s'] = time.time() - t0
    print(f"  [grid point runtime {out['runtime_s']:.1f}s]")
    return out, arrays


# =========================================================================== #
# 8. Reporting
# =========================================================================== #
def _wl(rec):
    return '>=' if rec['window_limited'] else ''


def build_table(results):
    L = []
    L.append('### P3b: exact continuous concentric rings (d=2)\n')
    L.append('Density: p(x) = (1/pi) sum_i w_i g_i(q), q=||x||^2, '
             'g_i = N(m_i, s_q^2) truncated to q>=0, w=(1/2,1/2), '
             f'm_1={results["m1"]}, m_2={results["m2"]}. '
             'Both shells are mean-zero, so no signed linear direction separates them.\n')
    cfg = results['grid'][0]['config']
    L.append(f"Estimator: floor 1/N, tau* = LARGEST lag with rho^2 > 1/N, integrate "
             f"THROUGH tau*, lag 0 excluded. N={cfg['N_ssa']}, dt={cfg['dt']}, "
             f"horizon H={cfg['T_ssa']*cfg['dt']:g}, max_lag={cfg['max_lag']} "
             f"(= {cfg['max_lag']*cfg['dt']:g} in physical time), "
             f"{cfg['n_ssa_seeds']} SSA seeds. `>=` marks WINDOW-LIMITED cells, "
             f"which are finite-horizon LOWER BOUNDS.\n")

    L.append('**Finite-horizon SSA, ambient vs radial readout (identical trajectories '
             'for the two-shell rows; identical N/dt/horizon/lag grid/stopping rule '
             'for every row):**\n')
    L.append('| (m2-m1)/s_q | system | ambient S_hat (x) | radial S_hat (q=norm(x)^2) | radial/ambient |')
    L.append('|---|---|---|---|---|')
    for g in results['grid']:
        r = g['ratio']
        rows = [('two shells', g['two_shell']),
                ('single inner shell', g['single_inner']),
                ('single outer shell', g['single_outer']),
                ('matched isotropic Gaussian', g['matched_gaussian'])]
        for nm, rec in rows:
            a, rd = rec['ambient'], rec['radial']
            L.append(f"| {r:g} | {nm} | {_wl(a)}{a['S_hat_mean']:.3f} "
                     f"+/- {a['S_hat_std']:.3f} | {_wl(rd)}{rd['S_hat_mean']:.3f} "
                     f"+/- {rd['S_hat_std']:.3f} | "
                     f"{rd['S_hat_mean']/a['S_hat_mean']:.2f}x |")
    L.append('')

    L.append('**Key ratios (two-shell relative to its controls). The DECISIVE columns '
             'are the two "worst single-shell" ones: the radial readout must gain '
             'a slow signal that neither single shell has, while the ambient '
             'readout must NOT.**\n')
    L.append('| (m2-m1)/s_q | radial: 2-shell / **worst single** | / inner | / outer | '
             '/ Gaussian | ambient: 2-shell / **worst single** | / inner | / outer | '
             '/ Gaussian |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for g in results['grid']:
        R = g['ratios']
        L.append(f"| {g['ratio']:g} | **{R['radial_two_shell_over_worst_single']:.2f}x** | "
                 f"{R['radial_two_shell_over_single_inner']:.2f}x | "
                 f"{R['radial_two_shell_over_single_outer']:.2f}x | "
                 f"{R['radial_two_shell_over_gaussian']:.2f}x | "
                 f"**{R['ambient_two_shell_over_worst_single']:.2f}x** | "
                 f"{R['ambient_two_shell_over_single_inner']:.2f}x | "
                 f"{R['ambient_two_shell_over_single_outer']:.2f}x | "
                 f"{R['ambient_two_shell_over_gaussian']:.2f}x |")
    L.append('')

    L.append('**Radial metastability and ambient linear (in)visibility:**\n')
    L.append('| (m2-m1)/s_q | barrier (nats) | committed transitions/traj | mean dwell | '
             'DA1 orientation spread (deg) | label~DA1 signed proj, r | '
             'best signed-direction AUC (perm p) | linear-on-x LDA bal.acc (perm p) | '
             'best linear HALF-PLANE bal.acc | q bal.acc |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for g in results['grid']:
        t = g['transitions'][0]
        ds, ob = g['da_stats'], g['observability']
        dwell = t['implied_mean_dwell_time']
        dwell_s = 'n/a' if dwell is None else f'{dwell:.2f}'
        bsd, lda = ob['best_signed_direction'], ob['classifier_linear_x_lda']
        L.append(
            f"| {g['ratio']:g} | {g['barrier']['log_depth_nats']:.2f} | "
            f"{t['committed_transitions_per_traj']:.2f} | {dwell_s} | "
            f"{ds['da1_orientation_spread_deg']:.1f} | "
            f"{ob['coupling_da1']['point_biserial_r']:+.4f} | "
            f"{bsd['best_auc']:.4f} (p={bsd['perm_p_value']:.3f}) | "
            f"{lda['balanced_accuracy_test']:.4f} (p={lda['perm_p_value']:.3f}) | "
            f"{ob['classifier_linear_x_best_halfplane']['balanced_accuracy']:.4f} | "
            f"{ob['classifier_q_logreg']['balanced_accuracy_test']:.4f} |")
    L.append('')
    L.append('HONEST CAVEAT, stated explicitly: the "best linear HALF-PLANE" column '
             'is ABOVE chance. No signed linear score ORDERS the shells (AUC = 0.5, '
             'permutation p large) and LDA is at chance, but the two shells differ '
             'in SPREAD, so an extreme half-plane {x.u > c} is almost purely outer '
             'shell. That rule identifies no direction -- by rotational symmetry '
             'every u gives the identical value, so the maximiser is arbitrary -- '
             'and it does not enter C(tau), the second-moment functional SSA and DA '
             'actually use, which the exact symmetry E[X psi_2(X)] = 0 annihilates.\n')
    L.append(f"Coupling / classifier / symmetry tests use N={results['grid'][0]['observability']['n_samples']} "
             f"EXACT stationary samples; the shell label is the exact-density "
             f"assignment 1[q > q_min]. Permutation p-values (label permutation, "
             f"statistic recomputed end to end) calibrate the max-over-directions "
             f"and the refit-LDA statistics.\n")

    L.append('**Rotational degeneracy of the ambient DA problem, at FIXED lags '
             '(population C(tau) = c(tau) I exactly, so lambda1 = lambda2 and the '
             'DA basis is arbitrary). The isotropic-Gaussian null is EXACTLY '
             'degenerate by construction and is run at the same N and the same '
             'lag, so it calibrates the sampling-noise scale of the splitting:**\n')
    L.append('| (m2-m1)/s_q | lag tau (phys) | lambda1/lambda2 two-shell | '
             'lambda1/lambda2 iso-Gauss NULL | abs(C12)/diag 2-shell | abs(C12)/diag NULL | '
             'DA1 orientations two-shell (deg mod 180) | DA1 orientations NULL |')
    L.append('|---|---|---|---|---|---|---|---|')
    for g in results['grid']:
        for key, rec in g['da_fixed_lag_comparison'].items():
            a, b = rec['two_shell'], rec['iso_gaussian_null']
            L.append(f"| {g['ratio']:g} | {rec['lag_phys']:g} | "
                     f"{a['gap_ratio_mean']:.3f} | {b['gap_ratio_mean']:.3f} | "
                     f"{a['offdiag_over_diag_mean']:.4f} | "
                     f"{b['offdiag_over_diag_mean']:.4f} | "
                     f"{np.round(a['da1_orientation_deg'],0).astype(int).tolist()} | "
                     f"{np.round(b['da1_orientation_deg'],0).astype(int).tolist()} |")
    L.append('')
    L.append('The DA lag selector used elsewhere in the paper (argmax lambda1/lambda2) '
             'is NOT usable here: under exact degeneracy it maximises a sampling '
             'fluctuation and lands on the noisiest checkpoint, where lambda2 can be '
             'negative. That is itself a symptom of the degeneracy, and it is why the '
             'comparison above is made at FIXED, matched lags.\n')

    L.append('**Modality tests (exact stationary samples, N=%d, Silverman B=%d):**\n'
             % (cfg['N_mod'], cfg['silverman_B']))
    L.append('IMPORTANT, and known ANALYTICALLY in advance: by rotational symmetry '
             'EVERY signed linear projection x.u has the SAME law, whose exact mode '
             'structure is listed below. Ring geometry makes that marginal multimodal '
             'with modes at +/- r, SYMMETRIC about 0. A rejection of unimodality on a '
             'DA projection is therefore a ring artefact and is NOT evidence that the '
             'linear readout separates the shells; the coupling / classifier tests '
             'above are what settle that.\n')
    for g in results['grid']:
        mm = g['checks_marginal']
        L.append(f"- (m2-m1)/s_q = {g['ratio']:g}: exact marginal p(x.u) has "
                 f"{mm['n_maxima']} maxima at {np.round(mm['maxima_x1'],3).tolist()} "
                 f"and {mm['n_minima']} minima at {np.round(mm['minima_x1'],3).tolist()}.")
    L.append('')
    L.append('| (m2-m1)/s_q | observable | dip p | Silverman p | Silverman modes |')
    L.append('|---|---|---|---|---|')
    for g in results['grid']:
        for k in ('da1_proj', 'da2_proj', 'x1_marginal', 'radial_q', 'radial_r'):
            v = g['modality'][k]
            L.append(f"| {g['ratio']:g} | {k} | {v['dip_p']:.3g} | "
                     f"{v['silverman_p']:.3g} | {v['silverman_nmodes']} |")
    L.append('')
    return '\n'.join(L)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items() if k != 'rho_sq'}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checks', action='store_true', help='verification only')
    ap.add_argument('--smoke', action='store_true', help='tiny CPU smoke test')
    ap.add_argument('--m1', type=float, default=M1_DEFAULT)
    ap.add_argument('--m2', type=float, default=M2_DEFAULT)
    ap.add_argument('--grid', type=str, default='3,4,5',
                    help='predeclared (m2-m1)/s_q values')
    # CANONICAL dt = 0.002.  dt = 0.01 (the P3 default) FAILS the stationarity
    # check C4 at (m2-m1)/s_q = 4 and 5: Euler-Maruyama bias moves the law of q.
    ap.add_argument('--dt', type=float, default=0.002)
    ap.add_argument('--N_ssa', type=int, default=2000)
    # Horizon H = T_ssa*dt = 40 physical time units; max_lag = T_ssa/2 = 20
    # physical, so every reported lag is estimated from at least T/2 time-pairs
    # per trajectory (estimates at lags near T are far too noisy for a
    # LARGEST-lag-above-floor stopping rule).  The SLOWEST cell is the
    # single-OUTER-shell ambient control, whose tau* is 14-15 physical at
    # N=2000; a 20-unit lag window resolves every cell with margin, while a
    # much longer window would start letting long-lag noise (rho^2 ~ 1e-4 at
    # 60 units) trip the stopping rule.
    ap.add_argument('--T_ssa', type=int, default=20000)
    ap.add_argument('--max_lag', type=int, default=10000)
    ap.add_argument('--N_da', type=int, default=5000)
    ap.add_argument('--N_mod', type=int, default=5000,
                    help='sample size for dip/Silverman (Silverman is the '
                         'expensive part: B bootstrap critical bandwidths)')
    ap.add_argument('--N_obs', type=int, default=200000,
                    help='sample size for the coupling / classifier / symmetry '
                         'tests (cheap, so it is much larger)')
    ap.add_argument('--silverman_B', type=int, default=500)
    ap.add_argument('--fixed_ref_lags', type=str, default='1,3',
                    help='physical lags at which lambda1/lambda2 and the DA1 '
                         'orientation are compared against the isotropic null')
    ap.add_argument('--n_ssa_seeds', type=int, default=3)
    ap.add_argument('--seeds', type=str, default='0,1,2,3,4')
    ap.add_argument('--tag', type=str, default='full')
    ap.add_argument('--output_dir', type=str, default=RESULTS_DIR)
    args = ap.parse_args()

    grid = [float(v) for v in args.grid.split(',')]

    if args.smoke:
        args.N_ssa, args.T_ssa, args.max_lag = 300, 3000, 1500
        args.N_da, args.N_mod, args.silverman_B = 600, 1500, 40
        args.N_obs = 20000
        args.n_ssa_seeds, args.seeds = 2, '0,1,2'
        args.tag = 'smoke' if args.tag == 'full' else args.tag

    if args.max_lag > args.T_ssa // 2:
        print(f'WARNING: max_lag={args.max_lag} > T_ssa/2={args.T_ssa//2}. '
              f'Lags near T are estimated from very few time-pairs and their '
              f'noise can trip the LARGEST-lag-above-floor stopping rule.')

    seeds = [int(s) for s in args.seeds.split(',')]
    # DA checkpoints are declared in PHYSICAL time and converted to steps, so
    # the lag grid is identical across dt values.
    cp_phys = [0.002, 0.004, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0,
               5.0, 7.5, 10.0, 15.0]
    cps = sorted({max(1, int(round(t / args.dt))) for t in cp_phys
                  if int(round(t / args.dt)) <= args.T_ssa})
    cfg = dict(dt=args.dt, N_ssa=args.N_ssa, T_ssa=args.T_ssa,
               max_lag=args.max_lag, N_da=args.N_da, N_mod=args.N_mod,
               N_obs=args.N_obs,
               silverman_B=args.silverman_B, n_ssa_seeds=args.n_ssa_seeds,
               seeds=seeds, n_pool=max(200000, 10 * args.N_da),
               checkpoints=cps, checkpoints_phys=[c * args.dt for c in cps],
               fixed_ref_lags_phys=[float(v) for v in args.fixed_ref_lags.split(',')],
               horizon_phys=args.T_ssa * args.dt,
               estimator=('floor=1/N; tau*=LARGEST lag with rho^2>1/N; integrate '
                          'THROUGH tau*; lag 0 excluded; S_hat=dt*sum rho^2'))

    ok, checks, _ = run_checks(args.m1, args.m2, grid, dt=args.dt)
    if not ok:
        print('\nCHECKS FAILED -- refusing to simulate.')
        sys.exit(2)
    if args.checks:
        os.makedirs(args.output_dir, exist_ok=True)
        p = os.path.join(args.output_dir, f'p3b_checks_{args.tag}.json')
        with open(p, 'w') as f:
            json.dump(_jsonable(dict(checks=checks, passed=ok)), f, indent=2)
        print(f'\nChecks written to {p}')
        return

    t0 = time.time()
    results = dict(m1=args.m1, m2=args.m2, grid_spec=grid, grid=[],
                   checks=checks, tag=args.tag)
    all_arrays = {}
    for ratio in grid:
        rec, arr = run_grid_point(args.m1, args.m2, ratio, cfg)
        results['grid'].append(rec)
        for k, v in arr.items():
            all_arrays[f'ratio{ratio:g}_{k}'] = np.asarray(v)
    results['total_runtime_s'] = time.time() - t0

    os.makedirs(args.output_dir, exist_ok=True)
    jp = os.path.join(args.output_dir, f'p3b_results_{args.tag}.json')
    with open(jp, 'w') as f:
        json.dump(_jsonable(results), f, indent=2)
    np.savez_compressed(
        os.path.join(args.output_dir, f'p3b_arrays_{args.tag}.npz'), **all_arrays)
    tbl = build_table(results)
    tp = os.path.join(args.output_dir, f'p3b_table_{args.tag}.md')
    with open(tp, 'w') as f:
        f.write(tbl)
    print('\n' + tbl)
    print(f'\nWrote {jp}\n      {tp}\n      '
          f'{os.path.join(args.output_dir, f"p3b_arrays_{args.tag}.npz")}')
    print(f'Total runtime {results["total_runtime_s"]:.1f}s')


if __name__ == '__main__':
    main()
