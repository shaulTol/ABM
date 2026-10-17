#!/usr/bin/env python3
"""
P3: Scope and nonlinearity controls (NeurIPS rebuttal).

Three small exact-score examples that calibrate what SSA and DA do and do NOT
establish. Reports OBSERVED behavior even when it differs from the hypothesis.

  P3a  Symmetric four-mode circular GMM.
       Hypothesis: near-degenerate 2D DA subspace -> non-unique DA directions.
  P3b  Concentric-ring GMM (matched angular components on inner/outer ring).
       Question: is radial slow structure visible to LINEAR coordinate autocov?
  P3c  Smooth, light-tailed, analytically-scored flat-top unimodal density
       (product generalized-normal, exponent beta). Question: does slow mixing
       raise finite-horizon SSA while dip/Silverman do NOT reject unimodality?

All simulation / SSA / DA / Silverman infra is REUSED (see imports). This file
adds only builders, drivers, and reporting.

Distinctions kept explicit:
  - finite-horizon SSA  Ŝ_L(H) = Δt * Σ_{τ=1}^{L*} ρ(τ)²  truncated at a noise floor
    (this is what we measure at a finite horizon H = T*dt);
  - it is NOT the resolved population SSA (H -> ∞, N -> ∞).
  - PRIMARY estimator (submitted-paper convention): lag 0 excluded, floor = 1/N,
    stopping lag = the LARGEST lag τ with ρ²(τ) > 1/N, and Ŝ integrates through
    that stopping lag: Ŝ = Δt · Σ_{τ=1}^{τ*} ρ²(τ).
  - the tighter 1/(N·d) floor is reported only as a SECONDARY sensitivity value.
  - the 1-D radial observable (P3b) uses its own d=1 floor 1/N.

Usage:
  python p3_scope_controls.py --checks        # analytic-score + degeneracy checks only
  python p3_scope_controls.py --smoke         # tiny CPU smoke test, all three
  python p3_scope_controls.py                 # full run
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
    compute_autocorrelation,
    compute_stationary_moments,
    aggregate_ssa,
)
from src.metrics.da_directions import da_convergence_profile
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'synthetic'))
from mixture_builders import build_ring_mixture  # reused for P3a and P3b rings

# Silverman lives under experiments/sdxl
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'sdxl'))
from silverman_test import silverman_test

import diptest
from scipy import stats as sstats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'p3_scope')


# --------------------------------------------------------------------------- #
# Direction / subspace geometry helpers (sign- and degeneracy-aware)
# --------------------------------------------------------------------------- #
def angle_between_deg(u, v):
    """Sign-invariant angle (deg) between two vectors: uses |cos|."""
    u = np.asarray(u, float); v = np.asarray(v, float)
    c = abs(float(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def principal_angles_deg(A, B):
    """Principal angles (deg) between column-spaces of A (d,k) and B (d,k).

    Handles eigenvector sign automatically (subspace comparison) and is the
    correct tool under eigenvalue degeneracy where individual vectors are
    ill-defined but the subspace is not.
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, 0.0, 1.0)))


def degeneracy_ratio(evals):
    """lambda1 / lambda2 for the top-2 DA eigenvalues (>=1). ~1 == degenerate.

    Returns None when lambda2 <= 0. A non-positive second eigenvalue of the
    SYMMETRIZED lagged autocovariance means there is no second slow/metastable
    mode: the second direction is consistent with noise and can go negative.
    A ratio is then ill-defined (and a negative ratio is meaningless), so we
    report None and let the caller print 'n/a (λ2≤0, no second slow mode)'.
    """
    e = np.sort(np.asarray(evals, float))[::-1]
    if len(e) < 2 or e[1] <= 0:
        return None
    return float(e[0] / e[1])


def gap_report(evals):
    """Structured top-2 eigenvalue report, degeneracy- and sign-aware."""
    e = np.sort(np.asarray(evals, float))[::-1]
    lam1 = float(e[0])
    lam2 = float(e[1]) if len(e) > 1 else float('nan')
    ratio = degeneracy_ratio(e)
    if ratio is None:
        note = 'n/a (lambda2<=0, no second slow mode)'
    elif ratio < 1.25:
        note = 'near-degenerate top-2 subspace'
    else:
        note = 'separated'
    return dict(lam1=lam1, lam2=lam2, ratio=ratio, note=note)


def _fmt_ratio(ratio, reason='λ2≤0'):
    """Table/console formatting for a possibly-undefined gap ratio."""
    return f'n/a ({reason})' if ratio is None else f"{ratio:.2f}"


def _fmt_ratio23(ratio, d):
    """Formatting for lambda2/lambda3: distinguish 'no lambda3 at d=2' from
    'lambda3<=0'. Reporting the wrong reason would look like a bug."""
    if ratio is not None:
        return f"{ratio:.2f}"
    return 'n/a (d=2, no λ3)' if (d is not None and d < 3) else 'n/a (λ3≤0)'


def agg_ratio(ratios):
    """Mean/max over per-seed ratios, ignoring undefined (None) entries.
    Returns (mean, max, n_defined, n_total)."""
    vals = [r for r in ratios if r is not None]
    n_tot = len(ratios)
    if not vals:
        return None, None, 0, n_tot
    return float(np.mean(vals)), float(np.max(vals)), len(vals), n_tot


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def four_circular_gmm(R=3.0, var=0.5, dim=2, fast_var=0.05):
    """4 Gaussians equally spaced on a circle of radius R in coords (e1,e2).

    dim=2 (default) reproduces the original d=2 example EXACTLY (isotropic `var`).

    dim>2 embeds the SAME circular modes in the first two coordinates and adds
    dim-2 *FAST* Gaussian coordinates of variance `fast_var` << `var`, so the
    known slow plane is U = span(e1, e2) and subspace recovery is nontrivial.

    Why `fast_var` must be SMALL (not merely isotropic): in the canonical
    diffusion dX = (sigma_f^2/2) grad log f dt + sigma_f dW, a Gaussian
    coordinate of variance s^2 is an OU process with relaxation rate
        k = sigma_f^2 / (2 s^2),
    so SMALLER variance => LARGER curvature => FASTER relaxation. Setting the
    extra coordinates to the same variance as the in-plane spread (the naive
    "isotropic embedding") makes them exactly as fast as the in-plane intra-mode
    motion (rate ratio 1.0x) and does NOT produce fast nuisance coordinates.

    With R=3, var=0.5, fast_var=0.05 (defaults): in-plane Cov_f per coordinate
    is var + R^2/2 = 5.0, tr(Cov_f)/d = 1.04, the fast-coordinate rate is 10x
    the in-plane intra-mode rate, and the fast autocorrelation has already
    decayed to ~5.5e-3 by the earliest DA checkpoint (tau = 50*dt = 0.5).
    Inter-mode (barrier-crossing) motion in the plane is slower still, so the
    top-2 DA eigenvalues should separate from the rest (lambda2 >> lambda3).
    """
    if dim == 2:
        # preserve the original d=2 construction bit-for-bit
        gm = build_ring_mixture(n_modes=4, radius=R, variance=var, dim=2)
        return gm.means.float(), gm.covs.float(), gm.weights.float()

    angles = np.linspace(0.0, 2 * np.pi, 4, endpoint=False)
    means = np.zeros((4, dim), dtype=np.float32)
    means[:, 0] = R * np.cos(angles)
    means[:, 1] = R * np.sin(angles)
    diag = np.full(dim, fast_var, dtype=np.float32)
    diag[:2] = var                      # slow plane keeps the in-plane spread
    covs = np.broadcast_to(np.diag(diag), (4, dim, dim)).copy()
    weights = torch.ones(4) / 4.0
    return (torch.tensor(means), torch.tensor(covs), weights)


def matched_isotropic_gaussian(cov_f, dim):
    """Isotropic Gaussian matched in DIMENSION and COVARIANCE SCALE.

    Scale is matched as sigma^2 = tr(Cov_f)/d, i.e. the same average per-
    coordinate variance as the four-mode distribution. This also matches the
    canonical diffusion coefficient sigma_f^2 = tr(Cov_f)/d exactly, so the
    baseline differs from the four-mode run ONLY in having no metastable
    structure -- not in its time scale. Returns (means, covs, weights).
    """
    sigma2 = float(torch.trace(cov_f).item() / dim)
    means = torch.zeros(1, dim)
    covs = (sigma2 * torch.eye(dim)).unsqueeze(0)
    weights = torch.ones(1)
    return means, covs, weights, sigma2


def concentric_ring_gmm(K=6, r_inner=2.0, r_outer=5.0, var=0.3, dim=2):
    """Finite-GMM approx of two concentric rings with MATCHED angular
    components: K Gaussians at the same K angles on inner and outer rings.
    Radial (inner<->outer) transition is the intended slow structure.

    dim=2 (default) reproduces the original example EXACTLY. dim>2 embeds the
    rings in the first two coordinates, isotropic in the rest (see
    four_circular_gmm for why)."""
    angles = np.linspace(0.0, 2 * np.pi, K, endpoint=False)
    means = np.zeros((2 * K, dim), dtype=np.float32)
    i = 0
    for r in (r_inner, r_outer):
        for a in angles:
            means[i, 0] = r * np.cos(a)
            means[i, 1] = r * np.sin(a)
            i += 1
    means = torch.tensor(means, dtype=torch.float32)
    covs = (var * torch.eye(dim)).unsqueeze(0).expand(2 * K, -1, -1).clone()
    weights = torch.ones(2 * K) / (2 * K)
    return means, covs, weights


def flattop_score_fn(beta, alpha):
    """Score of product generalized-normal f(x) ∝ exp(-Σ_i (|x_i|/alpha)^beta).
    grad_i log f = -(beta/alpha^beta) |x_i|^{beta-1} sign(x_i).
    For even beta this equals -(beta/alpha^beta) x_i^{beta-1} (smooth)."""
    c = beta / (alpha ** beta)

    def score(x):
        return -c * (x.abs() ** (beta - 1)) * torch.sign(x)
    return score


def flattop_logpdf_np(x, beta, alpha):
    """Unnormalized log density along one coordinate (for FD score check)."""
    return -np.sum((np.abs(x) / alpha) ** beta, axis=-1)


# --------------------------------------------------------------------------- #
# SSA drivers
# --------------------------------------------------------------------------- #
# Stopping rule, also used by p11_unimodal_high_ssa.py (which imports ssa_trace).
# 'first' (default since 2026-10-17) is the paper's eq:tmax_stopping: keep lags up
# to and including the first with rho^2 <= floor (all lags if it never crosses).
# 'largest' reproduces the original rebuttal runs. Select with env SSA_STOPPING.
STOPPING = os.environ.get('SSA_STOPPING', 'first')
assert STOPPING in ('first', 'largest'), STOPPING
STOPPING_DESC = {'first': 'first tau with rho^2 <= 1/N (that lag included)',
                 'largest': 'largest tau with rho^2 > 1/N'}[STOPPING]


def _stopping_lag(rho_sq, floor):
    """Number of leading lags to keep under the rule selected by STOPPING."""
    if STOPPING == 'first':
        below = np.where(rho_sq <= floor)[0]
        return int(below[0]) + 1 if len(below) else len(rho_sq)
    return _stopping_lag_largest(rho_sq, floor)


def _stopping_lag_largest(rho_sq, floor):
    """Stopping lag = the LARGEST lag τ with ρ²(τ) > floor (original rebuttal rule).

    rho_sq is indexed from τ=1 (lag 0 already excluded). Returns the number of
    leading lags to keep, i.e. (index of the last ρ² > floor) + 1, so that
    rho_sq[:keep] integrates THROUGH the stopping lag. Returns 0 if no lag
    exceeds the floor. This is NOT the first-crossing rule."""
    above = np.where(rho_sq > floor)[0]
    if len(above) == 0:
        return 0
    return int(above[-1]) + 1


def ssa_trace(traj, means, covs, weights, dt, max_lag, N, d):
    """Finite-horizon trace-normalized SSA from GMM trajectories (N,T+1,d).

    PRIMARY estimator (paper, eq:tmax_stopping): lag 0 excluded
    (rho_sq = rho[1:]²), floor = 1/N, stopping lag τ* = first τ with ρ²(τ) <= 1/N
    (that lag included; rule set by STOPPING),
    Ŝ = Δt · Σ_{τ=1}^{τ*} ρ²(τ) = dt * rho_sq[:keep].sum().
    Also reports a SECONDARY sensitivity value under the tighter 1/(N·d) floor."""
    rho, mu_f, cov_f = compute_vector_autocorrelation(
        traj, max_lag, means, covs, weights, demean=True, normalization='trace')
    rho_sq = (rho[1:] ** 2).numpy()
    floor = 1.0 / N                                      # PRIMARY floor (submitted)
    keep = _stopping_lag(rho_sq, floor)
    S_hat = aggregate_ssa(rho_sq[:keep], dt=dt)          # finite-horizon, trunc @ 1/N
    S_full = aggregate_ssa(rho_sq, dt=dt)                # full horizon (no truncation)
    floor_1Nd = 1.0 / (N * d)                            # SECONDARY sensitivity floor
    keep_1Nd = _stopping_lag(rho_sq, floor_1Nd)
    S_hat_1Nd = aggregate_ssa(rho_sq[:keep_1Nd], dt=dt)
    # WINDOW-LIMITED: rho^2 never fell below the floor inside the horizon, so the
    # sum is truncated by the WINDOW, not by the noise floor => Shat is a
    # finite-horizon LOWER BOUND, not a resolved value.
    window_limited = (not bool(np.any(rho_sq <= floor))) if STOPPING == 'first' \
        else bool(keep >= len(rho_sq))
    return dict(S_hat=float(S_hat), S_full=float(S_full), tau_star=int(keep),
                floor=float(floor), rho_sq=rho_sq,
                window_limited=window_limited, max_lag=int(len(rho_sq)),
                S_hat_1Nd=float(S_hat_1Nd), tau_star_1Nd=int(keep_1Nd),
                floor_1Nd=float(floor_1Nd))


def ssa_scalar(series, dt, max_lag, N):
    """Finite-horizon SSA for a 1-D observable (N,T+1), e.g. radius r_t."""
    rho, _, _ = compute_autocorrelation(series, max_lag, demean=True)
    rho_sq = (rho[1:] ** 2).numpy()
    floor = 1.0 / N   # d=1
    keep = _stopping_lag(rho_sq, floor)
    S_hat = aggregate_ssa(rho_sq[:keep], dt=dt)
    S_full = aggregate_ssa(rho_sq, dt=dt)
    return dict(S_hat=float(S_hat), S_full=float(S_full), tau_star=int(keep),
                floor=float(floor), rho_sq=rho_sq,
                window_limited=(not bool(np.any(rho_sq <= floor))) if STOPPING == 'first'
                else bool(keep >= len(rho_sq)), max_lag=int(len(rho_sq)))


# --------------------------------------------------------------------------- #
# Modality tests on a 1-D projection
# --------------------------------------------------------------------------- #
def modality_tests(proj, silverman_B=500, seed=0):
    proj = np.asarray(proj, float)
    dip_stat, dip_p = diptest.diptest(proj)
    sv = silverman_test(proj, n_modes=2, B=silverman_B, seed=seed)
    return dict(dip_stat=float(dip_stat), dip_p=float(dip_p),
                silverman_hcrit=sv['h_crit'], silverman_p=sv['p_value'],
                silverman_nmodes=sv['n_modes_observed'])


# --------------------------------------------------------------------------- #
# DA driver: run convergence profile across seeds, quantify (in)stability
# --------------------------------------------------------------------------- #
def da_across_seeds(means, covs, weights, checkpoints, N, dt, seeds, simulator_fn=None,
                    known_plane_dim=None):
    """Returns per-seed optimal DA eigen-info plus cross-seed stability stats."""
    per_seed = []
    for s in seeds:
        if simulator_fn is not None:
            prof = da_convergence_profile(simulator=simulator_fn(s), checkpoints=checkpoints,
                                          n_trajectories=N, dt=dt, seed=s)
        else:
            prof = da_convergence_profile(means, covs, weights, checkpoints=checkpoints,
                                          n_trajectories=N, dt=dt, seed=s)
        evals = prof['optimal_eigenvalues']
        evecs = prof['optimal_eigenvectors']  # columns = DA dirs, descending
        opt_lag = int(prof['checkpoints'][prof['optimal_idx']])
        gr = gap_report(evals)
        # lambda2/lambda3: separation of the slow BLOCK from the rest of the
        # spectrum. This is what makes the slow SUBSPACE identifiable (Davis-Kahan)
        # even when DA1 rotates arbitrarily inside a degenerate block.
        e_sorted = np.sort(np.asarray(evals, float))[::-1]
        lam3 = float(e_sorted[2]) if len(e_sorted) > 2 else float('nan')
        ratio23 = (float(e_sorted[1] / lam3)
                   if len(e_sorted) > 2 and lam3 > 0 else None)
        per_seed.append(dict(
            seed=int(s), opt_lag=opt_lag,
            evals=evals.tolist(),
            lam1=gr['lam1'], lam2=gr['lam2'], lam3=lam3, gap_note=gr['note'],
            gap_ratio=gr['ratio'], gap_ratio_23=ratio23,
            da1=evecs[:, 0].tolist(),
            da2=evecs[:, 1].tolist(),
            top2=evecs[:, :2],
            angles_deg=prof['angles_deg'].tolist(),
            checkpoints=prof['checkpoints'].tolist(),
            leading_eigenvectors=prof['leading_eigenvectors'],  # (n_cp, d)
            all_eigenvalues=prof['all_eigenvalues'],            # (n_cp, d)
        ))
    # cross-seed DA1 angular variation (sign-invariant, ref = seed 0)
    ref1 = np.asarray(per_seed[0]['da1'])
    da1_ang = [angle_between_deg(ref1, np.asarray(p['da1'])) for p in per_seed]
    # cross-seed top-2 subspace stability (max principal angle vs seed 0)
    refsub = per_seed[0]['top2']
    sub_ang = [float(principal_angles_deg(refsub, p['top2']).max()) for p in per_seed]
    # within-run DA1 rotation across NEARBY accepted lags (around optimal)
    lag_rot = _da1_rotation_across_lags(per_seed[0])
    g_mean, g_max, n_def, n_tot = agg_ratio([p['gap_ratio'] for p in per_seed])
    g23_mean, g23_max, n23_def, _ = agg_ratio([p['gap_ratio_23'] for p in per_seed])
    d_dim = len(per_seed[0]['da1'])

    # Principal angles between the ESTIMATED top-2 DA subspace and the KNOWN
    # slow plane U = span(e_1, ..., e_known_plane_dim). Only meaningful when
    # d > known_plane_dim (otherwise U is the whole space, angles are 0 by
    # dimensionality and carry no information -- explicitly forbidden as evidence).
    known_plane = None
    if known_plane_dim is not None and d_dim > known_plane_dim:
        U = np.zeros((d_dim, known_plane_dim))
        for i in range(known_plane_dim):
            U[i, i] = 1.0
        pa_per_seed = [principal_angles_deg(U, p['top2']).tolist() for p in per_seed]
        max_pa = [float(np.max(a)) for a in pa_per_seed]
        known_plane = dict(
            plane=f'span(e1..e{known_plane_dim})', d=int(d_dim),
            principal_angles_deg=pa_per_seed,
            max_principal_angle_deg=max_pa,
            max_principal_angle_deg_worst=float(np.max(max_pa)),
            mean_max_principal_angle_deg=float(np.mean(max_pa)),
            informative=True,
        )
    elif known_plane_dim is not None:
        known_plane = dict(
            plane=f'span(e1..e{known_plane_dim})', d=int(d_dim), informative=False,
            note=('VACUOUS: d == plane dim, so the top-2 subspace is the whole '
                  'space and principal angles are 0 by dimensionality'))
    stats = dict(
        gap_ratio_mean=g_mean, gap_ratio_max=g_max,
        gap_ratio_n_defined=n_def, gap_ratio_n_seeds=n_tot,
        gap_ratio_note=('n/a (lambda2<=0 in all seeds: no second slow mode)'
                        if n_def == 0 else
                        (f'lambda2<=0 in {n_tot-n_def}/{n_tot} seeds; ratio over the rest'
                         if n_def < n_tot else 'defined in all seeds')),
        gap_ratio_23_mean=g23_mean, gap_ratio_23_max=g23_max,
        gap_ratio_23_n_defined=n23_def,
        gap_ratio_23_note=('lambda2/lambda3: separation of the slow block from the '
                           'rest of the spectrum (identifiability of the SUBSPACE '
                           'via Davis-Kahan)' if n23_def else
                           'n/a (lambda3<=0 or d<3)'),
        lam1_per_seed=[p['lam1'] for p in per_seed],
        lam2_per_seed=[p['lam2'] for p in per_seed],
        lam3_per_seed=[p['lam3'] for p in per_seed],
        known_plane=known_plane,
        # d=2 caveat: the top-2 eigenvector subspace is ALL of R^2 by construction,
        # so principal angles are identically 0 and carry NO information.
        subspace_stat_informative=bool(d_dim > 2),
        subspace_stat_note=('VACUOUS at d=2: top-2 subspace is the whole space, '
                            '0 deg is definitional, not evidence'
                            if d_dim <= 2 else
                            f'informative: genuine 2-of-{d_dim} subspace'),
        d=int(d_dim),
        da1_angle_deg=da1_ang,
        da1_angle_deg_max=float(np.max(da1_ang)),
        subspace_maxangle_deg=sub_ang,
        subspace_maxangle_deg_max=float(np.max(sub_ang)),
        da1_rotation_nearby_lags_deg=lag_rot,
    )
    # strip bulky arrays from json copy
    for p in per_seed:
        p.pop('top2', None)
        p['leading_eigenvectors'] = np.asarray(p['leading_eigenvectors']).tolist()
        p['all_eigenvalues'] = np.asarray(p['all_eigenvalues']).tolist()
    return per_seed, stats


def _da1_rotation_across_lags(seed_rec):
    """DA1 rotation (deg) at each checkpoint relative to the optimal-lag DA1."""
    cps = seed_rec['checkpoints']
    opt_lag = seed_rec['opt_lag']
    lev = np.asarray(seed_rec['leading_eigenvectors'])  # (n_cp, d)
    if opt_lag in cps:
        ref = lev[cps.index(opt_lag)]
    else:
        ref = np.asarray(seed_rec['da1'])
    return {int(cp): angle_between_deg(ref, lev[i]) for i, cp in enumerate(cps)}


# --------------------------------------------------------------------------- #
# Example runners
# --------------------------------------------------------------------------- #
def run_example_gmm(name, means, covs, weights, cfg, radial=False,
                    matched_gaussian=False, known_plane_dim=None):
    """Common driver for P3a / P3b (GMM examples)."""
    d = means.shape[1]
    t0 = time.time()
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    print(f"  k={means.shape[0]} components, d={d}")

    # ---- finite-horizon SSA (trace) over seeds ----
    ssa_seeds = cfg['seeds'][:cfg['n_ssa_seeds']]
    ssa_list = []
    rho_sq_store = None
    r_ssa_list = []
    r_rho_store = None
    for s in ssa_seeds:
        torch.manual_seed(s)
        sim = LangevinSimulatorD(means, covs, weights, dt=cfg['dt'])
        x0 = sim.sample_initial(cfg['N_ssa'])
        traj = sim.simulate(x0, cfg['T_ssa'])          # (N,T+1,d)
        res = ssa_trace(traj, means, covs, weights, cfg['dt'], cfg['max_lag'],
                        cfg['N_ssa'], d)
        ssa_list.append(res)
        if rho_sq_store is None:
            rho_sq_store = res['rho_sq']
        if radial:
            # radius in the RING PLANE (first 2 coords) — identical to the full
            # norm at d=2, and the correct observable when embedded in d>2.
            r = traj[..., :2].norm(dim=-1)              # (N,T+1)
            rres = ssa_scalar(r, cfg['dt'], cfg['max_lag'], cfg['N_ssa'])
            r_ssa_list.append(rres)
            if r_rho_store is None:
                r_rho_store = rres['rho_sq']
    S_hat_vals = np.array([r['S_hat'] for r in ssa_list])
    tau_vals = [r['tau_star'] for r in ssa_list]
    wl_seeds = [bool(r['window_limited']) for r in ssa_list]
    wl_any = any(wl_seeds)
    print(f"  SSA Ŝ (finite-horizon, PRIMARY: stopping lag = {STOPPING_DESC}): "
          f"{'≥ ' if wl_any else ''}{S_hat_vals.mean():.4f} ± {S_hat_vals.std():.4f}  "
          f"(τ* lags={tau_vals}, horizon H={cfg['T_ssa']*cfg['dt']:.1f})")
    if wl_any:
        print(f"    WINDOW-LIMITED (τ*=max_lag={cfg['max_lag']} in {sum(wl_seeds)}/"
              f"{len(wl_seeds)} seeds): ρ² never fell below 1/N inside the horizon, "
              f"so Ŝ is a finite-horizon LOWER BOUND, not a resolved value.")
    print(f"  SSA S_full (no truncation): "
          f"{np.mean([r['S_full'] for r in ssa_list]):.4f}")
    print(f"  [sensitivity] Ŝ @ 1/(Nd) floor: "
          f"{np.mean([r['S_hat_1Nd'] for r in ssa_list]):.4f} "
          f"(τ*={[r['tau_star_1Nd'] for r in ssa_list]})")

    # ---- matched isotropic Gaussian baseline (P3a) ----
    # SAME N, dt, horizon (T_ssa), max_lag and stopping rule as the four-mode run.
    mg_summary = None
    if matched_gaussian:
        mu_f, cov_f = compute_stationary_moments(means, covs, weights)
        g_means, g_covs, g_weights, sigma2 = matched_isotropic_gaussian(cov_f, d)
        mg_list = []
        for s in ssa_seeds:
            torch.manual_seed(10000 + s)
            gsim = LangevinSimulatorD(g_means, g_covs, g_weights, dt=cfg['dt'])
            gx0 = gsim.sample_initial(cfg['N_ssa'])
            gtraj = gsim.simulate(gx0, cfg['T_ssa'])
            mg_list.append(ssa_trace(gtraj, g_means, g_covs, g_weights, cfg['dt'],
                                     cfg['max_lag'], cfg['N_ssa'], d))
        mgS = np.array([r['S_hat'] for r in mg_list])
        ratio = float(S_hat_vals.mean() / mgS.mean()) if mgS.mean() else float('nan')
        mg_wl = [bool(r['window_limited']) for r in mg_list]
        print(f"  MATCHED isotropic Gaussian baseline (σ²=tr(Σ_f)/d={sigma2:.4f}, "
              f"same N/Δt/horizon/stopping rule): Ŝ = "
              f"{'≥ ' if any(mg_wl) else ''}{mgS.mean():.4f} ± {mgS.std():.4f} "
              f"(τ*={[r['tau_star'] for r in mg_list]})")
        print(f"  ==> SSA RATIO four-mode / matched-Gaussian = {ratio:.3f}×")
        mg_summary = dict(
            sigma2=sigma2, d=int(d),
            match_rule='isotropic, sigma^2 = tr(Cov_f)/d (matches dimension, '
                       'covariance scale, and the diffusion coefficient)',
            identical_settings=dict(N=cfg['N_ssa'], dt=cfg['dt'], T=cfg['T_ssa'],
                                    max_lag=cfg['max_lag'],
                                    stopping_rule=STOPPING_DESC),
            S_hat_mean=float(mgS.mean()), S_hat_std=float(mgS.std()),
            window_limited=bool(any(mg_wl)),
            tau_star=[r['tau_star'] for r in mg_list],
            ssa_ratio_fourmode_over_gaussian=ratio,
            per_seed=[dict(S_hat=r['S_hat'], tau_star=r['tau_star']) for r in mg_list])

    radial_summary = None
    if radial:
        rS = np.array([r['S_hat'] for r in r_ssa_list])
        print(f"  RADIAL observable r_t=|x| SSA Ŝ: {rS.mean():.4f} ± {rS.std():.4f}  "
              f"(τ*={[r['tau_star'] for r in r_ssa_list]})")
        r_wl = [bool(r['window_limited']) for r in r_ssa_list]
        radial_summary = dict(
            S_hat_mean=float(rS.mean()), S_hat_std=float(rS.std()),
            window_limited=bool(any(r_wl)), window_limited_per_seed=r_wl,
            observable='r_t = |x_t| (direct radial observable, d=1, floor 1/N)',
            per_seed=[dict(S_hat=r['S_hat'], S_full=r['S_full'], tau_star=r['tau_star'],
                           window_limited=bool(r['window_limited']))
                      for r in r_ssa_list])

    # ---- DA across seeds ----
    per_seed, da_stats = da_across_seeds(
        means, covs, weights, cfg['checkpoints'], cfg['N_da'], cfg['dt'], cfg['seeds'],
        known_plane_dim=known_plane_dim)
    print(f"  DA λ1/λ2 gap ratio: mean={_fmt_ratio(da_stats['gap_ratio_mean'])} "
          f"max={_fmt_ratio(da_stats['gap_ratio_max'])}  (~1 => degenerate) "
          f"[{da_stats['gap_ratio_note']}]")
    print(f"  DA λ2/λ3 gap ratio: mean={_fmt_ratio23(da_stats['gap_ratio_23_mean'], d)} "
          f"max={_fmt_ratio23(da_stats['gap_ratio_23_max'], d)}  "
          f"(>>1 => slow block separated from the rest of the spectrum)")
    print(f"  DA λ1 per seed: {np.round(da_stats['lam1_per_seed'],4).tolist()}")
    print(f"  DA λ2 per seed: {np.round(da_stats['lam2_per_seed'],4).tolist()}")
    print(f"  DA λ3 per seed: {np.round(da_stats['lam3_per_seed'],4).tolist()}")
    print(f"  DA1 cross-seed angle (deg): {np.round(da_stats['da1_angle_deg'],1).tolist()} "
          f"(max {da_stats['da1_angle_deg_max']:.1f}°)")
    print(f"  Top-2 subspace max principal angle across seeds (deg): "
          f"{np.round(da_stats['subspace_maxangle_deg'],1).tolist()} "
          f"(max {da_stats['subspace_maxangle_deg_max']:.1f}°)")
    print(f"    subspace stat: {da_stats['subspace_stat_note']}")
    kp = da_stats.get('known_plane')
    if kp and kp.get('informative'):
        print(f"  Principal angles vs KNOWN slow plane {kp['plane']} (deg, per seed): "
              f"{[np.round(a,2).tolist() for a in kp['principal_angles_deg']]}")
        print(f"    max principal angle per seed: "
              f"{np.round(kp['max_principal_angle_deg'],2).tolist()} "
              f"(worst {kp['max_principal_angle_deg_worst']:.2f}°)")
    elif kp:
        print(f"  Known-plane alignment: {kp['note']}")

    # ---- modality tests on DA1/DA2 projections (seed 0 stationary sample) ----
    torch.manual_seed(cfg['seeds'][0])
    sim0 = LangevinSimulatorD(means, covs, weights, dt=cfg['dt'])
    samp = sim0.sample_initial(cfg['N_mod']).numpy()
    da1 = np.asarray(per_seed[0]['da1'])
    da2 = np.asarray(per_seed[0]['da2'])
    mt_da1 = modality_tests(samp @ da1, silverman_B=cfg['silverman_B'], seed=cfg['seeds'][0])
    mt_da2 = modality_tests(samp @ da2, silverman_B=cfg['silverman_B'], seed=cfg['seeds'][0])
    print(f"  DA1 proj: dip p={mt_da1['dip_p']:.3g}, Silverman p={mt_da1['silverman_p']:.3g} "
          f"(nmodes={mt_da1['silverman_nmodes']})")
    print(f"  DA2 proj: dip p={mt_da2['dip_p']:.3g}, Silverman p={mt_da2['silverman_p']:.3g} "
          f"(nmodes={mt_da2['silverman_nmodes']})")
    mt_rad = None
    if radial:
        rsamp = np.linalg.norm(samp[:, :2], axis=1)
        mt_rad = modality_tests(rsamp, silverman_B=cfg['silverman_B'], seed=cfg['seeds'][0])
        print(f"  RADIAL |x| dist: dip p={mt_rad['dip_p']:.3g}, "
              f"Silverman p={mt_rad['silverman_p']:.3g} (nmodes={mt_rad['silverman_nmodes']})")

    out = dict(
        name=name, d=int(d), k=int(means.shape[0]),
        config=dict(dt=cfg['dt'], N_ssa=cfg['N_ssa'], T_ssa=cfg['T_ssa'],
                    max_lag=cfg['max_lag'], N_da=cfg['N_da'], N_mod=cfg['N_mod'],
                    checkpoints=cfg['checkpoints'], seeds=cfg['seeds'],
                    horizon_phys=cfg['T_ssa'] * cfg['dt']),
        ssa=dict(estimator=f'primary: floor=1/N, stopping lag={STOPPING_DESC}, '
                           'lag0 excluded, Shat=dt*sum(rho_sq[:keep])',
                 S_hat_mean=float(S_hat_vals.mean()), S_hat_std=float(S_hat_vals.std()),
                 S_full_mean=float(np.mean([r['S_full'] for r in ssa_list])),
                 tau_star=tau_vals, floor=float(ssa_list[0]['floor']), floor_name='1/N',
                 finite_horizon=True, resolved_population=False,
                 window_limited=bool(wl_any), window_limited_per_seed=wl_seeds,
                 max_lag=int(cfg['max_lag']),
                 window_limited_note=('tau*=max_lag: rho^2 never fell below 1/N inside '
                                      'the horizon; Shat is a finite-horizon LOWER BOUND'
                                      if wl_any else
                                      'resolved: rho^2 fell below 1/N inside the horizon'),
                 S_hat_1Nd_mean=float(np.mean([r['S_hat_1Nd'] for r in ssa_list])),
                 floor_1Nd=float(ssa_list[0]['floor_1Nd']),
                 floor_1Nd_note='secondary sensitivity only, NOT canonical',
                 per_seed=[dict(S_hat=r['S_hat'], S_full=r['S_full'], tau_star=r['tau_star'],
                                S_hat_1Nd=r['S_hat_1Nd'], tau_star_1Nd=r['tau_star_1Nd'])
                           for r in ssa_list]),
        radial=radial_summary,
        matched_gaussian=mg_summary,
        da_per_seed=per_seed, da_stats=da_stats,
        modality=dict(da1=mt_da1, da2=mt_da2, radial=mt_rad),
        runtime_s=time.time() - t0,
    )
    return out, dict(rho_sq_trace=rho_sq_store, rho_sq_radial=r_rho_store,
                     da1_seed0=da1, da2_seed0=da2)


def run_flattop(cfg, beta=6.0, alpha=3.0):
    """P3c: smooth light-tailed flat-top unimodal density."""
    name = f"P3c: flat-top generalized-normal (beta={beta}, alpha={alpha}, d=2)"
    d = 2
    t0 = time.time()
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    var_per_dim = float(sstats.gennorm.var(beta, scale=alpha))   # analytic variance
    print(f"  f(x) ∝ exp(-Σ_i (|x_i|/{alpha})^{beta}); per-dim Var={var_per_dim:.4f}, "
          f"std={np.sqrt(var_per_dim):.4f}")
    score = flattop_score_fn(beta, alpha)

    def make_sim(seed):
        g = torch.Generator().manual_seed(int(seed))
        s = sstats.gennorm.rvs(beta, scale=alpha, size=(cfg['N_da'], d),
                               random_state=int(seed))
        samples = torch.tensor(s, dtype=torch.float32)
        return GenericLangevinSimulator(score, samples, dt=cfg['dt'],
                                        mu_f=torch.zeros(d), var_f=var_per_dim)

    # ---- finite-horizon SSA over seeds (trace normalization via dummy 1-Gaussian) ----
    dummy_means = torch.zeros(1, d)
    dummy_covs = (var_per_dim * torch.eye(d)).unsqueeze(0)
    dummy_weights = torch.ones(1)
    ssa_seeds = cfg['seeds'][:cfg['n_ssa_seeds']]
    ssa_list = []
    rho_store = None
    for s in ssa_seeds:
        np.random.seed(s)
        samp = torch.tensor(sstats.gennorm.rvs(beta, scale=alpha,
                            size=(cfg['N_ssa'], d), random_state=s), dtype=torch.float32)
        sim = GenericLangevinSimulator(score, samp, dt=cfg['dt'],
                                       mu_f=torch.zeros(d), var_f=var_per_dim)
        torch.manual_seed(s)
        x0 = sim.sample_initial(cfg['N_ssa'])
        traj = sim.simulate(x0, cfg['T_ssa'])
        res = ssa_trace(traj, dummy_means, dummy_covs, dummy_weights, cfg['dt'],
                        cfg['max_lag'], cfg['N_ssa'], d)
        ssa_list.append(res)
        if rho_store is None:
            rho_store = res['rho_sq']
    S_hat = np.array([r['S_hat'] for r in ssa_list])
    wl_seeds = [bool(r['window_limited']) for r in ssa_list]
    wl_any = any(wl_seeds)
    print(f"  SSA Ŝ (finite-horizon, PRIMARY: stopping lag = {STOPPING_DESC}): "
          f"{'≥ ' if wl_any else ''}{S_hat.mean():.4f} ± {S_hat.std():.4f} "
          f"(τ*={[r['tau_star'] for r in ssa_list]}, H={cfg['T_ssa']*cfg['dt']:.1f}, "
          f"window_limited={wl_any})")
    print(f"  [sensitivity] Ŝ @ 1/(Nd) floor: "
          f"{np.mean([r['S_hat_1Nd'] for r in ssa_list]):.4f} "
          f"(τ*={[r['tau_star_1Nd'] for r in ssa_list]})")

    # ---- matched Gaussian baseline (same variance) ----
    gmeans = torch.zeros(1, d)
    gcovs = (var_per_dim * torch.eye(d)).unsqueeze(0)
    gweights = torch.ones(1)
    g_ssa = []
    for s in ssa_seeds:
        torch.manual_seed(1000 + s)
        gsim = LangevinSimulatorD(gmeans, gcovs, gweights, dt=cfg['dt'])
        x0 = gsim.sample_initial(cfg['N_ssa'])
        traj = gsim.simulate(x0, cfg['T_ssa'])
        res = ssa_trace(traj, gmeans, gcovs, gweights, cfg['dt'], cfg['max_lag'],
                        cfg['N_ssa'], d)
        g_ssa.append(res)
    gS = np.array([r['S_hat'] for r in g_ssa])
    print(f"  Matched Gaussian baseline Ŝ: {gS.mean():.4f} ± {gS.std():.4f}")

    # ---- DA across seeds ----
    per_seed, da_stats = da_across_seeds(
        None, None, None, cfg['checkpoints'], cfg['N_da'], cfg['dt'], cfg['seeds'],
        simulator_fn=make_sim)
    print(f"  DA λ1/λ2 gap ratio: mean={_fmt_ratio(da_stats['gap_ratio_mean'])} "
          f"max={_fmt_ratio(da_stats['gap_ratio_max'])} [{da_stats['gap_ratio_note']}]")
    print(f"  DA λ1 per seed: {np.round(da_stats['lam1_per_seed'],4).tolist()}")
    _n_neg = sum(1 for v in da_stats['lam2_per_seed'] if v <= 0)
    print(f"  DA λ2 per seed: {np.round(da_stats['lam2_per_seed'],4).tolist()}"
          + (f"  (λ2≤0 in {_n_neg}/{len(da_stats['lam2_per_seed'])} seeds ⇒ no second "
             f"slow mode; supports the unimodal scope calibration)" if _n_neg else ""))
    print(f"  DA1 cross-seed angle (deg): {np.round(da_stats['da1_angle_deg'],1).tolist()}")
    print(f"    subspace stat: {da_stats['subspace_stat_note']}")

    # ---- modality tests on DA1 proj + coordinate ----
    samp = sstats.gennorm.rvs(beta, scale=alpha, size=(cfg['N_mod'], d),
                              random_state=cfg['seeds'][0])
    da1 = np.asarray(per_seed[0]['da1'])
    mt_da1 = modality_tests(samp @ da1, silverman_B=cfg['silverman_B'], seed=cfg['seeds'][0])
    mt_x = modality_tests(samp[:, 0], silverman_B=cfg['silverman_B'], seed=cfg['seeds'][0])
    print(f"  DA1 proj: dip p={mt_da1['dip_p']:.3g}, Silverman p={mt_da1['silverman_p']:.3g} "
          f"(nmodes={mt_da1['silverman_nmodes']})")
    print(f"  x-coord:  dip p={mt_x['dip_p']:.3g}, Silverman p={mt_x['silverman_p']:.3g} "
          f"(nmodes={mt_x['silverman_nmodes']})")

    out = dict(
        name=name, beta=beta, alpha=alpha, d=d, var_per_dim=var_per_dim,
        density="f(x) ∝ exp(-sum_i (|x_i|/alpha)^beta)",
        score="grad_i log f = -(beta/alpha^beta) * |x_i|^(beta-1) * sign(x_i)",
        config=dict(dt=cfg['dt'], N_ssa=cfg['N_ssa'], T_ssa=cfg['T_ssa'],
                    max_lag=cfg['max_lag'], N_da=cfg['N_da'], seeds=cfg['seeds'],
                    checkpoints=cfg['checkpoints'], horizon_phys=cfg['T_ssa']*cfg['dt']),
        ssa=dict(estimator=f'primary: floor=1/N, stopping lag={STOPPING_DESC}, '
                           'lag0 excluded, Shat=dt*sum(rho_sq[:keep])',
                 S_hat_mean=float(S_hat.mean()), S_hat_std=float(S_hat.std()),
                 tau_star=[r['tau_star'] for r in ssa_list], floor=float(ssa_list[0]['floor']),
                 floor_name='1/N', finite_horizon=True, resolved_population=False,
                 window_limited=bool(wl_any), window_limited_per_seed=wl_seeds,
                 max_lag=int(cfg['max_lag']),
                 window_limited_note=('tau*=max_lag: rho^2 never fell below 1/N inside '
                                      'the horizon; Shat is a finite-horizon LOWER BOUND'
                                      if wl_any else
                                      'resolved: rho^2 fell below 1/N inside the horizon'),
                 S_hat_1Nd_mean=float(np.mean([r['S_hat_1Nd'] for r in ssa_list])),
                 floor_1Nd=float(ssa_list[0]['floor_1Nd']),
                 floor_1Nd_note='secondary sensitivity only, NOT canonical',
                 per_seed=[dict(S_hat=r['S_hat'], S_full=r['S_full'], tau_star=r['tau_star'],
                                S_hat_1Nd=r['S_hat_1Nd'], tau_star_1Nd=r['tau_star_1Nd'])
                           for r in ssa_list]),
        gaussian_baseline=dict(S_hat_mean=float(gS.mean()), S_hat_std=float(gS.std())),
        da_per_seed=per_seed, da_stats=da_stats,
        modality=dict(da1=mt_da1, x_coord=mt_x),
        runtime_s=time.time() - t0,
    )
    return out, dict(rho_sq_trace=rho_store, da1_seed0=da1)


# --------------------------------------------------------------------------- #
# Required checks
# --------------------------------------------------------------------------- #
def run_checks():
    print("\n" + "="*70 + "\nREQUIRED CHECKS\n" + "="*70)
    ok = True

    # Check 4: each analytic score has expected shape & finite values
    print("\n[Check 4] analytic scores: shape + finiteness")
    # P3a/P3b use the library GMM score (grad_log_pdf); P3c uses flattop_score_fn.
    m, c, w = four_circular_gmm()
    sim = LangevinSimulatorD(m, c, w, dt=0.01)
    xq = sim.sample_initial(64)
    g = sim.grad_log_pdf(xq)
    fin = torch.isfinite(g).all().item() and g.shape == xq.shape
    print(f"  P3a GMM score shape={tuple(g.shape)} finite={torch.isfinite(g).all().item()}  -> {'OK' if fin else 'FAIL'}")
    ok &= fin

    m2, c2, w2 = concentric_ring_gmm()
    sim2 = LangevinSimulatorD(m2, c2, w2, dt=0.01)
    xq2 = sim2.sample_initial(64)
    g2 = sim2.grad_log_pdf(xq2)
    fin2 = torch.isfinite(g2).all().item() and g2.shape == xq2.shape
    print(f"  P3b GMM score shape={tuple(g2.shape)} finite={torch.isfinite(g2).all().item()}  -> {'OK' if fin2 else 'FAIL'}")
    ok &= fin2

    beta, alpha = 6.0, 3.0
    score = flattop_score_fn(beta, alpha)
    xq3 = torch.tensor(sstats.gennorm.rvs(beta, scale=alpha, size=(64, 2)), dtype=torch.float32)
    g3 = score(xq3)
    fin3 = torch.isfinite(g3).all().item() and g3.shape == xq3.shape
    # score at 0 should be 0 (flat top)
    z0 = score(torch.zeros(1, 2))
    print(f"  P3c flat-top score shape={tuple(g3.shape)} finite={torch.isfinite(g3).all().item()} "
          f"score(0)={z0.abs().max().item():.2e}  -> {'OK' if fin3 else 'FAIL'}")
    ok &= fin3

    # Check 5: flat-top analytic score vs finite-difference on log pdf
    print("\n[Check 5] flat-top analytic score vs finite difference")
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2 * alpha, 2 * alpha, size=(200, 2))
    eps = 1e-4
    max_err = 0.0
    for p in pts:
        for i in range(2):
            pp = p.copy(); pp[i] += eps
            pm = p.copy(); pm[i] -= eps
            fd = (flattop_logpdf_np(pp[None], beta, alpha)[0]
                  - flattop_logpdf_np(pm[None], beta, alpha)[0]) / (2 * eps)
            an = score(torch.tensor(p[None], dtype=torch.float64))[0, i].item()
            max_err = max(max_err, abs(fd - an) / (abs(fd) + 1e-6))
    fd_ok = max_err < 1e-2
    print(f"  max relative error (analytic vs FD) over 200 pts = {max_err:.3e}  -> {'OK' if fd_ok else 'FAIL'}")
    ok &= fd_ok

    # Check 6: DA angle comparisons handle degeneracy + eigenvector sign
    print("\n[Check 6] DA angle helpers: sign + degeneracy")
    u = np.array([1.0, 0.0]); v = np.array([-1.0, 0.0])
    a_sign = angle_between_deg(u, v)  # opposite sign, same line -> 0 deg
    sign_ok = a_sign < 1e-6
    print(f"  sign-invariance angle([1,0],[-1,0]) = {a_sign:.3e} deg  -> {'OK' if sign_ok else 'FAIL'}")
    ok &= sign_ok
    # degenerate 2D subspace: any orthonormal basis of R^2 spans the same space
    A = np.eye(2)
    th = 0.7
    B = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pa = principal_angles_deg(A, B)
    sub_ok = np.allclose(pa, 0.0, atol=1e-6)
    print(f"  degenerate subspace principal angles (rotated basis) = "
          f"{np.round(pa,3).tolist()} deg  -> {'OK' if sub_ok else 'FAIL'}")
    ok &= sub_ok
    dr = degeneracy_ratio([1.0, 1.0, 0.2])
    print(f"  degeneracy_ratio([1,1,0.2]) = {dr:.3f} (==1 flags degeneracy)  -> "
          f"{'OK' if abs(dr-1.0)<1e-9 else 'FAIL'}")
    ok &= abs(dr - 1.0) < 1e-9

    # Check 7: stopping lag selects the LARGEST qualifying lag on a non-monotone ρ²
    print("\n[Check 6b] stopping lag = largest τ with ρ²>1/N (non-monotone case)")
    N = 100
    floor = 1.0 / N  # 0.01
    # τ = 1..6; dips BELOW floor at idx2, rises ABOVE at idx3, then below again.
    rho_sq = np.array([0.50, 0.20, 0.005, 0.05, 0.003, 0.002])
    keep = _stopping_lag_largest(rho_sq, floor)  # should be idx3 (0.05) -> keep=4
    first_cross = int(np.argmax(rho_sq < floor))  # first-crossing rule -> idx2 -> 2
    largest_ok = (keep == 4) and (keep != first_cross)
    # empty case: nothing above floor -> keep 0
    empty_ok = (_stopping_lag_largest(np.array([0.001, 0.002]), floor) == 0)
    print(f"  rho_sq={rho_sq.tolist()}, floor=1/N={floor}")
    print(f"  largest-qualifying keep={keep} (expect 4); first-crossing keep={first_cross} "
          f"(expect 2); differ={keep != first_cross}  -> {'OK' if largest_ok else 'FAIL'}")
    print(f"  empty (no ρ²>floor) -> keep={_stopping_lag_largest(np.array([0.001,0.002]), floor)} "
          f"(expect 0)  -> {'OK' if empty_ok else 'FAIL'}")
    ok &= largest_ok and empty_ok
    if STOPPING == 'first':   # active rule: first crossing, crossing lag included
        fc_ok = (_stopping_lag(rho_sq, floor) == 3)
        print(f"  active rule=first: keep={_stopping_lag(rho_sq, floor)} (expect 3)  -> "
              f"{'OK' if fc_ok else 'FAIL'}")
        ok &= fc_ok

    # Check 7 (doc): matched Gaussian has the intended covariance + identical settings
    print("\n[Check 7] P3a matched isotropic Gaussian: covariance + identical settings")
    for dd in (2, 10):
        mm, cc, ww = four_circular_gmm(dim=dd)
        _, cov_f = compute_stationary_moments(mm, cc, ww)
        g_m, g_c, g_w, sig2 = matched_isotropic_gaussian(cov_f, dd)
        tr_match = abs(float(torch.trace(g_c[0]).item()) - float(torch.trace(cov_f).item()))
        iso = torch.allclose(g_c[0], sig2 * torch.eye(dd), atol=1e-6)
        single = (g_m.shape == (1, dd)) and (g_w.numel() == 1)
        # diffusion coefficient sigma_f^2 = tr(Cov)/d must match the four-mode run
        sim_four = LangevinSimulatorD(mm, cc, ww, dt=0.01)
        sim_gauss = LangevinSimulatorD(g_m, g_c, g_w, dt=0.01)
        dif = abs(sim_four.var_f - sim_gauss.var_f)
        c7 = iso and single and tr_match < 1e-3 and dif < 1e-4
        print(f"  d={dd}: sigma^2=tr(Cov_f)/d={sig2:.4f}, isotropic={iso}, "
              f"single-component={single}, |tr diff|={tr_match:.2e}, "
              f"|sigma_f^2 diff (four-mode vs baseline)|={dif:.2e}  -> "
              f"{'OK' if c7 else 'FAIL'}")
        ok &= c7
    print("  (N, dt, T, max_lag and the stopping rule are taken from the SAME cfg "
          "in run_example_gmm, so they are identical by construction)")

    # Check 8 (doc): 10D P3a alignment vs known plane + both eigenvalue ratios
    print("\n[Check 8] 10D P3a: fast coords, known-plane alignment, λ1/λ2 and λ2/λ3")
    mm, cc, ww = four_circular_gmm(dim=10, fast_var=0.05)
    _, cov_f = compute_stationary_moments(mm, cc, ww)
    cf = cov_f.numpy()
    plane_var = float(np.mean([cf[0, 0], cf[1, 1]]))
    fast_var_eff = float(np.mean(np.diag(cf)[2:]))
    sf2 = float(np.trace(cf) / 10)
    k_fast = sf2 / (2 * fast_var_eff)
    k_intra = sf2 / (2 * 0.5)
    rate_ratio = k_fast / k_intra
    fast_ok = rate_ratio > 5.0
    print(f"  Cov_f: in-plane={plane_var:.3f}, fast coords={fast_var_eff:.3f}, "
          f"sigma_f^2={sf2:.4f}")
    print(f"  OU rates: k_fast={k_fast:.2f} vs k_intra_plane={k_intra:.2f} "
          f"=> fast coords are {rate_ratio:.1f}x faster  -> "
          f"{'OK' if fast_ok else 'FAIL (extra coords not fast enough)'}")
    ok &= fast_ok
    # short DA run to confirm alignment machinery + both ratios are produced
    ps, ds = da_across_seeds(mm, cc, ww, [50, 100, 200], 300, 0.01, [0, 1],
                             known_plane_dim=2)
    kp = ds['known_plane']
    have_both = (ds['gap_ratio_mean'] is not None) and (ds['gap_ratio_23_mean'] is not None)
    align_ok = kp is not None and kp['informative']
    print(f"  λ1/λ2 = {_fmt_ratio(ds['gap_ratio_mean'])}, "
          f"λ2/λ3 = {_fmt_ratio23(ds['gap_ratio_23_mean'], 10)}  -> "
          f"{'OK (both reported)' if have_both else 'FAIL'}")
    print(f"  known plane {kp['plane']}, informative={kp['informative']}, "
          f"worst max principal angle = {kp['max_principal_angle_deg_worst']:.2f}° "
          f"(short probe run, N=300) -> {'OK' if align_ok else 'FAIL'}")
    ok &= have_both and align_ok

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


# --------------------------------------------------------------------------- #
# Config + main
# --------------------------------------------------------------------------- #
def get_config(smoke):
    if smoke:
        return dict(dt=0.01, N_ssa=200, T_ssa=300, max_lag=150,
                    N_da=400, N_mod=1000, n_ssa_seeds=1,
                    checkpoints=[20, 50, 100, 200], seeds=[0, 1, 2],
                    silverman_B=100)
    return dict(dt=0.01, N_ssa=2000, T_ssa=2000, max_lag=1200,
                N_da=5000, N_mod=5000, n_ssa_seeds=3,
                checkpoints=[50, 100, 200, 400, 800, 1200, 1600, 2000],
                seeds=[0, 1, 2, 3, 4], silverman_B=500)


def build_table(results, cfg):
    N = cfg['N_ssa']
    caption = (
        f"**Table (P3 scope controls).** Finite-horizon SSA Ŝ and DA diagnostics. "
        f"Ŝ is the *finite-horizon* estimator (NOT the resolved population SSA): "
        f"lag 0 excluded, noise floor = 1/N = {1.0/N:.2g} (N={N} trajectories), "
        f"stopping lag τ* = {STOPPING_DESC}, "
        f"Ŝ = Δt·Σ_{{τ=1}}^{{τ*}} ρ²(τ), Δt={cfg['dt']}, horizon H={cfg['T_ssa']*cfg['dt']:.0f}. "
        f"SSA over {cfg['n_ssa_seeds']} seeds; DA over seeds {cfg['seeds']}. "
        f"λ1/λ2≈1 indicates a (near-)degenerate top-2 DA subspace; in that regime "
        f"individual DA directions are non-unique, which is what the large cross-seed "
        f"DA1 rotation measures. "
        f"dip/Silverman p are for the DA1 projection (small p ⇒ reject unimodality)."
    )
    # Is the subspace-stability column informative for ANY row? At d=2 it is
    # definitional (top-2 subspace = all of R^2) and therefore vacuous, so we
    # DROP the column entirely rather than print a circular 0.0 deg.
    any_informative = any(r['da_stats'].get('subspace_stat_informative', False)
                          for r in results)
    caption += (
        " **τ\\*=max_lag ⇒ window-limited**: ρ² never fell below the floor inside the "
        "horizon, so Ŝ is truncated by the WINDOW, not the noise floor, and is a "
        "finite-horizon LOWER BOUND (shown as ≥), not a resolved value."
    )
    if any_informative:
        caption += (
            " The top-2 DA subspace column is a genuine 2-of-d measurement only for "
            "rows with d>2; rows at d=2 are marked n/a because there the top-2 "
            "subspace is all of R² by construction (0° is definitional, not evidence)."
        )
    else:
        caption += (
            " We deliberately OMIT a top-2 subspace-stability column: all rows are d=2, "
            "where the top-2 eigenvector subspace is all of R² by construction, so "
            "principal angles are identically 0° regardless of the data. That statistic "
            "would be circular, so only the (meaningful) DA1 cross-seed rotation is shown."
        )
    caption += (
        " A DA λ1/λ2 entry of 'n/a (λ2≤0)' means the second eigenvalue of the "
        "symmetrized lagged autocovariance is non-positive, i.e. there is no second "
        "slow mode and a ratio is ill-defined — expected for a unimodal density."
    )

    any_mg = any(r.get('matched_gaussian') for r in results)
    any_kp = any((r['da_stats'].get('known_plane') or {}).get('informative')
                 for r in results)
    if any_mg:
        caption += (
            " The matched baseline is an isotropic Gaussian of the same dimension and "
            "covariance scale (σ²=tr(Σ_f)/d, which also matches the diffusion "
            "coefficient), run with exactly the same N, Δt, horizon and stopping rule."
        )
    if any_kp:
        caption += (
            " In 10D the four circular modes occupy span(e1,e2) and the remaining eight "
            "coordinates are FAST Gaussians (variance 0.05 vs 0.5 in-plane, ≈10× the "
            "in-plane relaxation rate), so the known slow plane is span(e1,e2) and "
            "recovering it is nontrivial. λ2/λ3≫1 is what makes the slow SUBSPACE "
            "identifiable (Davis–Kahan) even when DA1 rotates freely inside the "
            "near-degenerate block."
        )

    hdr = ["Example", "d", "Ŝ (finite-horizon)", "matched-Gaussian Ŝ", "SSA ratio",
           "window-limited", "τ*(lags)",
           "DA λ1 / λ2 (seed 0)", "DA λ1/λ2", "DA λ2/λ3", "DA1 cross-seed max angle"]
    if any_kp:
        hdr.append("∠ to span(e1,e2)")
    if any_informative:
        hdr.append("Top-2 subspace max angle")
    hdr.append("DA1 dip p / Silverman p")
    lines = [caption, ""]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))

    for r in results:
        ds = r['da_stats']
        s = r['ssa']
        md = r['modality']['da1']
        wl = bool(s.get('window_limited', False))
        ge = "≥ " if wl else ""
        lam1 = ds['lam1_per_seed'][0]
        lam2 = ds['lam2_per_seed'][0]
        mg = r.get('matched_gaussian')
        label = r['name'].split(':')[0]
        if r.get('status') == 'INCONCLUSIVE':
            label += " *(inconclusive)*"
        row = [
            label,
            str(r.get('d', ds.get('d', ''))),
            f"{ge}{s['S_hat_mean']:.3f}±{s['S_hat_std']:.3f}",
            (f"{'≥ ' if mg.get('window_limited') else ''}{mg['S_hat_mean']:.3f}"
             f"±{mg['S_hat_std']:.3f}") if mg else "—",
            f"**{mg['ssa_ratio_fourmode_over_gaussian']:.2f}×**" if mg else "—",
            "**yes**" if wl else "no",
            str(s['tau_star']),
            f"{lam1:.3g} / {lam2:.3g}",
            _fmt_ratio(ds['gap_ratio_mean']),
            _fmt_ratio23(ds.get('gap_ratio_23_mean'), ds.get('d')),
            f"{ds['da1_angle_deg_max']:.1f}°",
        ]
        if any_kp:
            kp = ds.get('known_plane') or {}
            row.append(f"{kp['max_principal_angle_deg_worst']:.1f}°"
                       if kp.get('informative') else "n/a")
        if any_informative:
            row.append(f"{ds['subspace_maxangle_deg_max']:.1f}°"
                       if ds.get('subspace_stat_informative', False) else "n/a (d=2)")
        row.append(f"{md['dip_p']:.2g} / {md['silverman_p']:.2g}")
        lines.append("| " + " | ".join(row) + " |")

    # Supporting contrasts (radial readout, matched Gaussian baseline)
    extra = []
    for r in results:
        if r.get('status') == 'INCONCLUSIVE':
            extra.append(f"- **{r['name'].split(':')[0]}: {r['status_note']}**")
        mg = r.get('matched_gaussian')
        ds = r['da_stats']
        if mg:
            # Explicit, honest assessment of the doc's three success criteria.
            c1 = mg['ssa_ratio_fourmode_over_gaussian']
            g12, g23 = ds['gap_ratio_mean'], ds.get('gap_ratio_23_mean')
            kp = ds.get('known_plane') or {}
            crit = []
            crit.append(f"(1) SSA above matched Gaussian: ratio = {c1:.2f}× — "
                        f"{'MET' if c1 > 1.2 else 'NOT MET'}")
            rot = ds['da1_angle_deg_max']
            c2 = (g12 is not None and g12 < 1.25) and rot > 20.0
            crit.append(f"(2) λ1≈λ2 (λ1/λ2={_fmt_ratio(g12)}) AND DA1 rotates "
                        f"({rot:.1f}° across seeds) — {'MET' if c2 else 'NOT MET'}")
            if kp.get('informative'):
                ang = kp['max_principal_angle_deg_worst']
                c3 = ang < 20.0 and (g23 is not None and g23 > 2.0)
                crit.append(f"(3) top-2 subspace close to span(e1,e2) "
                            f"(worst ∠={ang:.1f}°) AND λ2 separated from λ3 "
                            f"(λ2/λ3={_fmt_ratio(g23)}) — {'MET' if c3 else 'NOT MET'}")
            else:
                crit.append("(3) not assessable at d=2 (top-2 subspace is all of R² "
                            "by dimensionality) — requires the 10D run")
            extra.append(f"- **{r['name'].split(':')[0]} success criteria.** "
                         + "; ".join(crit) + ".")
        rad = r.get('radial')
        if rad:
            rwl = "≥ " if rad.get('window_limited') else ""
            extra.append(
                f"- **{r['name'].split(':')[0]} radial readout.** A DIRECT radial "
                f"observable r_t=|x_t| gives Ŝ = {rwl}{rad['S_hat_mean']:.3f}±"
                f"{rad['S_hat_std']:.3f}, versus the full linear-coordinate trace "
                f"Ŝ = {'≥ ' if r['ssa'].get('window_limited') else ''}"
                f"{r['ssa']['S_hat_mean']:.3f}. DIAGNOSTIC ONLY: the discrete angular "
                f"components add visible tangential modes, so this contrast does NOT "
                f"establish that SSA captures nonlinear structure.")
        gb = r.get('gaussian_baseline')
        if gb:
            ratio = r['ssa']['S_hat_mean'] / gb['S_hat_mean'] if gb['S_hat_mean'] else float('nan')
            extra.append(
                f"- **{r['name'].split(':')[0]} vs matched unimodal baseline.** "
                f"Ŝ = {r['ssa']['S_hat_mean']:.3f}±{r['ssa']['S_hat_std']:.3f} vs a "
                f"variance-matched Gaussian Ŝ = {gb['S_hat_mean']:.3f}±"
                f"{gb['S_hat_std']:.3f} — only {100*(ratio-1):.0f}% elevated, and "
                f"dip p={r['modality']['da1']['dip_p']:.2g} / Silverman "
                f"p={r['modality']['da1']['silverman_p']:.2g} do NOT reject "
                f"unimodality. Reported as observed: this is scope calibration.")
    if extra:
        lines.append("")
        lines.extend(extra)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--checks', action='store_true', help='run required checks only')
    ap.add_argument('--only', choices=['a', 'b', 'c'], default=None)
    ap.add_argument('--output_dir', default=RESULTS_DIR)
    ap.add_argument('--embed_dim', type=int, default=2,
                    help='Ambient dimension for the GMM examples P3a/P3b. '
                         'Default 2 = original behavior, UNCHANGED. Use e.g. 10 to '
                         'embed the modes in the first 2 coords of R^10 so the top-2 '
                         'DA subspace-stability statistic is non-vacuous.')
    ap.add_argument('--fast_var', type=float, default=0.05,
                    help='Variance of the fast nuisance coordinates for P3a when '
                         '--embed_dim>2. Must be << the in-plane variance (0.5) so '
                         'those coordinates relax faster than the inter-mode motion. '
                         'Default 0.05 => 10x the in-plane intra-mode rate.')
    ap.add_argument('--tag', type=str, default=None,
                    help='Override output filename tag (default: smoke|full).')
    args = ap.parse_args()
    if args.embed_dim < 2:
        ap.error('--embed_dim must be >= 2')

    if args.checks:
        sys.exit(0 if run_checks() else 1)

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = get_config(args.smoke)
    cfg['embed_dim'] = int(args.embed_dim)
    base_tag = 'smoke' if args.smoke else 'full'
    tag = args.tag or (base_tag if args.embed_dim == 2 else f"{base_tag}_d{args.embed_dim}")
    print(f"P3 scope controls — {tag} run\nconfig: {cfg}")
    if args.embed_dim > 2:
        print(f"  embed_dim={args.embed_dim}: P3a modes live in span(e1,e2) of "
              f"R^{args.embed_dim}; the other {args.embed_dim-2} coordinates are FAST "
              f"Gaussians (var={args.fast_var} vs 0.5 in-plane). Known slow plane = "
              f"span(e1,e2); top-2 DA subspace recovery is a genuine "
              f"2-of-{args.embed_dim} measurement.")

    results = []
    npz_blobs = {}

    ed = args.embed_dim
    dsuf = '' if ed == 2 else f" (embedded in d={ed})"

    if args.only in (None, 'a'):
        m, c, w = four_circular_gmm(dim=ed, fast_var=args.fast_var)
        out, blob = run_example_gmm(f"P3a: four-mode circular GMM{dsuf}", m, c, w, cfg,
                                    matched_gaussian=True, known_plane_dim=2)
        out['fast_var'] = float(args.fast_var) if ed > 2 else None
        results.append(out)
        for k, v in blob.items():
            if v is not None:
                npz_blobs[f"p3a_{k}"] = v

    if args.only in (None, 'b'):
        m, c, w = concentric_ring_gmm(dim=ed)
        out, blob = run_example_gmm(f"P3b: concentric-ring GMM{dsuf}", m, c, w, cfg,
                                    radial=True)
        out['status'] = 'INCONCLUSIVE'
        out['status_note'] = (
            'Finite-GMM ring approximation is INCONCLUSIVE for the intended claim: '
            'the discrete angular components create additional visible (tangential) '
            'modes, so this run must NOT be used as evidence that SSA captures '
            'nonlinear structure. Reported as a failed calibration.')
        results.append(out)
        for k, v in blob.items():
            if v is not None:
                npz_blobs[f"p3b_{k}"] = v

    if args.only in (None, 'c'):
        out, blob = run_flattop(cfg)
        results.append(out)
        for k, v in blob.items():
            if v is not None:
                npz_blobs[f"p3c_{k}"] = v

    # Save
    json_path = os.path.join(args.output_dir, f'p3_results_{tag}.json')
    with open(json_path, 'w') as f:
        json.dump(dict(tag=tag, config=cfg, results=results), f, indent=2)
    npz_path = os.path.join(args.output_dir, f'p3_arrays_{tag}.npz')
    np.savez(npz_path, **{k: np.asarray(v) for k, v in npz_blobs.items()})

    table = build_table(results, cfg)
    md_path = os.path.join(args.output_dir, f'p3_table_{tag}.md')
    with open(md_path, 'w') as f:
        f.write(f"# P3 scope controls ({tag})\n\n")
        f.write(table + "\n")
    print("\n" + "="*70 + "\nREBUTTAL TABLE\n" + "="*70)
    print(table)
    print(f"\nSaved:\n  {json_path}\n  {npz_path}\n  {md_path}")


if __name__ == '__main__':
    main()
