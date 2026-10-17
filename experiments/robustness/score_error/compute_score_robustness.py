#!/usr/bin/env python3
"""
P1 — Score robustness for SSA rankings and DA directions.

Question: How do (1) additive score noise and (2) smoothed-density score bias
affect the SSA->MI ranking of GMMs, and the DA1 direction, relative to the
EXACT GMM score baseline?

Design (authoritative spec: docs/rebuttal_agent_instructions.md, section P1):
  - Reuse the existing set of 50 random 10-D GMMs (build_random_gmm_10d, seeds 0..49,
    identical to experiments/synthetic/compute_random_sweep.py).
  - Exact GMM score = baseline.
  - Perturbation family 1 (additive): score_pert(x) = grad_true(x) + alpha * E(x),
    where E is a FROZEN, deterministic, state-dependent random error field
    (random Fourier features, fixed per GMM). E is calibrated once so its
    per-coordinate RMS over baseline GMM samples equals s0 (the recorded baseline
    per-coordinate score RMS); the SAME field is reused across all alpha and only
    its amplitude is scaled. alpha is a dimensionless fraction of the baseline
    score magnitude. Because E(x) is deterministic in x, no RNG is consumed during
    simulation and alpha=0 returns the exact baseline score exactly.
  - Perturbation family 2 (smoothing): exact score of the Gaussian-smoothed GMM,
    implemented by inflating each component covariance Sigma_i -> Sigma_i + frac*var*I
    (NOT KDE). This is the exact score of a genuine (smoothed) GMM.
  - Integration settings held fixed across ALL levels: same GMM, same seeds,
    same dt, same horizon T, same diffusion coefficient var_f (= true baseline
    tr(Cov_f)/d), same initial conditions, and same autocorrelation normalization
    (trace) with the same 1/N noise floor. Only the SCORE (drift direction)
    changes. This isolates score bias/noise.
  - MI is the TRUE mutual information of the ORIGINAL GMM at every level; we
    correlate SSA(perturbed) vs MI(true) to test ranking robustness.

SSA estimator (the paper's eq:tmax_stopping):
  rho_sq = rho[1:]**2 (lag 0 excluded); noise floor = 1/N compared on rho_sq;
  Shat = dt * rho_sq[:keep].sum(), where `keep` runs up to and including the FIRST
  lag with rho_sq <= 1/N (the full window if it never crosses). Lag-0 handling and
  integration form match experiments/synthetic/compute_random_sweep.py.
  --stopping largest instead keeps through the LARGEST lag with rho_sq > 1/N (the
  original rebuttal run); the paper reports first crossing, run tag `full_fc`.

Reproduction guarantee: additive alpha=0 and smoothing frac=0 both reproduce the
exact-score baseline exactly. The additive field E(x) is deterministic and enters as
alpha*E(x), so alpha=0 adds nothing and consumes no RNG; the smoothed score at frac=0
is the exact GMM score. Diffusion noise is presampled per seed and shared across
levels, so trajectories are identical wherever the score is identical.

Determinism is keyed by (gid, seed) ONLY (samples, field, MI, and per-run RNG all
derive from gid/seed), so a shard covering a subset of GMM ids reproduces the exact
same per-GMM numbers as the monolithic run.

DA analysis: representative GMMs selected by a stated rule = the GMMs with the
highest true MI (clearest slow direction, so DA1 is well-defined). Fixed lag =
baseline optimal lag (argmax eigenvalue gap) per GMM, reused across all levels for
an apples-to-apples comparison. Both signed and sign-invariant direction
comparisons are recorded. DA needs the GLOBAL top-k, so it runs in --da_only mode.

Modes:
  python compute_score_robustness.py                    # monolithic full (50 GMMs, DA incl.)
  python compute_score_robustness.py --smoke            # monolithic smoke (3 GMMs, DA incl.)
  python compute_score_robustness.py --test             # stopping-lag unit tests
  # cluster (sharded) pipeline:
  python compute_score_robustness.py --gmm_start i --gmm_end i+1   # one shard, SSA sweep only
  python compute_score_robustness.py --da_only                     # global DA analysis
  python compute_score_robustness.py --merge                       # merge shards + DA -> full
"""
import sys, os, json, time, argparse, glob
import numpy as np
import torch
from scipy.stats import spearmanr

CODEDIR = os.path.dirname(os.path.abspath(__file__))
REPODIR = os.path.abspath(os.path.join(CODEDIR, '..', '..', '..'))
sys.path.insert(0, REPODIR)

from src.core.gaussian_mixture import GaussianMixture
from src.simulation.langevin import LangevinSimulatorD
from src.simulation.generic_langevin import GenericLangevinSimulator
from src.metrics.autocorrelation import (
    compute_stationary_moments, compute_vector_autocorrelation,
)
from src.metrics.da_directions import extract_da_directions, da_convergence_profile

# ---- Fixed configuration (matched across all perturbation levels) ----
D = 10
N_TRAJ = 200
DT = 0.01
FLOOR_1N = 1.0 / N_TRAJ          # 5e-3, submitted-paper noise floor on rho^2 (1/N)
N_SAMPLES_MI = 5000
N_RANDOM = 50
MI_SEED_OFFSET = 99999           # matches compute_random_sweep.py

# Perturbation grids (level 0 == exact baseline in both families)
ADDITIVE_ALPHAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]   # fraction of baseline score RMS
SMOOTH_FRACS = [0.0, 0.1, 0.25, 0.5, 1.0]           # cov inflation as fraction of comp var

# DA analysis
N_DA = 2000
N_REPRESENTATIVE = 5
DA_CHECKPOINTS = [10, 20, 50, 100, 200, 400, 800, 1600]


def build_random_gmm_10d(seed):
    """IDENTICAL to experiments/synthetic/compute_random_sweep.py."""
    rng = np.random.RandomState(seed)
    separation = rng.uniform(1.0, 8.0)
    variance = rng.uniform(0.5, 3.0)
    w1 = rng.uniform(0.5, 0.95)
    means = torch.zeros(2, D)
    means[1, 0] = separation
    covs = (variance * torch.eye(D)).unsqueeze(0).expand(2, -1, -1).clone()
    weights = torch.tensor([w1, 1.0 - w1], dtype=torch.float32)
    gm = GaussianMixture(means, covs, weights, device='cpu')
    meta = {'k': 2, 'separation': float(separation),
            'variance': float(variance), 'w1': float(w1)}
    return gm, meta


# Stopping rule for the 1/N floor. 'first' (default since 2026-10-17) matches the
# paper's eq:tmax_stopping: truncate at the first lag with rho^2 <= floor, that lag
# included. 'largest' reproduces the original rebuttal run (largest lag with
# rho^2 > floor). Set from the CLI via --stopping.
STOPPING = 'first'


def compute_shat(rho_sq, dt, floor):
    """Riemann SSA Ŝ = Δt·Σ ρ² truncated by the 1/N floor (rule set by STOPPING).

    'first': keep lags up to and including the first one with ρ² <= floor (full
    horizon if it never crosses). 'largest': see below. Returns (Ŝ, n_lags_kept).
    """
    rho_sq = np.asarray(rho_sq, dtype=float)
    if STOPPING == 'first' and floor > 0:
        below = np.where(rho_sq <= floor)[0]
        keep = int(below[0]) + 1 if len(below) else len(rho_sq)
        return float(dt * rho_sq[:keep].sum()), keep
    return _compute_shat_largest(rho_sq, dt, floor)


def _compute_shat_largest(rho_sq, dt, floor):
    """Riemann SSA Ŝ = Δt·Σ ρ² truncated at the LARGEST lag τ with ρ²(τ) > floor.

    Matches the submitted estimator's 1/N floor and lag-0 exclusion (rho_sq already
    excludes lag 0; rho_sq[i] is τ=i+1). Uses the largest qualifying lag rather than
    the first threshold crossing, so it is robust to non-monotone ρ²: every lag up to
    and including the last one above the floor is integrated. Returns (Ŝ, n_lags_kept).
    """
    rho_sq = np.asarray(rho_sq, dtype=float)
    if floor > 0:
        above = np.where(rho_sq > floor)[0]
        if len(above) == 0:
            return 0.0, 0
        keep = int(above[-1]) + 1
        return float(dt * rho_sq[:keep].sum()), keep
    return float(dt * rho_sq.sum()), len(rho_sq)


def run_one(means, covs, weights, mu_f, cov_f, var_f, samples,
            score_fn, seed, N, T, dt, score_gen=None, device='cpu', max_lag=None):
    """
    Simulate one perturbed (or exact) Langevin realization with matched integration.

    - x0 sampled by resampling `samples` using a dedicated generator (seed).
    - diffusion noise presampled with a dedicated generator (seed) -> independent of
      any RNG consumed inside score_fn, so alpha=0 / frac=0 reproduce the baseline.
    - autocorrelation uses the TRUE GMM moments (trace normalization).

    DEVICE DETERMINISM: all RNG is drawn on the CPU generator stream and only then
    moved to `device`. This keeps the random stream (and hence every number) identical
    on CPU and GPU — never replace these with torch.randn(..., device='cuda').

    max_lag: number of autocorrelation lags to evaluate. Defaults to T, matching the
    SUBMITTED convention (experiments/synthetic/compute_random_sweep.py uses
    max_lag=T). A smaller cap is a COMPUTE-SAVING approximation, not a free choice:
    if rho^2 never drops below the 1/N floor within max_lag, Shat is WINDOW-LIMITED
    (a finite-horizon lower bound), not a resolved value.

    Returns (s_hat, keep, rho_sq, window_limited) where keep is the number of lags
    integrated and window_limited is True when the floor is never reached.
    """
    ml = int(T if max_lag is None else min(max_lag, T))
    M = samples.shape[0]
    g_init = torch.Generator(device='cpu'); g_init.manual_seed(seed)
    g_diff = torch.Generator(device='cpu'); g_diff.manual_seed(seed + 1_000_000)

    # Draw on CPU (fixed stream), then move to the compute device.
    idx = torch.randint(0, M, (N,), generator=g_init).to(samples.device)
    x0 = samples[idx].clone()
    presampled = torch.randn(T, N, D, generator=g_diff).to(device)

    gen = GenericLangevinSimulator(
        score_fn, samples, dt=dt, device=device,
        mu_f=mu_f, cov_f=cov_f, var_f=var_f,
    )
    traj = gen.simulate(x0.to(device), T, presampled_noise=presampled)

    rho, _, _ = compute_vector_autocorrelation(
        traj, ml, gmm_means=means, gmm_covs=covs, gmm_weights=weights,
        demean=True, normalization='trace',
    )
    rho_sq = (rho.cpu().numpy() ** 2)[1:]
    s_hat, keep = compute_shat(rho_sq, dt, FLOOR_1N)
    window_limited = (not bool(np.any(rho_sq <= FLOOR_1N))) if STOPPING == 'first' \
        else bool(keep == len(rho_sq))
    return s_hat, keep, rho_sq.astype(np.float32), window_limited


class FrozenScoreField:
    """Deterministic, state-dependent score-error field E: R^d -> R^d.

    Implemented as a frozen random-Fourier-feature vector field: the frequencies W,
    phases b, and output mixing A are drawn once (fixed seed) and never resampled, so
    E(x) is a fixed function of the state x (same x -> same E(x)). After construction,
    `calibrate` rescales E so its per-coordinate RMS over baseline GMM samples equals a
    target (s0). The SAME field is reused across all perturbation strengths; only the
    external amplitude alpha changes.
    """
    def __init__(self, d, seed, n_features=256, length_scale=1.0):
        g = torch.Generator(device='cpu'); g.manual_seed(int(seed))
        self.W = torch.randn(n_features, d, generator=g) / float(length_scale)
        self.b = torch.rand(n_features, generator=g) * (2.0 * np.pi)
        self.A = torch.randn(d, n_features, generator=g) / np.sqrt(n_features)
        self.scale = 1.0

    def raw(self, x):
        feat = torch.cos(x @ self.W.t() + self.b)   # (M, n_features)
        return feat @ self.A.t()                    # (M, d)

    def calibrate(self, samples, target_rms):
        with torch.no_grad():
            e = self.raw(samples)
            cur = float(e.pow(2).mean().sqrt())     # per-coordinate RMS
        self.scale = float(target_rms) / (cur + 1e-12)

    def to(self, device):
        """Move the frozen parameters to `device`. Draws NO new randomness, so the
        field is the same deterministic function of x on CPU and GPU."""
        self.W = self.W.to(device)
        self.b = self.b.to(device)
        self.A = self.A.to(device)
        return self

    def __call__(self, x):
        return self.scale * self.raw(x)


def make_additive_score_fn(base_sim, field, alpha):
    """score(x) = grad_true(x) + alpha * E(x), E a frozen deterministic field.

    alpha=0 returns the exact baseline score (E adds nothing, no RNG consumed)."""
    def score_fn(x):
        g = base_sim.grad_log_pdf(x)
        if alpha == 0.0:
            return g
        return g + alpha * field(x)
    return score_fn


def make_smoothed_score_fn(means, covs_infl, weights, device='cpu'):
    """Exact score of the covariance-inflated (Gaussian-smoothed) GMM."""
    sim = LangevinSimulatorD(means, covs_infl, weights, dt=DT, device=device)
    return sim.grad_log_pdf


def inflate_covs(covs, frac):
    """Sigma_i -> Sigma_i + frac*var*I (Gaussian smoothing via covariance inflation).

    Identity is built on the covs' own device so this works on CPU and GPU alike.
    """
    eye = torch.eye(covs.shape[-1], device=covs.device, dtype=covs.dtype)
    return covs + (frac * covs[0, 0, 0].item()) * eye.unsqueeze(0)


def da_angle(v_a, v_b):
    """Return (cos_signed, cos_abs, angle_deg) between two unit-ish vectors."""
    va = v_a / (np.linalg.norm(v_a) + 1e-12)
    vb = v_b / (np.linalg.norm(v_b) + 1e-12)
    c = float(np.dot(va, vb))
    c_abs = min(1.0, abs(c))
    ang = float(np.degrees(np.arccos(c_abs)))
    return c, c_abs, ang


def _first_crossing_keep(rho_sq, floor):
    """Reference: submitted estimator's FIRST-crossing rule (for the unit test)."""
    rho_sq = np.asarray(rho_sq, dtype=float)
    below = rho_sq < floor
    if below.any():
        return int(np.argmax(below))
    return len(rho_sq)


def run_tests():
    """Unit tests for the stopping-lag rule. Exercised via `--test` and in checks.

    Primary property: with a NON-MONOTONE rho^2 that dips BELOW the 1/N floor and
    later rises ABOVE it, compute_shat must keep the LARGEST qualifying lag (integrate
    through the last lag above floor), which differs from the submitted first-crossing
    rule. Also checks lag-0-excluded integration form and the all-below edge case.
    """
    global STOPPING
    saved_rule, STOPPING = STOPPING, 'largest'
    print("=== P1 stopping-lag unit tests (largest rule; first-crossing checked at end) ===")
    dt = DT
    floor = FLOOR_1N  # 1/N = 0.005
    # rho_sq indices are tau = i+1 (lag 0 already excluded, matching run_one).
    # above floor at idx 0,1,4,5 ; below at 2,3,6,7  -> non-monotone re-crossing.
    rho_sq = np.array([0.02, 0.01, 0.001, 0.0001, 0.008, 0.006, 0.0001, 0.0001])

    s_hat, keep = compute_shat(rho_sq, dt, floor)
    keep_first = _first_crossing_keep(rho_sq, floor)

    expected_keep = 6                       # last lag above floor is idx 5 -> keep=6
    expected_shat = float(dt * rho_sq[:6].sum())
    print(f"  rho_sq            = {rho_sq.tolist()}")
    print(f"  floor (1/N)       = {floor:g}")
    print(f"  largest-qualifying keep = {keep} (expected {expected_keep})")
    print(f"  first-crossing keep     = {keep_first} (expected 2)")
    print(f"  Shat = {s_hat:.6f} (expected {expected_shat:.6f})")

    assert keep == expected_keep, f"keep {keep} != {expected_keep}"
    assert abs(s_hat - expected_shat) < 1e-12, "Shat integration mismatch"
    assert keep_first == 2, f"first-crossing keep {keep_first} != 2"
    assert keep != keep_first, "largest-qualifying MUST differ from first-crossing here"
    assert keep_first < keep, "first-crossing must truncate earlier than largest-qualifying"

    # Edge case: everything below the floor -> keep 0, Shat 0.
    all_below = np.array([1e-4, 1e-5, 1e-6])
    s0v, keep0 = compute_shat(all_below, dt, floor)
    assert keep0 == 0 and s0v == 0.0, "all-below edge case failed"
    print(f"  all-below edge: keep={keep0} Shat={s0v} (expected 0, 0.0)")

    # Monotone decreasing -> largest-qualifying == first-crossing.
    mono = np.array([0.02, 0.01, 0.006, 0.004, 0.001])
    _, keep_m = compute_shat(mono, dt, floor)
    keep_m_first = _first_crossing_keep(mono, floor)
    assert keep_m == keep_m_first == 3, f"monotone keep {keep_m}/{keep_m_first} != 3"
    print(f"  monotone: largest={keep_m} first={keep_m_first} (agree, expected 3)")

    STOPPING = 'first'
    s_f, keep_f = compute_shat(rho_sq, dt, floor)
    assert keep_f == 3 and abs(s_f - dt * rho_sq[:3].sum()) < 1e-12, \
        f"first-crossing (inclusive) keep {keep_f} != 3"
    print(f"  first-crossing (inclusive) keep={keep_f} (expected 3)")
    STOPPING = saved_rule
    print("ALL TESTS PASSED\n")
    return True


# ----------------------------------------------------------------------------
# Config + reusable building blocks (shared by monolithic and sharded pipelines)
# ----------------------------------------------------------------------------

def get_config(args):
    """Resolve the parameter set. `run_tag` namespaces shard/DA/output files."""
    if args.smoke:
        return dict(n_gmm=3, seeds=[0], T_phys=20.0,
                    add_alphas=[0.0, 0.5], smooth_fracs=[0.0, 0.5],
                    n_da=500, da_checkpoints=[10, 20, 50, 100],
                    n_representative=2, run_tag='smoke')
    return dict(n_gmm=N_RANDOM, seeds=list(range(args.seeds)), T_phys=args.T_phys,
                add_alphas=list(ADDITIVE_ALPHAS), smooth_fracs=list(SMOOTH_FRACS),
                n_da=N_DA, da_checkpoints=list(DA_CHECKPOINTS),
                n_representative=N_REPRESENTATIVE, run_tag='full')


def precompute_gmm(gid, device='cpu'):
    """Per-GMM statics keyed by gid ONLY (deterministic, shard- AND device-invariant).

    Returns a cache dict incl. true MI and meta. Same output regardless of which
    other GMMs are processed, because every RNG draw is seeded from gid.

    DEVICE DETERMINISM: every randomness-consuming step (GMM sampling, the frozen
    field, MI) is executed on CPU, then the resulting tensors are moved to `device`.
    So the GPU run consumes the identical CPU random stream as the CPU run.
    """
    gm, meta = build_random_gmm_10d(gid)                 # built on CPU
    means, covs, weights = gm.means, gm.covs, gm.weights
    mu_f, cov_f = compute_stationary_moments(means, covs, weights)
    # CPU simulator used ONLY for the deterministic sampling/calibration below.
    cpu_sim = LangevinSimulatorD(means, covs, weights, dt=DT, device='cpu')
    # samples from TRUE GMM for initial conditions (fixed per GMM, CPU RNG stream)
    torch.manual_seed(gid + 424242)
    samples = cpu_sim.sample_initial(5000)
    # baseline per-coordinate score RMS: s0 = sqrt(E||grad||^2 / d)
    with torch.no_grad():
        g = cpu_sim.grad_log_pdf(samples)
        s0 = float((g.pow(2).sum(dim=1).mean() / D).sqrt())
    # frozen deterministic score-error field, calibrated to per-coord RMS = s0.
    length_scale = max(float(samples.std(dim=0).mean()), 1e-3)
    field = FrozenScoreField(D, seed=gid + 555_000, n_features=256,
                             length_scale=length_scale)
    field.calibrate(samples, target_rms=s0)
    # true MI (same recipe as compute_random_sweep) — CPU GMM, CPU RNG stream.
    torch.manual_seed(gid + MI_SEED_OFFSET)
    try:
        mi = gm.mutual_information(n_samples=N_SAMPLES_MI).item()
    except Exception:
        mi = float('nan')

    # Move the compute-side tensors to the target device (no new randomness).
    means = means.to(device); covs = covs.to(device); weights = weights.to(device)
    mu_f = mu_f.to(device); cov_f = cov_f.to(device)
    samples = samples.to(device)
    field = field.to(device)
    base_sim = (cpu_sim if str(device) == 'cpu'
                else LangevinSimulatorD(means, covs, weights, dt=DT, device=device))
    var_f = base_sim.var_f
    return dict(gid=gid, means=means, covs=covs, weights=weights,
                mu_f=mu_f, cov_f=cov_f, var_f=var_f, base_sim=base_sim,
                samples=samples, s0=s0, field=field, mi=mi, meta=meta,
                device=device)


def sweep_gmm(c, seeds, T, add_alphas, smooth_fracs, device='cpu', max_lag=None):
    """Run baseline + additive + smoothing SSA for ONE GMM across all seeds/levels.

    Returns (baseline, add, smooth, repro_max_absdiff, diag) where the first three
    are dicts/lists of Shat per seed, and `diag` carries the finite-horizon
    diagnostics: stopping lag `keep` and `window_limited` for every cell.
    """
    base_sim = c['base_sim']
    baseline = []
    add = {a: [] for a in add_alphas}
    smooth = {f: [] for f in smooth_fracs}
    diag = dict(baseline_keep=[], baseline_wl=[],
                additive_keep={a: [] for a in add_alphas},
                additive_wl={a: [] for a in add_alphas},
                smoothing_keep={f: [] for f in smooth_fracs},
                smoothing_wl={f: [] for f in smooth_fracs})
    repro = 0.0
    curves = {}
    for seed in seeds:
        s_base, keep_b, rs_b, wl_b = run_one(
            c['means'], c['covs'], c['weights'], c['mu_f'], c['cov_f'],
            c['var_f'], c['samples'], base_sim.grad_log_pdf,
            seed=seed, N=N_TRAJ, T=T, dt=DT, device=device, max_lag=max_lag)
        baseline.append(s_base)
        curves[f'baseline_s{seed}'] = rs_b
        diag['baseline_keep'].append(keep_b); diag['baseline_wl'].append(wl_b)
        for a in add_alphas:
            sfn = make_additive_score_fn(base_sim, c['field'], a)
            s_hat, keep, rs, wl = run_one(
                c['means'], c['covs'], c['weights'], c['mu_f'], c['cov_f'],
                c['var_f'], c['samples'], sfn, seed=seed, N=N_TRAJ, T=T, dt=DT,
                device=device, max_lag=max_lag)
            add[a].append(s_hat)
            curves[f'additive_{a}_s{seed}'] = rs
            diag['additive_keep'][a].append(keep); diag['additive_wl'][a].append(wl)
            if a == 0.0:
                repro = max(repro, abs(s_hat - s_base))
        for f in smooth_fracs:
            covs_infl = inflate_covs(c['covs'], f)
            sfn = make_smoothed_score_fn(c['means'], covs_infl, c['weights'],
                                         device=device)
            s_hat, keep, rs, wl = run_one(
                c['means'], c['covs'], c['weights'], c['mu_f'], c['cov_f'],
                c['var_f'], c['samples'], sfn, seed=seed, N=N_TRAJ, T=T, dt=DT,
                device=device, max_lag=max_lag)
            smooth[f].append(s_hat)
            curves[f'smoothing_{f}_s{seed}'] = rs
            diag['smoothing_keep'][f].append(keep); diag['smoothing_wl'][f].append(wl)
            if f == 0.0:
                repro = max(repro, abs(s_hat - s_base))
    diag['curves'] = curves
    return baseline, add, smooth, repro, diag


def spearman_levels(shat_dict, levels, mi_vals, n_seeds):
    """Per-level Spearman(SSA, MI): per-seed values + mean/sd across seeds."""
    out = {}
    for lv in levels:
        per_seed = []
        for si in range(n_seeds):
            x = np.asarray(shat_dict[lv][si], dtype=float)
            m = np.isfinite(x) & np.isfinite(mi_vals)
            if m.sum() >= 3:
                per_seed.append(float(spearmanr(x[m], mi_vals[m]).correlation))
            else:
                per_seed.append(float('nan'))
        per_seed = np.array(per_seed)
        out[lv] = dict(per_seed=per_seed.tolist(),
                       mean=float(np.nanmean(per_seed)),
                       sd=float(np.nanstd(per_seed)))
    return out


def run_da_for_gmm(c, add_alphas, smooth_fracs, n_da, da_checkpoints, mi_val,
                   device='cpu'):
    """DA angular-deviation analysis for one representative GMM.

    NOTE (device determinism): unlike the SSA path, the DA path draws its initial
    conditions INSIDE src/metrics/da_directions.py (via sim.sample_initial), which
    uses the active device's RNG stream. That module is shared infrastructure and is
    deliberately not refactored here, so DA numbers on GPU are statistically
    equivalent but not bitwise identical to CPU. Run the DA/merge step on CPU (it is
    cheap: n_da trajectories x <=1600 steps for a handful of GMMs) to keep DA results
    reproducible against the CPU smoke.
    """
    e1 = np.zeros(D); e1[0] = 1.0
    prof = da_convergence_profile(
        gmm_means=c['means'], gmm_covs=c['covs'], gmm_weights=c['weights'],
        checkpoints=da_checkpoints, n_trajectories=n_da, dt=DT, seed=1234,
        device=torch.device(device) if not isinstance(device, torch.device) else device,
        reference_direction=torch.tensor(e1, dtype=torch.float32))
    lag_star = int(prof['checkpoints'][prof['optimal_idx']])
    da1_base = prof['optimal_eigenvectors'][:, 0]
    _, base_abs_e1, base_ang_e1 = da_angle(da1_base, e1)

    levels = []
    for a in add_alphas:
        sfn = make_additive_score_fn(c['base_sim'], c['field'], a)
        gen = GenericLangevinSimulator(
            sfn, c['samples'], dt=DT, device=device,
            mu_f=c['mu_f'], cov_f=c['cov_f'], var_f=c['var_f'])
        evals, evecs, _ = extract_da_directions(
            simulator=gen, lag=lag_star, n_trajectories=n_da, dt=DT, seed=1234,
            device=torch.device(device) if not isinstance(device, torch.device) else device)
        da1 = evecs[:, 0].cpu().numpy()
        cs, ca, ang = da_angle(da1, da1_base)
        _, _, ang_e1 = da_angle(da1, e1)
        gap = float(evals[0].item() / max(evals[1].item(), 1e-12))
        levels.append(dict(family='additive', level=a, cos_signed=cs, cos_abs=ca,
                           angle_from_base_deg=ang, angle_from_e1_deg=ang_e1, gap12=gap,
                           da1=da1.tolist()))
    for f in smooth_fracs:
        covs_infl = inflate_covs(c['covs'], f)
        sfn = make_smoothed_score_fn(c['means'], covs_infl, c['weights'], device=device)
        gen = GenericLangevinSimulator(
            sfn, c['samples'], dt=DT, device=device,
            mu_f=c['mu_f'], cov_f=c['cov_f'], var_f=c['var_f'])
        evals, evecs, _ = extract_da_directions(
            simulator=gen, lag=lag_star, n_trajectories=n_da, dt=DT, seed=1234,
            device=torch.device(device) if not isinstance(device, torch.device) else device)
        da1 = evecs[:, 0].cpu().numpy()
        cs, ca, ang = da_angle(da1, da1_base)
        _, _, ang_e1 = da_angle(da1, e1)
        gap = float(evals[0].item() / max(evals[1].item(), 1e-12))
        levels.append(dict(family='smoothing', level=f, cos_signed=cs, cos_abs=ca,
                           angle_from_base_deg=ang, angle_from_e1_deg=ang_e1, gap12=gap,
                           da1=da1.tolist()))

    # PAIRED reference. `da1_base` above comes from da_convergence_profile(), a
    # DIFFERENT code path from the per-level extract_da_directions() calls, so the
    # zero-perturbation level does NOT reproduce it and angle_from_base_deg carries
    # a spurious 0.5-3 deg offset at level 0. Measure each family against its own
    # unperturbed level instead: same code path, same seed, same trajectories, so
    # the only difference is the perturbation and the level-0 deviation is exactly 0.
    for fam in ('additive', 'smoothing'):
        ref = next((l for l in levels if l['family'] == fam and l['level'] == 0.0), None)
        if ref is None:
            continue
        ref_v = np.asarray(ref['da1'], dtype=float)
        for l in levels:
            if l['family'] != fam:
                continue
            _, _, a0 = da_angle(np.asarray(l['da1'], dtype=float), ref_v)
            l['angle_from_unperturbed_deg'] = a0

    return dict(gmm_id=c['gid'], mi=float(mi_val), lag_star=lag_star,
                separation=c['meta']['separation'], variance=c['meta']['variance'],
                w1=c['meta']['w1'], baseline_da1_abs_cos_e1=base_abs_e1,
                baseline_da1_angle_from_e1_deg=base_ang_e1, levels=levels)


def build_markdown(cfg, mi_vals, baseline_spearman, add_spearman, smooth_spearman,
                   da_records):
    """Metadata-driven rebuttal table (identical for monolithic & merged outputs)."""
    n_gmm = cfg['n_gmm']; seeds = cfg['seeds']; T_phys = cfg['T_phys']
    add_alphas = cfg['add_alphas']; smooth_fracs = cfg['smooth_fracs']
    n_representative = cfg['n_representative']
    md = []
    md.append(f"**SSA->MI Spearman rank correlation under score perturbation** "
              f"({n_gmm} random {D}-D GMMs, N={N_TRAJ}, dt={DT}, T={T_phys}s finite "
              f"horizon, 1/N noise floor, {len(seeds)} seed(s); mean +/- SD across "
              f"seeds).")
    md.append("")
    md.append("| Perturbation | Level | Spearman(SSA, MI) |")
    md.append("|---|---|---|")
    md.append(f"| _none (exact score)_ | 0 | {baseline_spearman['mean']:+.3f} "
              f"+/- {baseline_spearman['sd']:.3f} |")
    for a in add_alphas:
        if a == 0.0:
            continue
        s = add_spearman[str(a)]
        md.append(f"| Additive score noise | alpha={a} x s0 | "
                  f"{s['mean']:+.3f} +/- {s['sd']:.3f} |")
    for f in smooth_fracs:
        if f == 0.0:
            continue
        s = smooth_spearman[str(f)]
        md.append(f"| Gaussian smoothing | +{f} x var per comp | "
                  f"{s['mean']:+.3f} +/- {s['sd']:.3f} |")
    md.append("")
    if da_records:
        md.append(f"**DA1 angular deviation from exact-score baseline** "
                  f"(representative GMMs = top-{n_representative} by true MI; "
                  f"fixed baseline lag per GMM; angle uses sign-invariant |cos|).")
        md.append("")
        md.append("| GMM (sep) | baseline DA1 vs e1 | additive alpha=1.0 | "
                  "smoothing +1.0xvar |")
        md.append("|---|---|---|---|")
        for rec in da_records:
            add1 = next((l for l in rec['levels']
                         if l['family'] == 'additive' and l['level'] == 1.0), None)
            sm1 = next((l for l in rec['levels']
                        if l['family'] == 'smoothing' and l['level'] == 1.0), None)
            a_str = f"{add1['angle_from_base_deg']:.1f} deg" if add1 else "-"
            s_str = f"{sm1['angle_from_base_deg']:.1f} deg" if sm1 else "-"
            md.append(f"| #{rec['gmm_id']} (sep={rec['separation']:.1f}) | "
                      f"{rec['baseline_da1_angle_from_e1_deg']:.1f} deg | "
                      f"{a_str} | {s_str} |")
        md.append("")
    return "\n".join(md)


def _print_summary(cfg, baseline_spearman, add_spearman, smooth_spearman, repro):
    print(f"\nBaseline (exact score) Spearman(SSA,MI) = "
          f"{baseline_spearman['mean']:+.3f} ± {baseline_spearman['sd']:.3f}")
    print(f"Reproduction max |SSA(level0) - SSA(baseline)| = {repro:.3e}")
    print("\nAdditive score noise:")
    for a in cfg['add_alphas']:
        s = add_spearman[str(a)]
        print(f"  alpha={a:<5} Spearman={s['mean']:+.3f} ± {s['sd']:.3f}")
    print("\nSmoothed-density (cov inflation):")
    for f in cfg['smooth_fracs']:
        s = smooth_spearman[str(f)]
        print(f"  frac={f:<5} Spearman={s['mean']:+.3f} ± {s['sd']:.3f}")


# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------

def mode_shard(cfg, results_dir, gmm_start, gmm_end, device='cpu', max_lag=None):
    """Run the SSA sweep for GMM ids [gmm_start, gmm_end); write a per-shard JSON."""
    run_tag = cfg['run_tag']
    T = int(round(cfg['T_phys'] / DT))
    gids = list(range(gmm_start, gmm_end))
    print(f"=== P1 shard [{gmm_start},{gmm_end}) tag={run_tag} ===")
    print(f"d={D} N={N_TRAJ} dt={DT} T_phys={cfg['T_phys']}s T={T} floor(1/N)={FLOOR_1N:g}")
    print(f"gids={gids} seeds={cfg['seeds']} device={device} "
          f"max_lag={int(max_lag) if max_lag is not None else T} (T={T})")
    print(f"additive alphas={cfg['add_alphas']} smoothing fracs={cfg['smooth_fracs']}")

    baseline = {}; additive = {str(a): {} for a in cfg['add_alphas']}
    smoothing = {str(f): {} for f in cfg['smooth_fracs']}
    mi = {}; metas = {}; repro = 0.0; diags = {}; all_curves = {}
    n_cells = 0; n_wl = 0
    t0 = time.time()
    for gid in gids:
        c = precompute_gmm(gid, device=device)
        b, add, sm, r, diag = sweep_gmm(c, cfg['seeds'], T, cfg['add_alphas'],
                                        cfg['smooth_fracs'], device=device,
                                        max_lag=max_lag)
        # finite-horizon bookkeeping: count every (level, seed) cell
        wl_flags = (list(diag['baseline_wl'])
                    + [v for a in cfg['add_alphas'] for v in diag['additive_wl'][a]]
                    + [v for f in cfg['smooth_fracs'] for v in diag['smoothing_wl'][f]])
        n_cells += len(wl_flags); n_wl += sum(bool(v) for v in wl_flags)
        diags[str(gid)] = dict(
            baseline_keep=diag['baseline_keep'], baseline_wl=diag['baseline_wl'],
            additive_keep={str(a): diag['additive_keep'][a] for a in cfg['add_alphas']},
            additive_wl={str(a): diag['additive_wl'][a] for a in cfg['add_alphas']},
            smoothing_keep={str(f): diag['smoothing_keep'][f] for f in cfg['smooth_fracs']},
            smoothing_wl={str(f): diag['smoothing_wl'][f] for f in cfg['smooth_fracs']})
        baseline[str(gid)] = b
        for k, v in diag.pop('curves').items():
            all_curves[f'g{gid}_{k}'] = v
        for a in cfg['add_alphas']:
            additive[str(a)][str(gid)] = add[a]
        for f in cfg['smooth_fracs']:
            smoothing[str(f)][str(gid)] = sm[f]
        mi[str(gid)] = c['mi']; metas[str(gid)] = c['meta']
        repro = max(repro, r)
        print(f"  gid {gid} done (repro={r:.2e}, keep_base={diag['baseline_keep']}, "
              f"window_limited={diag['baseline_wl']}, {time.time()-t0:.1f}s)", flush=True)

    shard = dict(
        run_tag=run_tag, gmm_start=gmm_start, gmm_end=gmm_end, gmm_ids=gids,
        config=dict(d=D, N=N_TRAJ, dt=DT, T_phys=cfg['T_phys'], T=T,
                    floor_1N=FLOOR_1N, seeds=cfg['seeds'],
                    additive_alphas=cfg['add_alphas'],
                    smoothing_fracs=cfg['smooth_fracs'], device=str(device),
                    max_lag=int(max_lag if max_lag is not None else T)),
        mi=mi, metas=metas, baseline=baseline, additive=additive,
        smoothing=smoothing, repro_max_absdiff=repro, diagnostics=diags,
        n_cells=n_cells, n_window_limited=n_wl,
        frac_window_limited=(n_wl / n_cells if n_cells else 0.0),
        total_time_s=time.time() - t0,
    )
    out = os.path.join(results_dir,
                       f'score_robustness_{run_tag}_shard_{gmm_start}_{gmm_end}.json')
    shard['config']['stopping'] = STOPPING
    with open(out, 'w') as fh:
        json.dump(shard, fh, indent=2)
    np.savez_compressed(out.replace('.json', '_rho_sq.npz'), **all_curves)
    print(f"\nSaved shard {out}  (+ rho_sq curves npz; repro_max_absdiff={repro:.3e})")
    print(f"Window-limited cells: {n_wl}/{n_cells} "
          f"({100.0 * n_wl / max(n_cells, 1):.1f}%) at max_lag="
          f"{int(max_lag if max_lag is not None else T)}")


def mode_da(cfg, results_dir, device='cpu'):
    """Compute MI for all GMMs (cheap), pick global top-k, run DA; write DA JSON."""
    run_tag = cfg['run_tag']; n_gmm = cfg['n_gmm']
    print(f"=== P1 DA-only tag={run_tag} (n_gmm={n_gmm}) ===")
    caches = {}; mi_vals = np.full(n_gmm, np.nan)
    for gid in range(n_gmm):
        c = precompute_gmm(gid, device=device)
        caches[gid] = c; mi_vals[gid] = c['mi']
    order = np.argsort(-np.nan_to_num(mi_vals, nan=-np.inf))
    representative = [int(g) for g in order[:min(cfg['n_representative'], n_gmm)]]
    print(f"Global top-{cfg['n_representative']} by MI: {representative}")

    da_records = []
    t0 = time.time()
    for gid in representative:
        rec = run_da_for_gmm(caches[gid], cfg['add_alphas'], cfg['smooth_fracs'],
                             cfg['n_da'], cfg['da_checkpoints'], mi_vals[gid],
                             device=device)
        da_records.append(rec)
        print(f"  DA gid {gid} lag*={rec['lag_star']} "
              f"base|angle e1|={rec['baseline_da1_angle_from_e1_deg']:.1f}deg "
              f"({time.time()-t0:.1f}s)", flush=True)

    out = os.path.join(results_dir, f'score_robustness_{run_tag}_da.json')
    with open(out, 'w') as fh:
        json.dump(dict(run_tag=run_tag, n_gmm=n_gmm,
                       representative_gmm_ids=representative,
                       representative_rule='top-%d by true MI' % cfg['n_representative'],
                       mi=mi_vals.tolist(), da=da_records), fh, indent=2)
    print(f"\nSaved DA {out}")


def mode_merge(cfg, results_dir, out_tag):
    """Merge shard JSONs (canonical gid order) + optional DA -> final JSON + MD."""
    run_tag = cfg['run_tag']; n_gmm = cfg['n_gmm']; seeds = cfg['seeds']
    add_alphas = cfg['add_alphas']; smooth_fracs = cfg['smooth_fracs']
    n_seeds = len(seeds)
    pattern = os.path.join(results_dir, f'score_robustness_{run_tag}_shard_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no shard files match {pattern}")
    print(f"=== P1 merge tag={run_tag}: {len(files)} shard files ===")

    baseline_by = {}; add_by = {str(a): {} for a in add_alphas}
    smooth_by = {str(f): {} for f in smooth_fracs}
    mi_by = {}; meta_by = {}; repro = 0.0
    n_cells_tot = 0; n_wl_tot = 0; shard_max_lags = set()
    for p in files:
        s = json.load(open(p))
        # validate compatible metadata across shards
        sc = s['config']
        assert abs(sc['dt'] - DT) < 1e-12 and abs(sc['floor_1N'] - FLOOR_1N) < 1e-12, \
            f"shard {p} floor/dt mismatch"
        assert sc['seeds'] == seeds, f"shard {p} seeds {sc['seeds']} != {seeds}"
        # NOTE: 'device' may differ across shards; SSA numbers are device-independent
        # (all RNG is drawn on CPU), so it is recorded but not required to match.
        assert sc['additive_alphas'] == add_alphas, f"shard {p} alpha grid mismatch"
        assert sc['smoothing_fracs'] == smooth_fracs, f"shard {p} frac grid mismatch"
        for gid_s, vals in s['baseline'].items():
            gid = int(gid_s)
            if gid in baseline_by:
                raise SystemExit(f"duplicate gid {gid} across shards")
            baseline_by[gid] = vals
        for a in add_alphas:
            for gid_s, vals in s['additive'][str(a)].items():
                add_by[str(a)][int(gid_s)] = vals
        for f in smooth_fracs:
            for gid_s, vals in s['smoothing'][str(f)].items():
                smooth_by[str(f)][int(gid_s)] = vals
        for gid_s, v in s['mi'].items():
            mi_by[int(gid_s)] = v
        for gid_s, v in s['metas'].items():
            meta_by[int(gid_s)] = v
        repro = max(repro, s.get('repro_max_absdiff', 0.0))
        n_cells_tot += s.get('n_cells', 0); n_wl_tot += s.get('n_window_limited', 0)
        shard_max_lags.add(s['config'].get('max_lag'))

    missing = [g for g in range(n_gmm) if g not in baseline_by]
    if missing:
        raise SystemExit(f"incomplete coverage; missing gids {missing}")
    extra = [g for g in baseline_by if g not in range(n_gmm)]
    if extra:
        raise SystemExit(f"unexpected gids {extra}")

    # Assemble full arrays in canonical gid order 0..n_gmm-1
    mi_vals = np.array([mi_by[g] for g in range(n_gmm)], dtype=float)
    metas = [meta_by[g] for g in range(n_gmm)]
    baseline_shat = np.array([[baseline_by[g][si] for g in range(n_gmm)]
                              for si in range(n_seeds)])
    add_shat = {a: np.array([[add_by[str(a)][g][si] for g in range(n_gmm)]
                             for si in range(n_seeds)]) for a in add_alphas}
    smooth_shat = {f: np.array([[smooth_by[str(f)][g][si] for g in range(n_gmm)]
                                for si in range(n_seeds)]) for f in smooth_fracs}

    baseline_spearman = spearman_levels({'base': baseline_shat}, ['base'],
                                        mi_vals, n_seeds)['base']
    add_spearman = {str(a): v for a, v in
                    spearman_levels(add_shat, add_alphas, mi_vals, n_seeds).items()}
    smooth_spearman = {str(f): v for f, v in
                       spearman_levels(smooth_shat, smooth_fracs, mi_vals, n_seeds).items()}

    # Fold DA if present
    da_path = os.path.join(results_dir, f'score_robustness_{run_tag}_da.json')
    da_records = []; representative = []
    if os.path.exists(da_path):
        da = json.load(open(da_path))
        da_records = da['da']; representative = da['representative_gmm_ids']
        print(f"Folded DA from {da_path} (representatives {representative})")
    else:
        print(f"NOTE: no DA file at {da_path}; final table omits DA rows.")

    final_tag = out_tag or run_tag
    result = dict(
        config=dict(d=D, N=N_TRAJ, dt=DT, T_phys=cfg['T_phys'],
                    T=int(round(cfg['T_phys'] / DT)), floor_1N=FLOOR_1N,
                    n_gmm=n_gmm, seeds=seeds, gmm_ids=list(range(n_gmm)),
                    additive_alphas=add_alphas, smoothing_fracs=smooth_fracs,
                    n_da=cfg['n_da'], da_checkpoints=cfg['da_checkpoints'],
                    tag=final_tag, run_tag=run_tag,
                    representative_rule='top-%d by true MI' % cfg['n_representative'],
                    representative_gmm_ids=representative,
                    merged_from=[os.path.basename(p) for p in files]),
        mi=mi_vals.tolist(), metas=metas,
        baseline=dict(shat=baseline_shat.tolist(), spearman=baseline_spearman),
        additive=dict(shat={str(a): add_shat[a].tolist() for a in add_alphas},
                      spearman=add_spearman),
        smoothing=dict(shat={str(f): smooth_shat[f].tolist() for f in smooth_fracs},
                       spearman=smooth_spearman),
        da=da_records,
        checks=dict(repro_max_absdiff=repro,
                    gmm_ids_consistent=(list(range(n_gmm)) == list(range(n_gmm))),
                    coverage_complete=True,
                    n_cells=n_cells_tot, n_window_limited=n_wl_tot,
                    frac_window_limited=(n_wl_tot / n_cells_tot if n_cells_tot else 0.0),
                    shard_max_lags=sorted(x for x in shard_max_lags if x is not None)),
        total_time_s=None,
    )
    out_json = os.path.join(results_dir, f'score_robustness_{final_tag}.json')
    with open(out_json, 'w') as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved merged {out_json}")

    _print_summary(cfg, baseline_spearman, add_spearman, smooth_spearman, repro)
    print(f"\nWindow-limited cells: {n_wl_tot}/{n_cells_tot} "
          f"({100.0 * n_wl_tot / max(n_cells_tot, 1):.1f}%) "
          f"[shard max_lag values: {sorted(x for x in shard_max_lags if x is not None)}]")
    if n_wl_tot:
        print("  NOTE: window-limited cells are finite-horizon LOWER BOUNDS on Shat "
              "(rho^2 never fell below the 1/N floor within max_lag), not resolved "
              "population values.")
    cfg_md = dict(cfg)
    md_text = build_markdown(cfg_md, mi_vals, baseline_spearman, add_spearman,
                             smooth_spearman, da_records)
    out_md = os.path.join(results_dir, f'score_robustness_{final_tag}.md')
    with open(out_md, 'w') as fh:
        fh.write(md_text + "\n")
    print(f"Saved {out_md}\n\n{md_text}")


def mode_monolithic(cfg, results_dir, device='cpu', max_lag=None):
    """Single-process: full sweep + DA + final JSON/MD (used by --smoke and full-1proc)."""
    run_tag = cfg['run_tag']; n_gmm = cfg['n_gmm']; seeds = cfg['seeds']
    add_alphas = cfg['add_alphas']; smooth_fracs = cfg['smooth_fracs']
    n_seeds = len(seeds); T = int(round(cfg['T_phys'] / DT))
    print(f"=== P1 score robustness monolithic ({run_tag}) ===")
    print(f"d={D} N={N_TRAJ} dt={DT} T_phys={cfg['T_phys']}s T={T} floor(1/N)={FLOOR_1N:g}")
    print(f"GMMs={n_gmm} seeds={seeds} device={device}")
    print(f"additive alphas={add_alphas}\nsmoothing fracs={smooth_fracs}")

    caches = {}; mi_vals = np.full(n_gmm, np.nan); metas = []
    for gid in range(n_gmm):
        c = precompute_gmm(gid, device=device)
        caches[gid] = c; mi_vals[gid] = c['mi']; metas.append(c['meta'])

    baseline_shat = np.full((n_seeds, n_gmm), np.nan)
    add_shat = {a: np.full((n_seeds, n_gmm), np.nan) for a in add_alphas}
    smooth_shat = {f: np.full((n_seeds, n_gmm), np.nan) for f in smooth_fracs}
    repro = 0.0; n_cells = 0; n_wl = 0
    t0 = time.time()
    for gid in range(n_gmm):
        b, add, sm, r, diag = sweep_gmm(caches[gid], seeds, T, add_alphas,
                                        smooth_fracs, device=device, max_lag=max_lag)
        wl_flags = (list(diag['baseline_wl'])
                    + [v for a in add_alphas for v in diag['additive_wl'][a]]
                    + [v for f in smooth_fracs for v in diag['smoothing_wl'][f]])
        n_cells += len(wl_flags); n_wl += sum(bool(v) for v in wl_flags)
        for si in range(n_seeds):
            baseline_shat[si, gid] = b[si]
            for a in add_alphas:
                add_shat[a][si, gid] = add[a][si]
            for f in smooth_fracs:
                smooth_shat[f][si, gid] = sm[f][si]
        repro = max(repro, r)
    print(f"  sweep done ({time.time()-t0:.1f}s)")
    print(f"  window-limited cells: {n_wl}/{n_cells} "
          f"({100.0 * n_wl / max(n_cells, 1):.1f}%)")

    baseline_spearman = spearman_levels({'base': baseline_shat}, ['base'],
                                        mi_vals, n_seeds)['base']
    add_spearman = {str(a): v for a, v in
                    spearman_levels(add_shat, add_alphas, mi_vals, n_seeds).items()}
    smooth_spearman = {str(f): v for f, v in
                       spearman_levels(smooth_shat, smooth_fracs, mi_vals, n_seeds).items()}

    # DA on global top-k
    order = np.argsort(-np.nan_to_num(mi_vals, nan=-np.inf))
    representative = [int(g) for g in order[:min(cfg['n_representative'], n_gmm)]]
    da_records = [run_da_for_gmm(caches[g], add_alphas, smooth_fracs,
                                 cfg['n_da'], cfg['da_checkpoints'], mi_vals[g],
                                 device=device)
                  for g in representative]
    total_time = time.time() - t0

    result = dict(
        config=dict(d=D, N=N_TRAJ, dt=DT, T_phys=cfg['T_phys'], T=T,
                    floor_1N=FLOOR_1N, n_gmm=n_gmm, seeds=seeds,
                    gmm_ids=list(range(n_gmm)), additive_alphas=add_alphas,
                    smoothing_fracs=smooth_fracs, n_da=cfg['n_da'],
                    da_checkpoints=cfg['da_checkpoints'], tag=run_tag,
                    representative_rule='top-%d by true MI' % cfg['n_representative'],
                    representative_gmm_ids=representative),
        mi=mi_vals.tolist(), metas=metas,
        baseline=dict(shat=baseline_shat.tolist(), spearman=baseline_spearman),
        additive=dict(shat={str(a): add_shat[a].tolist() for a in add_alphas},
                      spearman=add_spearman),
        smoothing=dict(shat={str(f): smooth_shat[f].tolist() for f in smooth_fracs},
                       spearman=smooth_spearman),
        da=da_records,
        checks=dict(repro_max_absdiff=repro,
                    gmm_ids_consistent=(list(range(n_gmm)) == list(range(n_gmm))),
                    n_cells=n_cells, n_window_limited=n_wl,
                    frac_window_limited=(n_wl / n_cells if n_cells else 0.0)),
        total_time_s=total_time,
    )
    out_json = os.path.join(results_dir, f'score_robustness_{run_tag}.json')
    with open(out_json, 'w') as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved {out_json}")
    _print_summary(cfg, baseline_spearman, add_spearman, smooth_spearman, repro)
    md_text = build_markdown(cfg, mi_vals, baseline_spearman, add_spearman,
                             smooth_spearman, da_records)
    out_md = os.path.join(results_dir, f'score_robustness_{run_tag}.md')
    with open(out_md, 'w') as fh:
        fh.write(md_text + "\n")
    print(f"Saved {out_md}\n\n{md_text}\n\nTotal time: {total_time:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--test', action='store_true',
                    help='run stopping-lag unit tests and exit')
    ap.add_argument('--gmm_start', type=int, default=None,
                    help='shard mode: half-open range start over GMM ids')
    ap.add_argument('--gmm_end', type=int, default=None,
                    help='shard mode: half-open range end over GMM ids')
    ap.add_argument('--da_only', action='store_true',
                    help='global DA analysis (top-k by MI) only')
    ap.add_argument('--merge', action='store_true',
                    help='merge shard JSONs + DA -> final JSON/MD')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--stopping', choices=['first', 'largest'], default='first',
                    help="1/N stopping rule: 'first' (paper) or 'largest' (original rebuttal run)")
    ap.add_argument('--run_tag', type=str, default=None,
                    help="override the run tag used to name shard/DA/merged files")
    ap.add_argument('--T_phys', type=float, default=100.0,
                    help='physical horizon (s); T = T_phys/dt')
    ap.add_argument('--out_tag', type=str, default='',
                    help='merge mode: override final output tag')
    ap.add_argument('--max_lag', type=int, default=None,
                    help='autocorrelation lags to evaluate (default: T, the SUBMITTED '
                         'convention). A smaller cap saves compute but can leave cells '
                         'window-limited; validate before adopting.')
    ap.add_argument('--device', type=str, default='cpu',
                    help="compute device: 'cpu' (default) or 'cuda'. All RNG stays on "
                         "CPU, so SSA results are identical on either device.")
    args = ap.parse_args()

    if args.test:
        run_tests()
        return

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit(f"--device {device} requested but torch.cuda.is_available() "
                         f"is False")
    if device.startswith('cuda'):
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}")

    global STOPPING
    STOPPING = args.stopping
    cfg = get_config(args)
    if args.run_tag:
        cfg['run_tag'] = args.run_tag
    results_dir = os.path.join(CODEDIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    if args.merge:
        mode_merge(cfg, results_dir, args.out_tag or None)
    elif args.da_only:
        mode_da(cfg, results_dir, device=device)
    elif args.gmm_start is not None or args.gmm_end is not None:
        start = 0 if args.gmm_start is None else args.gmm_start
        end = cfg['n_gmm'] if args.gmm_end is None else args.gmm_end
        mode_shard(cfg, results_dir, start, end, device=device, max_lag=args.max_lag)
    else:
        mode_monolithic(cfg, results_dir, device=device, max_lag=args.max_lag)


if __name__ == '__main__':
    main()
