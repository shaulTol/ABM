#!/usr/bin/env python3
"""
P2: Numerical robustness of the MAIN-FIGURE dispersion-vs-fragmentation pair.

Target distributions are the EXACT pair from experiments/synthetic/fig_dispersion.py:
  1. 'gaussian' : isotropic Gaussian N(0, 25 I_2)              (the "blob")
  2. 'ellipses' : equal-weight parallel-ellipses GMM, means (0,-3),(0,+3),
                  component covariance diag(25, 1)

PRIMARY QUESTION: does the headline ordering  S_ellipse > S_Gaussian  survive
changes in step size, trajectory count, and finite horizon, even when the
ABSOLUTE finite-horizon SSA values move? Absolute Shat here is a finite-horizon
quantity, never a resolved population value.

BASELINE (main-figure settings, reproduced first):
  dt=0.01, T_sim=5000 steps (physical sim horizon 50.0), max_lag=4000
  (physical lag window 40.0), N=2000, seeds 42+100 (ellipses) / 42+200 (gaussian).
  Submitted values: ~1.822 (ellipses), ~1.022 (gaussian). The run STOPS if the
  baseline does not reproduce within Monte Carlo uncertainty across seeds.

GRIDS (one factor at a time around the main-figure baseline):
  - dt      {0.005, 0.01, 0.02} at MATCHED physical simulation horizon (50.0)
            AND matched physical lag window (40.0): T = 50/dt, max_lag = 40/dt.
  - N       {200, 500, 1000, 2000} at dt=0.01.
  - horizon physical lag windows {10, 20, 30, 40}, evaluated POST-HOC from the
            SAME stored baseline rho^2 curve (no re-simulation).

BOTH stopping rules are computed from the SAME stored rho^2 curve:
  (i)  'first_crossing' : truncate at the FIRST lag with rho^2 < 1/N.
       This is the paper's stopping rule (eq:tmax_stopping) and what the figure
       code (fig_dispersion.py) implements, so it REPRODUCES the paper's numbers.
  (ii) 'largest_lag'    : integrate through the LARGEST lag with rho^2 > 1/N.
       An alternative rule, kept as a SENSITIVITY check.
  The paper reports that the ordering holds under both rules.

Autocorrelation kernel: the ORIGINAL 'loop' implementation is used by default.
src/metrics/autocorrelation.py has an opt-in method='fft' path that has NOT been
validated end-to-end; --acf_method fft is available but runs a mandatory
equivalence check against the loop kernel on BOTH distributions first.

The four-sweep Delta-t robustness result (Appendix Fig. A3) is CITED, not
re-derived here; this experiment exists to make the MAIN VISUAL EXAMPLE
directly interpretable.

Usage:
  python compute_p2_dispersion_robustness.py --device cuda      # full run
  python compute_p2_dispersion_robustness.py --smoke            # CPU smoke
  python compute_p2_dispersion_robustness.py --test             # unit tests
  python compute_p2_dispersion_robustness.py --baseline_only    # reproduction gate
"""
import sys, os, json, time, argparse, ast, inspect
import torch
import numpy as np

CODEDIR = os.path.dirname(os.path.abspath(__file__))
REPODIR = os.path.join(CODEDIR, '..', '..', '..')
sys.path.insert(0, REPODIR)
sys.path.insert(0, CODEDIR)
sys.path.insert(0, os.path.join(REPODIR, 'experiments', 'synthetic'))

from src.simulation.langevin import LangevinSimulatorD, simulate_d_trajectories
from src.metrics.autocorrelation import compute_vector_autocorrelation, aggregate_ssa

# ── Main-figure baseline settings (from fig_dispersion.py) ──
BASE_SEED = 42
BASE_DT = 0.01
BASE_T_STEPS = 5000            # physical simulation horizon = 50.0
BASE_MAX_LAG = 4000            # physical lag window = 40.0
BASE_N = 2000
BASE_T_PHYS = BASE_T_STEPS * BASE_DT      # 50.0
BASE_LAG_WINDOW = BASE_MAX_LAG * BASE_DT  # 40.0

# Submitted values to reproduce (fig_dispersion.py / main figure annotation)
SUBMITTED = {'gaussian': 1.022, 'ellipses': 1.822}

# Grids (one factor at a time)
DT_GRID = [0.005, 0.01, 0.02]
N_GRID = [200, 500, 1000, 2000]
HORIZON_GRID = [10.0, 20.0, 30.0, 40.0]   # physical lag windows, post-hoc
N_SEEDS = 5


def get_distribution(name):
    """Exact distributions from fig_dispersion.py. seed_offset reproduces its seeds."""
    if name == 'gaussian':
        return {
            'means': torch.tensor([[0.0, 0.0]]),
            'covs': torch.tensor([[[25.0, 0.0], [0.0, 25.0]]]),
            'weights': torch.tensor([1.0]),
            'seed_offset': 200,       # fig_dispersion: seed + 200
            'label': 'Gaussian N(0,25I)',
        }
    elif name == 'ellipses':
        return {
            'means': torch.tensor([[0.0, -3.0], [0.0, 3.0]]),
            'covs': torch.tensor([[[25.0, 0.0], [0.0, 1.0]],
                                  [[25.0, 0.0], [0.0, 1.0]]]),
            'weights': torch.tensor([0.5, 0.5]),
            'seed_offset': 100,       # fig_dispersion: seed + 100
            'label': 'parallel ellipses GMM',
        }
    raise ValueError(f"unknown distribution {name!r}")


DIST_NAMES = ['gaussian', 'ellipses']


def seed_for(dist_name, seed_index):
    """Deterministic seed. seed_index=0 reproduces fig_dispersion.py exactly."""
    return BASE_SEED + get_distribution(dist_name)['seed_offset'] + 1000 * seed_index


# ────────────────────────── stopping rules ──────────────────────────

def ssa_first_crossing(rho_sq, dt, floor):
    """(i) PAPER RULE (eq:tmax_stopping, fig_dispersion.py): truncate at the FIRST lag with rho^2 < floor.

    Mirrors fig_dispersion.py compute_ssa_truncated exactly.
    Returns (S, keep, stop_lag_phys, window_limited) where keep = #retained lags.
    """
    keep = len(rho_sq)
    below = rho_sq < floor
    if below.any():
        keep = int(np.argmax(below))
    S = float(aggregate_ssa(rho_sq[:keep], dt=dt, normalize='riemann'))
    return S, keep, float(keep * dt), bool(keep == len(rho_sq))


def ssa_largest_lag(rho_sq, dt, floor):
    """(ii) ALTERNATIVE RULE (sensitivity): integrate through the LARGEST lag with rho^2 > floor."""
    above = rho_sq > floor
    if not above.any():
        return 0.0, 0, 0.0, False
    keep = int(np.nonzero(above)[0].max()) + 1
    S = float(aggregate_ssa(rho_sq[:keep], dt=dt, normalize='riemann'))
    return S, keep, float(keep * dt), bool(keep == len(rho_sq))


RULES = {'first_crossing': ssa_first_crossing, 'largest_lag': ssa_largest_lag}


# ────────────────────────── simulation ──────────────────────────

def simulate_rho_sq(dist_name, n_traj, seed, dt, T, max_lag, device='cpu', acf_method='loop'):
    """Simulate one ensemble; return per-lag rho^2 for lags 1..max_lag.

    DETERMINISM: all RNG is drawn on the CPU generator seeded by `seed` and the
    tensors are moved to `device`; device-side RNG is never used, so --device cuda
    reproduces the --device cpu stream (and hence fig_dispersion.py run on CPU).
    """
    dist = get_distribution(dist_name)
    means, covs, weights = dist['means'], dist['covs'], dist['weights']

    torch.manual_seed(seed)
    sim_cpu = LangevinSimulatorD(means, covs, weights, dt, torch.device('cpu'))
    x0 = sim_cpu.sample_initial(n_traj)
    d = sim_cpu.d

    if str(device) == 'cpu':
        sim, x = sim_cpu, x0
    else:
        sim = LangevinSimulatorD(means.to(device), covs.to(device), weights.to(device),
                                 dt, torch.device(device))
        x = x0.to(device)

    traj = torch.zeros(n_traj, T + 1, d, device=sim.device)
    traj[:, 0, :] = x
    for t in range(T):
        z = torch.randn(n_traj, d)                       # CPU RNG, per step
        if str(device) != 'cpu':
            z = z.to(device, non_blocking=True)
        x = sim.step(x, z=z)
        traj[:, t + 1, :] = x

    kw = {}
    if acf_method != 'loop':
        # only pass through if the installed function supports it
        if 'method' in inspect.signature(compute_vector_autocorrelation).parameters:
            kw['method'] = acf_method
        else:
            raise SystemExit("ERROR: --acf_method fft requested but compute_vector_autocorrelation "
                             "has no 'method' parameter in this checkout.")
    rho, _, _ = compute_vector_autocorrelation(
        traj, max_lag, gmm_means=sim.means, gmm_covs=sim.covs, gmm_weights=sim.weights,
        demean=True, normalization='trace', **kw)
    return (rho.cpu().numpy() ** 2)[1:].astype(np.float64)


def evaluate_curve(rho_sq, dt, n_traj, max_lag=None):
    """Apply BOTH stopping rules to one stored rho^2 curve (optionally truncated)."""
    curve = rho_sq if max_lag is None else rho_sq[:max_lag]
    floor = 1.0 / n_traj                      # N-dependent criterion, recomputed
    out = {'floor_1N': float(floor), 'max_lag': int(len(curve)),
           'phys_lag_window': float(len(curve) * dt)}
    for rname, fn in RULES.items():
        S, keep, stop_phys, wlim = fn(curve, dt, floor)
        out[rname] = {'S': S, 'keep': int(keep), 'stop_lag_phys': stop_phys,
                      'window_limited': bool(wlim)}
    return out


def run_condition(n_traj, dt, T, max_lag, n_seeds, device, acf_method, label, verbose=True):
    """Simulate BOTH distributions x seeds for one (N, dt, horizon) condition."""
    res = {'N': int(n_traj), 'dt': float(dt), 'T': int(T),
           'T_phys': float(T * dt), 'max_lag': int(max_lag),
           'phys_lag_window': float(max_lag * dt), 'label': label,
           'n_seeds': int(n_seeds), 'per_dist': {}}
    rho_store = {}
    wall_total = 0.0
    for dname in DIST_NAMES:
        curves = np.zeros((n_seeds, max_lag), dtype=np.float32)
        t0 = time.time()
        for si in range(n_seeds):
            curves[si] = simulate_rho_sq(dname, n_traj, seed_for(dname, si),
                                         dt, T, max_lag, device, acf_method)
        wall = time.time() - t0
        wall_total += wall
        evals = [evaluate_curve(curves[si].astype(np.float64), dt, n_traj) for si in range(n_seeds)]
        per = {'wall_s': float(wall),
               'score_evals': int(n_traj) * int(T) * int(n_seeds),
               'seeds': [seed_for(dname, si) for si in range(n_seeds)]}
        for rname in RULES:
            S = np.array([e[rname]['S'] for e in evals])
            per[rname] = {
                'S_mean': float(S.mean()),
                'S_sd': float(S.std(ddof=1)) if n_seeds > 1 else 0.0,
                'S_sem': float(S.std(ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0,
                'S_per_seed': S.tolist(),
                'stop_lag_phys_mean': float(np.mean([e[rname]['stop_lag_phys'] for e in evals])),
                'window_limited_any': bool(any(e[rname]['window_limited'] for e in evals)),
                'window_limited_all': bool(all(e[rname]['window_limited'] for e in evals)),
            }
        per['floor_1N'] = 1.0 / n_traj
        res['per_dist'][dname] = per
        rho_store[dname] = curves
        if verbose:
            fc, ll = per['first_crossing'], per['largest_lag']
            print(f"    {dname:>9}: first_crossing S={fc['S_mean']:.4f}±{fc['S_sd']:.4f} "
                  f"(tau*={fc['stop_lag_phys_mean']:.2f}{'*wl' if fc['window_limited_any'] else ''}) | "
                  f"largest_lag S={ll['S_mean']:.4f}±{ll['S_sd']:.4f} "
                  f"(tau*={ll['stop_lag_phys_mean']:.2f}{'*wl' if ll['window_limited_any'] else ''}) "
                  f"[{wall:.1f}s]", flush=True)
    res['wall_s'] = wall_total
    res['score_evals'] = sum(res['per_dist'][d]['score_evals'] for d in DIST_NAMES)
    add_ordering(res)
    return res, rho_store


def add_ordering(res):
    """Ellipse/Gaussian ratio + ordering flag, per stopping rule."""
    res['ordering'] = {}
    for rname in RULES:
        g = res['per_dist']['gaussian'][rname]
        e = res['per_dist']['ellipses'][rname]
        ratio = e['S_mean'] / g['S_mean'] if g['S_mean'] > 0 else float('inf')
        # per-seed ordering (paired by seed index)
        pair = [float(a) > float(b) for a, b in zip(e['S_per_seed'], g['S_per_seed'])]
        res['ordering'][rname] = {
            'ratio_ellipse_over_gaussian': float(ratio),
            'ordering_preserved_mean': bool(e['S_mean'] > g['S_mean']),
            'ordering_preserved_all_seeds': bool(all(pair)),
            'n_seeds_ordering_preserved': int(sum(pair)),
            'n_seeds': len(pair),
        }
    return res


# ────────────────────────── unit tests ──────────────────────────

def unit_test_stopping_rules():
    """Both conventions on the SAME non-monotone rho^2 sequence (required check 6)."""
    N = 200
    floor = 1.0 / N          # 0.005
    dt = 0.1
    #                 l1    l2     l3      l4      l5       l6
    rho_sq = np.array([0.90, 0.50, 0.001, 0.20, 0.0001, 0.00005])

    S_fc, keep_fc, stop_fc, wl_fc = ssa_first_crossing(rho_sq, dt, floor)
    S_ll, keep_ll, stop_ll, wl_ll = ssa_largest_lag(rho_sq, dt, floor)

    # first-crossing stops at the FIRST dip below the floor (index 2 -> keep 2)
    assert keep_fc == 2, f"first_crossing keep expected 2, got {keep_fc}"
    assert abs(S_fc - dt * rho_sq[:2].sum()) < 1e-12
    # largest-lag keeps through the LAST lag above the floor (index 3 -> keep 4)
    assert keep_ll == 4, f"largest_lag keep expected 4, got {keep_ll}"
    assert abs(S_ll - dt * rho_sq[:4].sum()) < 1e-12
    # the two conventions MUST disagree on this sequence
    assert keep_fc != keep_ll and abs(S_fc - S_ll) > 1e-9, "rules must differ here"
    assert wl_fc is False and wl_ll is False

    # all-above sequence -> both rules agree and are window-limited
    allabove = np.array([0.9, 0.8, 0.7])
    a1 = ssa_first_crossing(allabove, dt, floor)
    a2 = ssa_largest_lag(allabove, dt, floor)
    assert a1[1] == a2[1] == 3 and a1[3] is True and a2[3] is True

    print("[unit-test] BOTH stopping rules PASSED on the same non-monotone rho^2:")
    print(f"    rho_sq={rho_sq.tolist()}, floor=1/N={floor}")
    print(f"    first_crossing: keep={keep_fc} stop_lag={stop_fc:.2f} S={S_fc:.6f}  (PAPER RULE)")
    print(f"    largest_lag   : keep={keep_ll} stop_lag={stop_ll:.2f} S={S_ll:.6f}  (SENSITIVITY)")
    print(f"    rules disagree as expected: keep {keep_fc} vs {keep_ll}, S {S_fc:.6f} vs {S_ll:.6f}")
    print("    all-above sequence: both rules keep=3 and window_limited=True")
    return True


def check_rng_matches_reference(device='cpu'):
    """Verify our CPU-RNG simulate path reproduces simulate_d_trajectories bitwise."""
    dname = 'ellipses'
    dist = get_distribution(dname)
    n, T, dt, seed = 8, 25, 0.01, 12345
    ref = simulate_d_trajectories(dist['means'], dist['covs'], dist['weights'],
                                  n_trajectories=n, T=T, dt=dt, device='cpu', seed=seed)
    torch.manual_seed(seed)
    sim = LangevinSimulatorD(dist['means'], dist['covs'], dist['weights'], dt, torch.device('cpu'))
    x = sim.sample_initial(n)
    ours = torch.zeros(n, T + 1, sim.d)
    ours[:, 0, :] = x
    for t in range(T):
        x = sim.step(x, z=torch.randn(n, sim.d))
        ours[:, t + 1, :] = x
    ok = torch.equal(ref, ours)
    print(f"[check] CPU-RNG path reproduces simulate_d_trajectories bitwise: {ok}")
    return bool(ok)


def verify_fft_against_loop(device='cpu'):
    """Required check 7: validate the opt-in FFT ACF path on BOTH distributions."""
    if 'method' not in inspect.signature(compute_vector_autocorrelation).parameters:
        print("[check] FFT path: compute_vector_autocorrelation has no 'method' param — SKIPPED")
        return None
    ok_all = True
    for dname in DIST_NAMES:
        a = simulate_rho_sq(dname, 64, seed_for(dname, 0), 0.01, 400, 300, device, 'loop')
        b = simulate_rho_sq(dname, 64, seed_for(dname, 0), 0.01, 400, 300, device, 'fft')
        md = float(np.abs(a - b).max())
        ok = md < 1e-6
        ok_all &= ok
        print(f"[check] FFT vs loop on {dname}: max|diff|={md:.3e} -> {'OK' if ok else 'MISMATCH'}")
    return ok_all


# ────────────────────────── baseline reproduction gate ──────────────────────────

def reproduce_baseline(n_seeds, device, acf_method, dt=BASE_DT, T=BASE_T_STEPS,
                       max_lag=BASE_MAX_LAG, n_traj=BASE_N, strict=True):
    """Reproduce the submitted main-figure values BEFORE running any grid.

    Uses the ORIGINAL figure settings and the FIRST-CROSSING rule (what the
    figure code implements). Passes if the submitted value lies within the
    Monte Carlo uncertainty across seeds (mean +/- max(3*SEM, 2% tolerance)).
    """
    print("\n=== BASELINE REPRODUCTION (main-figure settings, first-crossing rule) ===")
    print(f"  dt={dt}, T={T} steps (T_phys={T*dt}), max_lag={max_lag} "
          f"(lag window={max_lag*dt}), N={n_traj}, seeds={n_seeds}")
    res, rho_store = run_condition(n_traj, dt, T, max_lag, n_seeds, device, acf_method,
                                   label='baseline_reproduction')
    report = {'settings': {'dt': dt, 'T': T, 'T_phys': T * dt, 'max_lag': max_lag,
                           'phys_lag_window': max_lag * dt, 'N': n_traj, 'n_seeds': n_seeds},
              'submitted': SUBMITTED, 'per_dist': {}, 'passed': True}
    for dname in DIST_NAMES:
        fc = res['per_dist'][dname]['first_crossing']
        target = SUBMITTED[dname]
        mean, sem, sd = fc['S_mean'], fc['S_sem'], fc['S_sd']
        # tolerance: MC uncertainty (3 SEM) or 2% of the submitted value, whichever is wider
        tol = max(3.0 * sem, 0.02 * target)
        dev = abs(mean - target)
        ok = dev <= tol
        report['per_dist'][dname] = {
            'submitted': target, 'reproduced_mean': mean, 'sd': sd, 'sem': sem,
            'abs_deviation': dev, 'tolerance': tol, 'rel_deviation': dev / target,
            'within_mc_uncertainty': bool(ok),
            'S_per_seed': fc['S_per_seed'],
            'seed0_value': fc['S_per_seed'][0],
            'stop_lag_phys_mean': fc['stop_lag_phys_mean'],
            'window_limited_any': fc['window_limited_any'],
        }
        report['passed'] &= ok
        flag = 'OK' if ok else 'MISMATCH'
        print(f"  {dname:>9}: submitted={target:.3f}  reproduced={mean:.4f}+/-{sd:.4f} "
              f"(seed0={fc['S_per_seed'][0]:.4f})  |dev|={dev:.4f}  tol={tol:.4f}  -> {flag}")
    o = res['ordering']['first_crossing']
    report['ordering'] = res['ordering']
    print(f"  ordering S_ellipse > S_Gaussian: {o['ordering_preserved_mean']} "
          f"(ratio={o['ratio_ellipse_over_gaussian']:.3f}, "
          f"{o['n_seeds_ordering_preserved']}/{o['n_seeds']} seeds)")
    if not report['passed']:
        msg = ("BASELINE DID NOT REPRODUCE within Monte Carlo uncertainty. "
               "Per the P2 instructions this is a STOP-AND-INVESTIGATE condition; "
               "the grid was NOT run. Investigate before proceeding.")
        print("\n*** " + msg + " ***")
        if strict:
            raise SystemExit(2)
    else:
        print("  BASELINE REPRODUCTION: PASSED")
    return report, res, rho_store


# ────────────────────────── main ──────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='small CPU smoke test')
    ap.add_argument('--test', action='store_true', help='unit tests only')
    ap.add_argument('--baseline_only', action='store_true', help='baseline reproduction gate only')
    ap.add_argument('--device', default='cpu', help="'cpu' (default) or 'cuda'")
    ap.add_argument('--acf_method', default='loop', choices=['loop', 'fft'],
                    help="autocorrelation kernel; 'loop' (default, original). 'fft' triggers "
                         "a mandatory equivalence check against 'loop' first.")
    ap.add_argument('--n_seeds', type=int, default=None)
    ap.add_argument('--out_tag', default='')
    ap.add_argument('--allow_baseline_mismatch', action='store_true',
                    help='continue past a failed baseline reproduction (records the failure)')
    args = ap.parse_args()

    if args.test:
        unit_test_stopping_rules()
        check_rng_matches_reference()
        print("UNIT TESTS OK")
        return 0

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit(f"ERROR: --device {device} requested but torch.cuda.is_available() is False")
    torch.set_num_threads(os.cpu_count())

    if args.smoke:
        n_seeds = args.n_seeds or 2
        dt_grid = [0.01, 0.02]
        n_grid = [100, 200]
        base_dt, base_n = 0.01, 200
        base_T_phys, base_lag_window = 10.0, 8.0
        horizon_grid = [4.0, 8.0]
        tag = '_smoke'
    else:
        n_seeds = args.n_seeds or N_SEEDS
        dt_grid, n_grid = DT_GRID, N_GRID
        base_dt, base_n = BASE_DT, BASE_N
        base_T_phys, base_lag_window = BASE_T_PHYS, BASE_LAG_WINDOW
        horizon_grid = HORIZON_GRID
        tag = args.out_tag

    print(f"=== P2 dispersion-pair numerical robustness {'(SMOKE)' if args.smoke else ''} ===")
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device.startswith('cuda') else ""))
    print(f"acf_method={args.acf_method}")
    print(f"distributions: {[get_distribution(d)['label'] for d in DIST_NAMES]}")
    print(f"baseline: dt={base_dt}, N={base_n}, T_phys={base_T_phys}, lag window={base_lag_window}, seeds={n_seeds}")
    print(f"grids: dt={dt_grid}  N={n_grid}  horizon={horizon_grid}")

    checks = {}
    print("\n=== CHECKS (pre-run) ===")
    checks['unit_test_stopping_rules'] = bool(unit_test_stopping_rules())
    checks['rng_matches_reference'] = bool(check_rng_matches_reference(device))
    if args.acf_method == 'fft':
        r = verify_fft_against_loop(device)
        checks['fft_matches_loop'] = r
        if r is False:
            raise SystemExit("ERROR: FFT ACF path does not match the loop kernel; aborting.")
    else:
        checks['fft_matches_loop'] = 'not used (loop kernel)'

    t_all = time.time()

    # ── Gate: reproduce the submitted baseline BEFORE any grid ──
    base_T = int(round(base_T_phys / base_dt))
    base_max_lag = int(round(base_lag_window / base_dt))
    if args.smoke:
        # smoke cannot hit the submitted values (tiny N/horizon); run non-strict
        baseline_report, base_res, base_rho = reproduce_baseline(
            n_seeds, device, args.acf_method, base_dt, base_T, base_max_lag, base_n, strict=False)
        baseline_report['note'] = ('SMOKE: settings differ from the main figure, so the submitted '
                                   'values are NOT expected to reproduce here; gate is non-strict.')
    else:
        baseline_report, base_res, base_rho = reproduce_baseline(
            n_seeds, device, args.acf_method, base_dt, base_T, base_max_lag, base_n,
            strict=not args.allow_baseline_mismatch)
    checks['baseline_reproduction_passed'] = bool(baseline_report['passed'])

    if args.baseline_only:
        print("\n--baseline_only: stopping after the reproduction gate.")
        return 0 if baseline_report['passed'] else 2

    conditions = {}
    rho_all = {}
    bkey = f"base_dt{str(base_dt).replace('.','p')}_N{base_n}"
    conditions[bkey] = base_res
    rho_all[bkey] = base_rho

    # ── dt study: matched physical sim horizon AND matched physical lag window ──
    print(f"\n=== dt STUDY (N={base_n}; T_phys={base_T_phys} and lag window={base_lag_window} held FIXED) ===")
    dt_keys = []
    for dt in dt_grid:
        k = f"dt{str(dt).replace('.','p')}_N{base_n}"
        T = int(round(base_T_phys / dt))
        ml = int(round(base_lag_window / dt))
        if abs(dt - base_dt) < 1e-12:
            conditions[k] = conditions[bkey]
            rho_all[k] = rho_all[bkey]
            dt_keys.append(k)
            print(f"  dt={dt} -> reuses baseline condition (T={T}, max_lag={ml})")
            continue
        print(f"  dt={dt}: T={T} steps, max_lag={ml}", flush=True)
        conditions[k], rho_all[k] = run_condition(base_n, dt, T, ml, n_seeds, device,
                                                  args.acf_method, label=f'dt={dt}')
        dt_keys.append(k)

    # ── N study ──
    print(f"\n=== N STUDY (dt={base_dt}, T_phys={base_T_phys}, lag window={base_lag_window}) ===")
    n_keys = []
    for N in n_grid:
        k = f"dt{str(base_dt).replace('.','p')}_N{N}"
        if N == base_n:
            conditions[k] = conditions[bkey]
            rho_all[k] = rho_all[bkey]
            n_keys.append(k)
            print(f"  N={N} -> reuses baseline condition")
            continue
        print(f"  N={N}: floor 1/N={1.0/N:.2e}", flush=True)
        conditions[k], rho_all[k] = run_condition(N, base_dt, base_T, base_max_lag, n_seeds,
                                                  device, args.acf_method, label=f'N={N}')
        n_keys.append(k)

    # ── Horizon study: POST-HOC from the SAME stored baseline rho^2 curve ──
    print(f"\n=== HORIZON STUDY (post-hoc truncation of the SAME baseline rho^2 curve) ===")
    horizon_results = {}
    for W in horizon_grid:
        ml = int(round(W / base_dt))
        if ml > base_max_lag:
            print(f"  window={W}: SKIPPED (needs max_lag={ml} > stored {base_max_lag})")
            continue
        entry = {'phys_lag_window': float(W), 'max_lag': ml, 'N': base_n, 'dt': base_dt,
                 'per_dist': {}}
        for dname in DIST_NAMES:
            curves = rho_all[bkey][dname]
            evs = [evaluate_curve(curves[si].astype(np.float64), base_dt, base_n, max_lag=ml)
                   for si in range(curves.shape[0])]
            per = {}
            for rname in RULES:
                S = np.array([e[rname]['S'] for e in evs])
                per[rname] = {
                    'S_mean': float(S.mean()),
                    'S_sd': float(S.std(ddof=1)) if len(S) > 1 else 0.0,
                    'S_per_seed': S.tolist(),
                    'stop_lag_phys_mean': float(np.mean([e[rname]['stop_lag_phys'] for e in evs])),
                    'window_limited_any': bool(any(e[rname]['window_limited'] for e in evs)),
                    'window_limited_all': bool(all(e[rname]['window_limited'] for e in evs)),
                }
            entry['per_dist'][dname] = per
        add_ordering(entry)
        horizon_results[f'W{int(W)}'] = entry
        for rname in RULES:
            o = entry['ordering'][rname]
            e_ = entry['per_dist']['ellipses'][rname]
            g_ = entry['per_dist']['gaussian'][rname]
            print(f"  window={W:5.1f} [{rname:>14}] S_gauss={g_['S_mean']:.4f} "
                  f"S_ell={e_['S_mean']:.4f} ratio={o['ratio_ellipse_over_gaussian']:.3f} "
                  f"ordering={o['ordering_preserved_mean']}")

    # ── Post-run structural checks ──
    dt_horizons = {k: (conditions[k]['T_phys'], conditions[k]['phys_lag_window']) for k in dt_keys}
    checks['dt_same_physical_sim_horizon'] = all(abs(v[0] - base_T_phys) < 1e-9 for v in dt_horizons.values())
    checks['dt_same_physical_lag_window'] = all(abs(v[1] - base_lag_window) < 1e-9 for v in dt_horizons.values())
    checks['dt_horizons'] = {k: {'T_phys': v[0], 'lag_window': v[1]} for k, v in dt_horizons.items()}
    checks['baseline_shared_across_studies'] = bool(
        conditions[f"dt{str(base_dt).replace('.','p')}_N{base_n}"] is conditions[bkey])
    checks['floor_recomputed_per_N'] = {
        k: abs(conditions[k]['per_dist']['gaussian']['floor_1N'] - 1.0 / conditions[k]['N']) < 1e-15
        for k in n_keys}

    # ordering preserved everywhere?
    ordering_flags = {}
    for k, c in conditions.items():
        for rname in RULES:
            ordering_flags[f"{k}|{rname}"] = c['ordering'][rname]['ordering_preserved_all_seeds']
    for k, c in horizon_results.items():
        for rname in RULES:
            ordering_flags[f"horizon_{k}|{rname}"] = c['ordering'][rname]['ordering_preserved_all_seeds']
    checks['ordering_preserved_everywhere'] = bool(all(ordering_flags.values()))
    checks['ordering_failures'] = [k for k, v in ordering_flags.items() if not v]

    print("\n=== CHECKS (post-run) ===")
    print(f"[3] both distributions x {len(dt_grid)} dt x {len(n_grid)} N: ran {len(conditions)} conditions")
    print(f"[4] dt runs share physical SIM horizon ({base_T_phys}): {checks['dt_same_physical_sim_horizon']}")
    print(f"[4b] dt runs share physical LAG window ({base_lag_window}): {checks['dt_same_physical_lag_window']}")
    print(f"     {checks['dt_horizons']}")
    print(f"[5] baseline reproduction passed: {checks['baseline_reproduction_passed']}")
    print(f"[6] both stopping rules unit-tested: {checks['unit_test_stopping_rules']}")
    print(f"[7] FFT vs loop: {checks['fft_matches_loop']}")
    print(f"[rng] CPU-RNG matches reference simulate_d_trajectories: {checks['rng_matches_reference']}")
    print(f"[floor] 1/N recomputed per N: {checks['floor_recomputed_per_N']}")
    print(f"[ORDERING] S_ellipse > S_Gaussian preserved in EVERY cell/rule/seed: "
          f"{checks['ordering_preserved_everywhere']}")
    if checks['ordering_failures']:
        print(f"           FAILURES: {checks['ordering_failures']}")

    out = {
        'description': 'P2 numerical robustness of the main-figure dispersion vs fragmentation pair',
        'distributions': {d: get_distribution(d)['label'] for d in DIST_NAMES},
        'primary_question': 'does the ordering S_ellipse > S_Gaussian survive dt / N / horizon changes',
        'stopping_rules': {
            'first_crossing': 'FIRST lag with rho^2 < 1/N (paper eq:tmax_stopping, fig_dispersion.py) '
                              '-> REPRODUCTION of submitted numbers',
            'largest_lag': 'LARGEST lag with rho^2 > 1/N (alternative rule) -> SENSITIVITY',
            'note': 'both computed from the SAME stored rho^2 curve',
        },
        'baseline_settings': {'dt': base_dt, 'N': base_n, 'T_phys': base_T_phys,
                              'phys_lag_window': base_lag_window, 'n_seeds': n_seeds,
                              'source': 'experiments/synthetic/fig_dispersion.py'},
        'submitted_values': SUBMITTED,
        'baseline_reproduction': baseline_report,
        'dt_grid': dt_grid, 'N_grid': n_grid, 'horizon_grid': horizon_grid,
        'seed_rule': 'seed = 42 + dist_offset(gaussian:200, ellipses:100) + 1000*seed_index; '
                     'seed_index 0 reproduces fig_dispersion.py exactly',
        'device': str(device), 'acf_method': args.acf_method,
        'rng_policy': 'all RNG on CPU generator; tensors moved to device; no device-side RNG',
        'conditions': conditions, 'horizon_study': horizon_results,
        'dt_keys': dt_keys, 'n_keys': n_keys, 'baseline_key': bkey,
        'checks': checks,
        'appendix_note': 'The four-sweep Delta-t robustness result (Appendix Fig. A3) is CITED, '
                         'not re-presented; this pair experiment makes the main visual example '
                         'directly interpretable.',
        'total_wall_s': time.time() - t_all,
    }

    results_dir = os.path.join(CODEDIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, f'p2_dispersion_robustness{tag}.json')
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved JSON -> {out_json}")

    npz_data = {}
    for k, store in rho_all.items():
        for dname, arr in store.items():
            npz_data[f'{k}__{dname}'] = arr
    out_npz = os.path.join(results_dir, f'p2_dispersion_robustness{tag}_rho_sq.npz')
    np.savez_compressed(out_npz, **npz_data)
    print(f"Saved rho_sq -> {out_npz}")

    md = build_markdown(out)
    out_md = os.path.join(results_dir, f'p2_dispersion_robustness{tag}_table.md')
    with open(out_md, 'w') as f:
        f.write(md)
    print(f"Saved table -> {out_md}")
    print("\n" + md)
    print(f"Total wall: {out['total_wall_s']:.1f}s")

    all_pass = (checks['unit_test_stopping_rules'] and checks['rng_matches_reference']
                and checks['dt_same_physical_sim_horizon'] and checks['dt_same_physical_lag_window']
                and all(checks['floor_recomputed_per_N'].values()))
    print(f"\nALL STRUCTURAL CHECKS PASSED: {all_pass}")
    print(f"ORDERING PRESERVED EVERYWHERE: {checks['ordering_preserved_everywhere']}")
    return 0


def build_markdown(out):
    conds, hz = out['conditions'], out['horizon_study']
    b = out['baseline_settings']
    lines = []
    lines.append("### P2 numerical robustness: dispersion vs fragmentation (main-figure pair)\n")
    lines.append(f"Distributions: isotropic Gaussian N(0,25I) vs equal-weight parallel-ellipses GMM "
                 f"(means (0,±3), component covariance diag(25,1)) — the exact pair from "
                 f"`fig_dispersion.py`. Baseline: dt={b['dt']}, N={b['N']}, physical simulation "
                 f"horizon {b['T_phys']}, physical lag window {b['phys_lag_window']}, "
                 f"{b['n_seeds']} seeds (mean ± sd).\n")
    br = out['baseline_reproduction']
    lines.append(f"**Baseline reproduction (first-crossing rule, the paper's rule, as implemented by the "
                 f"figure code):** Gaussian {br['per_dist']['gaussian']['reproduced_mean']:.3f} "
                 f"(submitted {br['submitted']['gaussian']}), ellipses "
                 f"{br['per_dist']['ellipses']['reproduced_mean']:.3f} "
                 f"(submitted {br['submitted']['ellipses']}) — "
                 f"{'PASSED' if br['passed'] else 'MISMATCH'}.\n")
    lines.append("Ŝ is a **finite-horizon** quantity throughout; † marks a window-limited entry "
                 "(stopping lag reached the edge of the lag window), which is a lower bound, not a "
                 "resolved population value. Two stopping rules are computed from the *same* stored "
                 "ρ² curve: **first-crossing** = the paper's rule (and the figure code); **largest-lag** "
                 "= an alternative rule, as a sensitivity check.\n")

    def row(tag_, c, rname):
        g = c['per_dist']['gaussian'][rname]
        e = c['per_dist']['ellipses'][rname]
        o = c['ordering'][rname]
        gm = '†' if g['window_limited_any'] else ''
        em = '†' if e['window_limited_any'] else ''
        return (f"| {tag_} | {g['S_mean']:.3f}±{g['S_sd']:.3f}{gm} | {e['S_mean']:.3f}±{e['S_sd']:.3f}{em} | "
                f"{o['ratio_ellipse_over_gaussian']:.2f}× | "
                f"**{'YES' if o['ordering_preserved_all_seeds'] else 'NO'}** | "
                f"{g['stop_lag_phys_mean']:.1f} / {e['stop_lag_phys_mean']:.1f} |")

    for rname, rlabel in [('first_crossing', 'first-crossing 1/N (paper rule)'),
                          ('largest_lag', 'largest-qualifying-lag 1/N (sensitivity)')]:
        lines.append(f"\n**Stopping rule: {rlabel}**\n")
        lines.append("| condition | Ŝ Gaussian | Ŝ ellipses | ratio | ordering preserved | τ* G / E |")
        lines.append("|---|---|---|---|---|---|")
        for k in out['dt_keys']:
            lines.append(row(f"dt={conds[k]['dt']}", conds[k], rname))
        for k in out['n_keys']:
            lines.append(row(f"N={conds[k]['N']}", conds[k], rname))
        for hk in sorted(hz, key=lambda s: int(s[1:])):
            lines.append(row(f"lag window={hz[hk]['phys_lag_window']:g}", hz[hk], rname))

    # cost table
    lines.append("\n**Computational cost (score evaluations and wall time reported separately)**\n")
    lines.append("| condition | N | dt | steps T | max_lag | score evals | wall (s) |")
    lines.append("|---|---|---|---|---|---|---|")
    seen = set()
    for k in out['dt_keys'] + out['n_keys']:
        if k in seen:
            continue
        seen.add(k)
        c = conds[k]
        lines.append(f"| {c['label']} | {c['N']} | {c['dt']} | {c['T']} | {c['max_lag']} | "
                     f"{c['score_evals']:.3e} | {c['wall_s']:.1f} |")
    lines.append("\n(The horizon study re-uses the stored baseline ρ² curve and costs no extra "
                 "score evaluations.)")
    ck = out['checks']
    lines.append(f"\n**Ordering S_ellipse > S_Gaussian preserved in every cell, both rules, all "
                 f"seeds: {ck['ordering_preserved_everywhere']}**")
    lines.append(f"\nThe existing four-sweep Δt-robustness result (Appendix Fig. A3) is cited, not "
                 f"re-presented here.")
    lines.append("")
    return "\n".join(lines)


if __name__ == '__main__':
    with open(__file__) as f:
        ast.parse(f.read())
    sys.exit(main())
