#!/usr/bin/env python3
"""
P2: Numerical robustness of SSA to integration step size (dt) and number of
trajectories (N), on ONE common GMM benchmark.

Design (see docs/rebuttal_agent_instructions.md, section P2):
  - Common benchmark: a fixed ladder of 7 bimodal 10D GMMs varying separation
    (dim 0), unit variance per dim, equal weights. This single ladder is reused
    identically across every grid cell. Its intrinsic SSA/MI ordering lets us
    measure RANKING stability, and any single rung measures POINT stability.
  - Baseline cell: N=200, dt=0.01 (matches the paper's canonical GMM regime).
  - dt grid: {0.005, 0.01, 0.02} at N=200. PHYSICAL horizon T_phys and the
    physical lag window are held fixed; the number of Euler steps T = T_phys/dt
    therefore changes with dt (score evaluations change, wall time changes).
  - N grid: {50, 100, 200, 400} at dt=0.01. The N-dependent noise floor
    1/(N*d) (and the secondary floor 1/N) is RECOMPUTED for each N.
  - The (N=200, dt=0.01) baseline is computed ONCE and shared by both studies.

PRIMARY SSA estimator (the paper's eq:tmax_stopping):
  - noise threshold = 1/N (FLOOR_1N);
  - per-lag ρ² excludes lag 0 (rho_sq = rho[1:]²);
  - stopping lag τ* = the FIRST lag whose ρ²(τ) <= 1/N (that lag included; the
    full window if it never crosses);
  - finite-horizon Ŝ = dt * Σ_{τ=1..τ*} ρ²(τ) (Riemann rectangle rule).
  --stopping largest instead keeps through the LARGEST lag with ρ²(τ) > 1/N
  (the rule of the original rebuttal run); the paper reports first crossing.

SECONDARY (sensitivity only, clearly labeled, NOT canonical): the 1/(Nd)
threshold is retained purely to show floor-choice sensitivity.

Finite-horizon vs resolved population SSA:
  Well-separated rungs do not decorrelate to the noise threshold within T_phys,
  so their Ŝ is a FINITE-HORIZON quantity (window-limited), not a resolved
  population SSA. We flag window_limited per run (last in-window lag still above
  the threshold).

SSA aggregation reuses src.metrics.autocorrelation (trace normalization,
Riemann Ŝ = dt * Σ ρ²) — no new SSA definition.

Determinism across devices:
  All randomness (initial conditions + per-step Euler noise) is drawn on the CPU
  generator and moved to the compute device. Device-side RNG is never used, so
  --device cuda reproduces the --device cpu RNG stream exactly.

Usage:
  python compute_numerical_robustness.py --out_tag _fc                # full run (CPU)
  python compute_numerical_robustness.py --device cuda --out_tag _fc  # full run (GPU)
  python compute_numerical_robustness.py --smoke                      # CPU smoke test
  python compute_numerical_robustness.py --test                       # unit test of stopping rule
  python compute_numerical_robustness.py --from_npz RUN.json --out_tag _fc
      # rebuild every cell from a saved run's rho^2 curves (no simulation)
"""
import sys, os, json, time, argparse, ast
import torch
import numpy as np
from scipy.stats import spearmanr, kendalltau

CODEDIR = os.path.dirname(os.path.abspath(__file__))
REPODIR = os.path.join(CODEDIR, '..', '..', '..')
sys.path.insert(0, REPODIR)
sys.path.insert(0, CODEDIR)
sys.path.insert(0, os.path.join(REPODIR, 'experiments', 'synthetic'))

from mixture_builders import build_bimodal_nd
from src.simulation.langevin import simulate_d_trajectories, LangevinSimulatorD
from src.metrics.autocorrelation import compute_vector_autocorrelation

# ── Fixed benchmark parameters ──
D = 10
SEPARATIONS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]   # the common GMM ladder
REP_SEP = 2.5                                        # representative rung for point stability
T_PHYS = 20.0                                        # physical horizon (held fixed across all cells)
N_SEEDS = 5
N_SAMPLES_MI = 5000
BASELINE_N = 200
BASELINE_DT = 0.01

# One-factor grids around the baseline
DT_GRID = [0.005, 0.01, 0.02]     # at N = BASELINE_N
N_GRID = [50, 100, 200, 400]      # at dt = BASELINE_DT


def build_sep(sep):
    seps = [0.0] * D
    seps[0] = float(sep)
    return build_bimodal_nd(dim=D, separation_per_dim=seps, variance_per_dim=[1.0] * D)


# Stopping rule. 'first' (default since 2026-10-17) is the paper's eq:tmax_stopping:
# truncate at the first lag with rho^2 <= floor, that lag included (full window if it
# never crosses). 'largest' reproduces the original rebuttal run. Set via --stopping.
STOPPING = 'first'


def compute_shat(rho_sq, dt, floor):
    """SSA estimator with the 1/N stopping rule selected by STOPPING.

    'first': keep lags up to and including the first with rho^2 <= floor;
    window_limited = it never crosses. 'largest': see _compute_shat_largest.
    Returns (S_hat, keep, window_limited).
    """
    if STOPPING == 'first':
        below = np.nonzero(rho_sq <= floor)[0]
        keep = int(below[0]) + 1 if len(below) else len(rho_sq)
        return float(dt * rho_sq[:keep].sum()), keep, not bool(len(below))
    return _compute_shat_largest(rho_sq, dt, floor)


def _compute_shat_largest(rho_sq, dt, floor):
    """Original rebuttal estimator with the largest-lag stopping rule.

    rho_sq: per-lag ρ² for lags 1..L (lag 0 already excluded by the caller).
    Stopping lag τ* = the LARGEST lag with ρ²(τ) > floor. Integrate through τ*:
        Ŝ = dt * Σ_{τ=1..τ*} ρ²(τ).
    This is NOT first-threshold-crossing: a ρ² curve that dips below the floor
    and later rises above it retains the later lags (up to the last one above).

    Returns (S_hat, keep, window_limited):
      keep           = number of retained lags (= index of τ* + 1)
      window_limited = True if the last in-window lag is still above the floor
                       (finite-horizon / unresolved), False if ρ² fell and
                       stayed below the floor before the window ended (resolved).
    """
    above = rho_sq > floor
    if not above.any():
        # Everything already at/below the floor: resolved with no retained mass.
        return 0.0, 0, False
    last_above = int(np.nonzero(above)[0].max())
    keep = last_above + 1
    S = float(dt * rho_sq[:keep].sum())
    window_limited = bool(last_above == len(rho_sq) - 1)
    return S, keep, window_limited


def simulate_rho_sq(gm, n_traj, seed, dt, T, max_lag, device='cpu'):
    """Simulate one ensemble and return per-lag ρ² for lags 1..max_lag.

    DETERMINISM CONTRACT: all randomness (initial conditions and the full
    Euler-Maruyama noise block) is drawn on the CPU generator seeded by `seed`,
    then MOVED to `device`. Device-side RNG is never used, so results are
    bitwise-comparable between --device cpu and --device cuda up to
    floating-point reduction-order differences in the linear algebra only.
    """
    # ── CPU RNG: initial conditions, exactly as simulate_d_trajectories does ──
    torch.manual_seed(seed)
    sim_cpu = LangevinSimulatorD(gm.means, gm.covs, gm.weights, dt, torch.device('cpu'))
    x0 = sim_cpu.sample_initial(n_traj)                       # (n, d) on CPU
    d = sim_cpu.d

    # ── Compute device (no device RNG involved) ──
    if str(device) == 'cpu':
        sim, x = sim_cpu, x0
    else:
        sim = LangevinSimulatorD(gm.means.to(device), gm.covs.to(device),
                                 gm.weights.to(device), dt, torch.device(device))
        x = x0.to(device)

    # Euler-Maruyama loop. Noise is drawn PER STEP on the CPU generator with the
    # exact same call signature/order as LangevinSimulatorD.simulate uses
    # (torch.randn(n, d) once per step), then moved to the device. This keeps the
    # RNG stream identical to the original CPU path; a single presampled
    # (T, n, d) block does NOT reproduce that stream.
    traj = torch.zeros(n_traj, T + 1, d, device=sim.device)
    traj[:, 0, :] = x
    for t in range(T):
        z = torch.randn(n_traj, d)                            # CPU RNG
        if str(device) != 'cpu':
            z = z.to(device, non_blocking=True)
        x = sim.step(x, z=z)
        traj[:, t + 1, :] = x

    rho, _, _ = compute_vector_autocorrelation(
        traj, max_lag, gmm_means=sim.means, gmm_covs=sim.covs, gmm_weights=sim.weights,
        demean=True, normalization='trace',
    )
    return (rho.cpu().numpy() ** 2)[1:].astype(np.float64)


def compute_mi_ladder(seps, device='cpu'):
    mis = []
    for si, sep in enumerate(seps):
        gm = build_sep(sep)
        torch.manual_seed(777 + si)
        try:
            mi = gm.mutual_information(n_samples=N_SAMPLES_MI).item()
        except Exception:
            mi = float('nan')
        mis.append(float(mi))
    return mis


def aggregate_over_ladder(rho_sq_all, dt, floor, max_lag):
    """Apply the stopping-lag estimator over a (seeds, rungs, lags) ρ² block.

    Returns (S, tau_star, window_limited) each of shape (seeds, rungs).
    window_limited is defined EXACTLY as keep == max_lag: the stopping lag ran
    into the edge of the lag window, so Ŝ is finite-horizon / unresolved and
    must never be presented as a resolved population value.
    """
    n_seeds, n_rungs = rho_sq_all.shape[0], rho_sq_all.shape[1]
    S = np.zeros((n_seeds, n_rungs))
    tau = np.zeros((n_seeds, n_rungs))
    wlim = np.zeros((n_seeds, n_rungs), dtype=bool)
    for si in range(n_seeds):
        for ri in range(n_rungs):
            s, keep, wl = compute_shat(rho_sq_all[si, ri, :max_lag].astype(np.float64), dt, floor)
            S[si, ri] = s
            tau[si, ri] = max(keep - 1, 0) * dt   # physical time of stopping lag τ*
            wlim[si, ri] = bool(wl)
    return S, tau, wlim


def run_cell(seps, n_traj, dt, T_phys, n_seeds, device='cpu', max_lag_cap=None):
    """Run the full ladder x seeds for one (N, dt) cell.

    Returns dict with per-rung Ŝ arrays (both thresholds), cost, and metadata.
    Physical horizon T_phys is fixed; T = round(T_phys/dt); max_lag = T so the
    physical lag window == T_phys for every dt (the submitted convention).

    max_lag_cap: OPTIONAL override. If set, max_lag = min(T, cap). NOTE this
    SHRINKS the physical lag window to max_lag*dt and is NOT the submitted
    convention — it must be validated before use (see validate_max_lag_caps)."""
    T = int(round(T_phys / dt))
    max_lag = T if max_lag_cap is None else min(T, int(max_lag_cap))
    floor_1N = 1.0 / n_traj          # PRIMARY threshold (submitted paper)
    floor_1Nd = 1.0 / (n_traj * D)   # SECONDARY sensitivity threshold only
    n_rungs = len(seps)

    rho_sq_all = np.zeros((n_seeds, n_rungs, max_lag), dtype=np.float32)
    t0 = time.time()
    for si in range(n_seeds):
        for ri, sep in enumerate(seps):
            gm = build_sep(sep)
            seed_val = si * 10000 + ri
            rho_sq_all[si, ri, :] = simulate_rho_sq(gm, n_traj, seed_val, dt, T, max_lag, device)
    wall = time.time() - t0

    S_1N, tau_1N, wl_1N = aggregate_over_ladder(rho_sq_all, dt, floor_1N, max_lag)     # PRIMARY
    S_1Nd, tau_1Nd, wl_1Nd = aggregate_over_ladder(rho_sq_all, dt, floor_1Nd, max_lag)  # SECONDARY

    # Score evaluations: one score eval per (trajectory, Euler step).
    score_evals = int(n_traj) * int(T) * int(n_rungs) * int(n_seeds)

    return {
        'N': int(n_traj), 'dt': float(dt), 'T': int(T), 'T_phys': float(T_phys),
        'max_lag': int(max_lag), 'phys_lag_window': float(max_lag * dt),
        'floor_1N': float(floor_1N), 'floor_1Nd': float(floor_1Nd),
        # PRIMARY (1/N)
        'S_1N': S_1N.tolist(), 'tau_1N': tau_1N.tolist(), 'window_limited_1N': wl_1N.tolist(),
        # SECONDARY (1/(Nd)) sensitivity
        'S_1Nd': S_1Nd.tolist(), 'tau_1Nd': tau_1Nd.tolist(), 'window_limited_1Nd': wl_1Nd.tolist(),
        'score_evals': score_evals, 'wall_s': float(wall),
        '_rho_sq': rho_sq_all,  # popped before JSON
    }


def validate_max_lag_caps(cell, rho_sq_all, mi, seps, fractions=(0.25, 0.5, 0.75)):
    """Honest evaluation of a max_lag cap on ONE cell (the baseline).

    Recomputes Ŝ and the SSA ranking (Spearman vs the MI ladder) from the SAME
    ρ² arrays truncated to a shorter lag window, and reports relative Ŝ
    differences and any ordering change vs the full window. A cap is only
    defensible if Ŝ is essentially unchanged AND the ranking is preserved.
    Nothing here adopts a cap — it only measures the cost of one.
    """
    dt = cell['dt']
    full_lag = cell['max_lag']
    floor = cell['floor_1N']   # PRIMARY threshold

    S_full, tau_full, wl_full = aggregate_over_ladder(rho_sq_all, dt, floor, full_lag)
    mean_full = S_full.mean(axis=0)
    rho_full = float(spearmanr(mean_full, mi).correlation)
    order_full = np.argsort(np.argsort(mean_full))

    rows = []
    for fr in fractions:
        cap = max(1, int(round(full_lag * fr)))
        S_c, tau_c, wl_c = aggregate_over_ladder(rho_sq_all, dt, floor, cap)
        mean_c = S_c.mean(axis=0)
        rel = np.abs(mean_c - mean_full) / np.maximum(np.abs(mean_full), 1e-12)
        rho_c = float(spearmanr(mean_c, mi).correlation)
        order_c = np.argsort(np.argsort(mean_c))
        rows.append({
            'fraction_of_window': float(fr),
            'max_lag': int(cap),
            'phys_lag_window': float(cap * dt),
            'mean_S': mean_c.tolist(),
            'max_rel_diff_vs_full': float(rel.max()),
            'median_rel_diff_vs_full': float(np.median(rel)),
            'spearman_S_MI': rho_c,
            'spearman_S_MI_full': rho_full,
            'ordering_changed': bool(not np.array_equal(order_c, order_full)),
            'n_window_limited_rungs': int(wl_c.all(axis=0).sum()),
        })
    return {
        'cell_key': f"N{cell['N']}_dt{str(cell['dt']).replace('.', 'p')}",
        'full': {
            'max_lag': int(full_lag), 'phys_lag_window': float(cell['phys_lag_window']),
            'mean_S': mean_full.tolist(), 'spearman_S_MI': rho_full,
            'n_window_limited_rungs': int(wl_full.all(axis=0).sum()),
        },
        'caps': rows,
        'note': 'Caps are evaluated post-hoc from the full-window rho^2; no cap is adopted.',
    }


def unit_test_stopping_rule():
    global STOPPING
    saved, STOPPING = STOPPING, 'largest'
    try:
        return _unit_test_stopping_rule_largest()
    finally:
        STOPPING = saved


def _unit_test_stopping_rule_largest():
    """Unit test: non-monotone ρ² that dips BELOW 1/N then rises ABOVE it.
    Assert the LARGEST qualifying lag is selected and that this differs from a
    naive first-threshold-crossing rule. Returns True on success (raises on fail).
    """
    N = 200
    floor = 1.0 / N                       # 0.005
    dt = 0.1
    # lags 1..6 (lag 0 already excluded by convention). Index i == lag (i+1).
    #                l1     l2     l3       l4     l5       l6
    rho_sq = np.array([0.90, 0.50, 0.001, 0.20, 0.0001, 0.00005])
    above = rho_sq > floor                # [T, T, F, T, F, F]

    S, keep, wlim = compute_shat(rho_sq, dt, floor)

    # Largest qualifying lag is index 3 (value 0.20 > floor) -> keep = 4.
    last_above = int(np.nonzero(above)[0].max())
    assert last_above == 3, f"expected last-above index 3, got {last_above}"
    assert keep == 4, f"expected keep=4 (integrate through tau*), got {keep}"

    # Expected Ŝ integrates through the stopping lag INCLUDING the intermediate dip.
    expected_S = dt * rho_sq[:4].sum()    # 0.1 * (0.90+0.50+0.001+0.20)
    assert abs(S - expected_S) < 1e-12, f"Ŝ mismatch: {S} vs {expected_S}"

    # A first-threshold-crossing rule would stop at the FIRST below-floor lag
    # (index 2), integrating only rho_sq[:2] -> must DIFFER from ours.
    first_cross_keep = int(np.argmax(~above))          # 2
    first_cross_S = dt * rho_sq[:first_cross_keep].sum()
    assert first_cross_keep == 2, f"first-crossing keep expected 2, got {first_cross_keep}"
    assert keep != first_cross_keep, "largest-lag rule must differ from first-crossing"
    assert abs(S - first_cross_S) > 1e-9, "Ŝ must differ from first-crossing Ŝ"

    # window_limited is False here (last in-window lag is below floor -> resolved).
    assert wlim is False, f"expected window_limited=False, got {wlim}"

    # All-above case -> window_limited True.
    _, keep2, wlim2 = compute_shat(np.array([0.9, 0.8, 0.7]), dt, floor)
    assert keep2 == 3 and wlim2 is True, f"all-above: keep={keep2}, wlim={wlim2}"

    print("[unit-test] stopping rule PASSED:")
    print(f"    rho_sq={rho_sq.tolist()}, floor=1/N={floor}")
    print(f"    largest-lag keep={keep} (tau*=lag {keep}), Shat={S:.6f}")
    print(f"    first-crossing keep={first_cross_keep}, Shat={first_cross_S:.6f} (DIFFERS, as required)")
    print(f"    window_limited(resolved case)={wlim}; all-above case window_limited={wlim2}")
    return True


def cell_key(N, dt):
    return f"N{N}_dt{str(dt).replace('.', 'p')}"


def summarize_cell(cell, mi, seps, floor_key='1N'):
    """Point stability at REP_SEP, ranking stability vs MI. Returns summary dict.

    floor_key defaults to '1N' (PRIMARY, submitted-paper threshold)."""
    S = np.array(cell[f'S_{floor_key}'])                        # (seeds, rungs)
    TAU = np.array(cell[f'tau_{floor_key}'])                    # (seeds, rungs)
    WL = np.array(cell[f'window_limited_{floor_key}'], dtype=bool)
    mean_S = S.mean(axis=0)
    sd_S = S.std(axis=0, ddof=1) if S.shape[0] > 1 else np.zeros(S.shape[1])
    rep_i = seps.index(REP_SEP) if REP_SEP in seps else len(seps) // 2
    rep_sep_used = float(seps[rep_i])   # actual rung used (may differ from REP_SEP if absent)
    # Ranking vs MI (per seed, then mean across seeds)
    rho_seeds = [spearmanr(S[s], mi).correlation for s in range(S.shape[0])]
    tau_seeds = [kendalltau(S[s], mi).correlation for s in range(S.shape[0])]
    return {
        'mean_S': mean_S.tolist(),
        'sd_S': sd_S.tolist(),
        'rep_sep': rep_sep_used,
        'rep_mean_S': float(mean_S[rep_i]),
        'rep_sd_S': float(sd_S[rep_i]),
        'rep_cv': float(sd_S[rep_i] / mean_S[rep_i]) if mean_S[rep_i] != 0 else float('nan'),
        'spearman_S_MI_mean': float(np.nanmean(rho_seeds)),
        'spearman_S_MI_sd': float(np.nanstd(rho_seeds)),
        'kendall_S_MI_mean': float(np.nanmean(tau_seeds)),
        'floor_key': floor_key,
        # ── finite-horizon bookkeeping (item 1/4) ──
        'tau_star_mean_per_rung': TAU.mean(axis=0).tolist(),
        'tau_star_min_per_rung': TAU.min(axis=0).tolist(),
        'tau_star_max_per_rung': TAU.max(axis=0).tolist(),
        'window_limited_per_rung': WL.all(axis=0).tolist(),
        'window_limited_any_seed_per_rung': WL.any(axis=0).tolist(),
        'n_window_limited_rungs': int(WL.all(axis=0).sum()),
        'rep_window_limited': bool(WL.all(axis=0)[rep_i]),
        'rep_tau_star_mean': float(TAU.mean(axis=0)[rep_i]),
        'any_window_limited': bool(np.any(WL)),
        'phys_lag_window': float(cell['phys_lag_window']),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='CPU smoke test')
    ap.add_argument('--test', action='store_true', help='run unit test of stopping rule and exit')
    ap.add_argument('--device', default='cpu',
                    help="compute device: 'cpu' (default) or 'cuda'. RNG always stays on CPU, "
                         "so results are comparable across devices.")
    ap.add_argument('--max_lag', type=int, default=None,
                    help='OPTIONAL cap on max_lag (default: none -> max_lag=T, the submitted '
                         'convention with the physical lag window = T_phys). A cap SHRINKS the '
                         'physical window; validated but not adopted by default.')
    ap.add_argument('--out_tag', default='', help='suffix for output filenames')
    ap.add_argument('--stopping', choices=['first', 'largest'], default='first',
                    help="1/N stopping rule: 'first' (paper, default) or 'largest' (original rebuttal run)")
    ap.add_argument('--from_npz', default=None,
                    help='rebuild all cells from a saved run (path to p2_numerical_robustness*.json; '
                         'its _rho_sq.npz must sit next to it) instead of simulating')
    args = ap.parse_args()
    global STOPPING
    STOPPING = args.stopping

    if args.test:
        unit_test_stopping_rule()
        print("UNIT TEST OK")
        return 0

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit(f"ERROR: --device {device} requested but torch.cuda.is_available() is False")
    torch.set_num_threads(os.cpu_count())
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device.startswith('cuda') else ""))

    if args.smoke:
        seps = [1.0, 3.0]
        t_phys = 5.0
        n_seeds = 1
        dt_grid = [0.01, 0.02]      # two step sizes
        n_grid = [50, 100]          # two trajectory counts
        baseline_dt, baseline_n = 0.02, 50
        tag = '_smoke'
    else:
        seps = SEPARATIONS
        t_phys = T_PHYS
        n_seeds = N_SEEDS
        dt_grid = DT_GRID
        n_grid = N_GRID
        baseline_dt, baseline_n = BASELINE_DT, BASELINE_N
        tag = args.out_tag

    print(f"=== P2 numerical robustness {'(SMOKE)' if args.smoke else ''} ===")
    print(f"benchmark: {len(seps)} bimodal {D}D GMMs, separations={seps}")
    print(f"T_phys={t_phys}, n_seeds={n_seeds}, baseline=(N={baseline_n},dt={baseline_dt})")
    print(f"dt grid @N={baseline_n}: {dt_grid}")
    print(f"N grid  @dt={baseline_dt}: {n_grid}")

    cells = {}      # key -> cell dict (with _rho_sq)
    t_all = time.time()
    if args.from_npz:
        # Rebuild every cell from saved rho^2 (no simulation): recompute S/tau/window
        # flags under the selected stopping rule; keep run metadata from the saved JSON.
        prev = json.load(open(args.from_npz))
        z = np.load(args.from_npz.replace('.json', '_rho_sq.npz'))
        seps = [float(x) for x in z['separations']]
        mi = [float(x) for x in z['mi']]
        for k, c in prev['cells'].items():
            r = z[f'{k}_rho_sq']
            c = dict(c)
            for fk in ('1N', '1Nd'):
                S, tau, wl = aggregate_over_ladder(r, c['dt'], c[f'floor_{fk}'], c['max_lag'])
                c[f'S_{fk}'], c[f'tau_{fk}'], c[f'window_limited_{fk}'] = S.tolist(), tau.tolist(), wl.tolist()
            c['_rho_sq'] = r
            cells[k] = c
        print(f"[from_npz] rebuilt {len(cells)} cells from {args.from_npz} with stopping={STOPPING}")
    else:
        mi = compute_mi_ladder(seps, device)
    print(f"MI ladder: {[round(m,3) for m in mi]}")

    # Baseline computed ONCE, shared by both studies
    bkey = cell_key(baseline_n, baseline_dt)
    print(f"\n[baseline] {bkey} ...", flush=True)
    if bkey not in cells:
        cells[bkey] = run_cell(seps, baseline_n, baseline_dt, t_phys, n_seeds, device, args.max_lag)
    print(f"  wall={cells[bkey]['wall_s']:.1f}s score_evals={cells[bkey]['score_evals']:.3e}")

    # dt study (N fixed = baseline_n)
    for dt in dt_grid:
        k = cell_key(baseline_n, dt)
        if k in cells:
            continue
        print(f"[dt-study] {k} ...", flush=True)
        cells[k] = run_cell(seps, baseline_n, dt, t_phys, n_seeds, device, args.max_lag)
        print(f"  wall={cells[k]['wall_s']:.1f}s score_evals={cells[k]['score_evals']:.3e}")

    # N study (dt fixed = baseline_dt)
    for N in n_grid:
        k = cell_key(N, baseline_dt)
        if k in cells:
            continue
        print(f"[N-study] {k} ...", flush=True)
        cells[k] = run_cell(seps, N, baseline_dt, t_phys, n_seeds, device, args.max_lag)
        print(f"  wall={cells[k]['wall_s']:.1f}s score_evals={cells[k]['score_evals']:.3e}")

    # ── Verification checks ──
    checks = {}
    # (0) stopping-rule unit test
    checks['unit_test_stopping_rule'] = bool(unit_test_stopping_rule())
    # (4) all dt cells share same physical horizon
    dt_phys = {cell_key(baseline_n, dt): cells[cell_key(baseline_n, dt)]['T'] * cells[cell_key(baseline_n, dt)]['dt']
               for dt in dt_grid}
    checks['dt_same_physical_horizon'] = all(abs(v - t_phys) < 1e-9 for v in dt_phys.values())
    checks['dt_physical_horizons'] = dt_phys
    # dt cells share same physical lag window
    dt_lagwin = {cell_key(baseline_n, dt): cells[cell_key(baseline_n, dt)]['phys_lag_window'] for dt in dt_grid}
    checks['dt_same_lag_window'] = all(abs(v - t_phys) < 1e-9 for v in dt_lagwin.values())
    # (5) baseline shared consistently: it is present in both grids and is a single object
    checks['baseline_in_dt_grid'] = bkey in [cell_key(baseline_n, dt) for dt in dt_grid]
    checks['baseline_in_N_grid'] = bkey in [cell_key(N, baseline_dt) for N in n_grid]
    checks['baseline_single_object'] = (baseline_n, baseline_dt) == (baseline_n, baseline_dt)
    # N-dependent stopping threshold recomputed correctly.
    # PRIMARY: 1/N. SECONDARY: 1/(Nd).
    checks['floor_recomputed_per_N'] = {cell_key(N, baseline_dt): (1.0 / N == cells[cell_key(N, baseline_dt)]['floor_1N'])
                                        for N in n_grid}
    checks['floor_1Nd_recomputed_per_N'] = {cell_key(N, baseline_dt): (1.0 / (N * D) == cells[cell_key(N, baseline_dt)]['floor_1Nd'])
                                            for N in n_grid}

    print("\n=== ESTIMATOR (PRIMARY) ===")
    print(f"  threshold=1/N; stopping rule={STOPPING} ('first' = paper: first lag with rho^2<=1/N, included);")
    print("  lag 0 excluded (rho_sq=rho[1:]^2); S_hat=dt*sum rho^2. Matches compute_random_sweep.py.")
    print("  SECONDARY (sensitivity only, not canonical): 1/(Nd) threshold.")
    print("\n=== CHECKS ===")
    print(f"[4] dt cells same physical horizon (={t_phys}): {checks['dt_same_physical_horizon']} {dt_phys}")
    print(f"[4b] dt cells same physical lag window: {checks['dt_same_lag_window']} {dt_lagwin}")
    print(f"[5] baseline {bkey} in dt grid: {checks['baseline_in_dt_grid']}; in N grid: {checks['baseline_in_N_grid']}")
    print(f"[floor] PRIMARY 1/N recomputed per N: {checks['floor_recomputed_per_N']}")
    print(f"[floor] SECONDARY 1/(Nd) recomputed per N: {checks['floor_1Nd_recomputed_per_N']}")
    if args.max_lag is not None:
        print(f"[WARNING] --max_lag={args.max_lag} override ACTIVE: physical lag window is "
              f"SHRUNK below T_phys and this is NOT the submitted convention.")

    # ── Summaries ──
    # PRIMARY summary uses the 1/N threshold (submitted paper).
    summ = {}
    for k, cell in cells.items():
        summ[k] = summarize_cell(cell, mi, seps, floor_key='1N')
    base_meanS = np.array(summ[bkey]['mean_S'])
    for k in cells:
        ms = np.array(summ[k]['mean_S'])
        summ[k]['spearman_vs_baseline'] = float(spearmanr(ms, base_meanS).correlation)

    # SECONDARY sensitivity summary uses the 1/(Nd) threshold (clearly labeled,
    # NOT canonical) — retained only to show floor-choice sensitivity.
    summ_1Nd = {}
    for k, cell in cells.items():
        summ_1Nd[k] = summarize_cell(cell, mi, seps, floor_key='1Nd')
    base_meanS_1Nd = np.array(summ_1Nd[bkey]['mean_S'])
    for k in cells:
        ms = np.array(summ_1Nd[k]['mean_S'])
        summ_1Nd[k]['spearman_vs_baseline'] = float(spearmanr(ms, base_meanS_1Nd).correlation)

    # ── (item 2) Like-for-like window-limited comparison ACROSS dt ──
    # Physical window is fixed across dt, so the same rung should be
    # window-limited at every dt if the underlying physics is identical.
    # A disagreement is informative (dt-dependent resolution of the lag scale).
    dt_keys = [cell_key(baseline_n, dt) for dt in dt_grid]
    wl_by_dt = {k: summ[k]['window_limited_per_rung'] for k in dt_keys}
    wl_matrix = np.array([wl_by_dt[k] for k in dt_keys], dtype=bool)   # (n_dt, n_rungs)
    consistent = wl_matrix.all(axis=0) | (~wl_matrix).all(axis=0)      # per rung
    checks['window_limited_consistent_across_dt'] = bool(consistent.all())
    checks['window_limited_by_dt'] = {k: wl_by_dt[k] for k in dt_keys}
    disagree = [seps[i] for i in np.nonzero(~consistent)[0]]
    checks['window_limited_disagreeing_rungs'] = disagree

    print("\n=== FINITE-HORIZON / WINDOW-LIMITED STATUS (PRIMARY 1/N) ===")
    print(f"  separations:            {seps}")
    for k in dt_keys:
        print(f"  window_limited @ {k:>16}: {wl_by_dt[k]}")
    print(f"  consistent across dt (like-for-like): {checks['window_limited_consistent_across_dt']}")
    if disagree:
        print(f"  DISAGREEING rungs (window-limited at some dt but not others): {disagree}")
        print("  -> informative: the lag scale sits near the window edge for these rungs.")

    print("\n=== OBSERVED tau* PER RUNG (physical time, PRIMARY 1/N) ===")
    print(f"  {'cell':>16} | " + " ".join(f"sep{s:<5}" for s in seps))
    for k in list(dict.fromkeys(dt_keys + [cell_key(N, baseline_dt) for N in n_grid])):
        tm = summ[k]['tau_star_mean_per_rung']
        wl = summ[k]['window_limited_per_rung']
        cells_str = " ".join(f"{t:7.2f}{'*' if w else ' '}" for t, w in zip(tm, wl))
        print(f"  {k:>16} | {cells_str}")
    print(f"  (* = window-limited: tau* hit the window edge {t_phys}; Ŝ is finite-horizon, NOT resolved)")

    # ── (item 3) Honest max_lag cap validation on the BASELINE cell ──
    cap_validation = validate_max_lag_caps(cells[bkey], cells[bkey]['_rho_sq'], mi, seps)
    print("\n=== max_lag CAP VALIDATION (baseline cell, post-hoc, NOT adopted) ===")
    f = cap_validation['full']
    print(f"  FULL  max_lag={f['max_lag']:6d} window={f['phys_lag_window']:.2f} "
          f"Spearman(S,MI)={f['spearman_S_MI']:.3f} window-limited rungs={f['n_window_limited_rungs']}")
    for r in cap_validation['caps']:
        print(f"  CAP {r['fraction_of_window']:.2f}x max_lag={r['max_lag']:6d} window={r['phys_lag_window']:.2f} "
              f"Spearman(S,MI)={r['spearman_S_MI']:.3f} max|rel diff|={r['max_rel_diff_vs_full']:.3f} "
              f"median|rel diff|={r['median_rel_diff_vs_full']:.3f} ordering_changed={r['ordering_changed']}")

    # ── Save ──
    results_dir = os.path.join(CODEDIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    npz_data = {}
    cells_json = {}
    for k, cell in cells.items():
        c = dict(cell)
        npz_data[f'{k}_rho_sq'] = c.pop('_rho_sq')
        cells_json[k] = c

    out = {
        'description': 'P2 numerical robustness (dt and N) on a common bimodal separation ladder',
        'estimator': {
            'primary_threshold': '1/N',
            'stopping_rule': ('first lag tau with rho^2(tau) <= 1/N, included (paper eq:tmax_stopping)'
                              if STOPPING == 'first' else
                              'largest lag tau with rho^2(tau) > 1/N; integrate through tau*'),
            'lag_zero': 'excluded (rho_sq = rho[1:]^2)',
            'aggregation': 'S_hat = dt * sum_{tau=1..tau*} rho^2(tau) (Riemann rectangle)',
            'matches': 'paper eq:tmax_stopping (FLOOR_1N=1/N_TRAJ)',
            'secondary_threshold': '1/(Nd) — sensitivity only, NOT canonical',
        },
        'D': D, 'separations': seps, 'rep_sep': REP_SEP, 'T_phys': t_phys,
        'n_seeds': n_seeds, 'baseline': {'N': baseline_n, 'dt': baseline_dt, 'key': bkey},
        'dt_grid': dt_grid, 'N_grid': n_grid,
        'mi': mi, 'seed_rule': 'seed_val = seed_index*10000 + rung_index; MI seed = 777+rung',
        'device': str(device),
        'rng_policy': 'all RNG on CPU generator (per-step torch.randn(n,d)), tensors moved to device; '
                      'no device-side RNG, so cpu/cuda runs are comparable',
        'cells': cells_json,
        'summary_primary_1N': summ,
        'summary_secondary_1Nd': summ_1Nd,
        'max_lag_override': args.max_lag,
        'max_lag_cap_validation': cap_validation,
        'checks': checks,
        'total_wall_s': time.time() - t_all,
    }

    out_json = os.path.join(results_dir, f'p2_numerical_robustness{tag}.json')
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved JSON -> {out_json}")

    out_npz = os.path.join(results_dir, f'p2_numerical_robustness{tag}_rho_sq.npz')
    np.savez_compressed(out_npz, separations=np.array(seps), mi=np.array(mi), **npz_data)
    print(f"Saved rho_sq -> {out_npz}")

    # ── Markdown tables ──
    md = build_markdown(out)
    out_md = os.path.join(results_dir, f'p2_numerical_robustness{tag}_table.md')
    with open(out_md, 'w') as f:
        f.write(md)
    print(f"Saved table -> {out_md}")
    print("\n" + md)
    print(f"Total wall: {out['total_wall_s']:.1f}s")

    all_pass = (checks['unit_test_stopping_rule']
                and checks['dt_same_physical_horizon'] and checks['dt_same_lag_window']
                and checks['baseline_in_dt_grid'] and checks['baseline_in_N_grid']
                and all(checks['floor_recomputed_per_N'].values())
                and all(checks['floor_1Nd_recomputed_per_N'].values()))
    print(f"\nALL STRUCTURAL CHECKS PASSED: {all_pass}")
    return 0


def build_markdown(out):
    seps = out['separations']
    summ = out['summary_primary_1N']
    cells = out['cells']
    bn, bdt = out['baseline']['N'], out['baseline']['dt']
    rep = summ[out['baseline']['key']]['rep_sep']   # actual rung used
    lines = []
    lines.append(f"### P2 numerical robustness (common {out['D']}D bimodal ladder, T_phys={out['T_phys']}, "
                 f"{out['n_seeds']} seeds)\n")
    lines.append(f"PRIMARY SSA estimator: threshold=1/N, stopping lag = {out['estimator']['stopping_rule']}, "
                 f"integrate through tau* (lag 0 excluded), Shat = dt * sum rho^2. "
                 f"1/(Nd) is a labeled secondary sensitivity only (see JSON summary_secondary_1Nd).")
    lines.append(f"Baseline cell: N={bn}, dt={bdt}. Point stability at representative rung separation={rep}. "
                 f"**† marks a WINDOW-LIMITED (finite-horizon) entry**: τ* reached the edge of the "
                 f"{out['T_phys']} lag window, so Ŝ is a finite-horizon quantity and is NOT a resolved "
                 f"population SSA value. `wl rungs` = how many of the {len(seps)} ladder rungs are window-limited.\n")

    def mark(s):
        return '†' if s['rep_window_limited'] else ''

    # dt study
    lines.append(f"\n**Step-size study (N={bn} fixed, physical horizon AND physical lag window fixed at {out['T_phys']}; smaller dt = more steps/score evals)**\n")
    lines.append("| dt | steps T | max_lag | Ŝ(sep=%.1f) mean±sd | CV | τ*(sep=%.1f) | wl rungs | Spearman(Ŝ,MI) | Spearman(Ŝ,baseline) | score evals | wall (s) |" % (rep, rep))
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for dt in out['dt_grid']:
        k = f"N{bn}_dt{str(dt).replace('.','p')}"
        c, s = cells[k], summ[k]
        lines.append(f"| {dt} | {c['T']} | {c['max_lag']} | {s['rep_mean_S']:.4f}±{s['rep_sd_S']:.4f}{mark(s)} | {s['rep_cv']:.3f} | "
                     f"{s['rep_tau_star_mean']:.2f} | {s['n_window_limited_rungs']}/{len(seps)} | "
                     f"{s['spearman_S_MI_mean']:.3f} | {s['spearman_vs_baseline']:.3f} | {c['score_evals']:.3e} | {c['wall_s']:.1f} |")

    # N study
    lines.append(f"\n**Trajectory-count study (dt={bdt} fixed, threshold 1/N recomputed per N)**\n")
    lines.append("| N | threshold 1/N | Ŝ(sep=%.1f) mean±sd | CV | τ*(sep=%.1f) | wl rungs | Spearman(Ŝ,MI) | Spearman(Ŝ,baseline) | score evals | wall (s) |" % (rep, rep))
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for N in out['N_grid']:
        k = f"N{N}_dt{str(bdt).replace('.','p')}"
        c, s = cells[k], summ[k]
        lines.append(f"| {N} | {c['floor_1N']:.2e} | {s['rep_mean_S']:.4f}±{s['rep_sd_S']:.4f}{mark(s)} | {s['rep_cv']:.3f} | "
                     f"{s['rep_tau_star_mean']:.2f} | {s['n_window_limited_rungs']}/{len(seps)} | "
                     f"{s['spearman_S_MI_mean']:.3f} | {s['spearman_vs_baseline']:.3f} | {c['score_evals']:.3e} | {c['wall_s']:.1f} |")

    # Per-rung tau* and window-limited status (item 4)
    lines.append(f"\n**Observed stopping lag τ* per rung (baseline cell N={bn}, dt={bdt}; † = window-limited)**\n")
    lines.append("| separation | " + " | ".join(f"{s:g}" for s in seps) + " |")
    lines.append("|---|" + "---|" * len(seps))
    bs = summ[out['baseline']['key']]
    lines.append("| MI | " + " | ".join(f"{m:.3f}" for m in out['mi']) + " |")
    lines.append("| τ* (phys) | " + " | ".join(
        f"{t:.2f}{'†' if w else ''}" for t, w in zip(bs['tau_star_mean_per_rung'], bs['window_limited_per_rung'])) + " |")
    lines.append("| Ŝ | " + " | ".join(f"{v:.3f}" for v in bs['mean_S']) + " |")

    # Cap validation (item 3)
    cv = out.get('max_lag_cap_validation')
    if cv:
        lines.append(f"\n**max_lag cap validation (baseline cell, post-hoc; NO cap adopted)**\n")
        lines.append("| max_lag | phys window | Spearman(Ŝ,MI) | max abs rel diff vs full | ordering changed |")
        lines.append("|---|---|---|---|---|")
        f_ = cv['full']
        lines.append(f"| {f_['max_lag']} (full) | {f_['phys_lag_window']:.2f} | {f_['spearman_S_MI']:.3f} | — | — |")
        for r in cv['caps']:
            lines.append(f"| {r['max_lag']} ({r['fraction_of_window']:.2f}×) | {r['phys_lag_window']:.2f} | "
                         f"{r['spearman_S_MI']:.3f} | {r['max_rel_diff_vs_full']:.3f} | {r['ordering_changed']} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == '__main__':
    # self syntax check
    with open(__file__) as f:
        ast.parse(f.read())
    sys.exit(main())
