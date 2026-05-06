"""
Lag-bootstrap on Ŝ_fixed20 (Riemann sum) across all 14 prompts × 4 guidance scales.

For each cell, we already have ρ²(τ) at the 100 fixed-20 lags (saved in JSON).
Bootstrap = resample these 100 lag indices with replacement, recompute
Ŝ = Δt × Σ ρ², repeat B=2000 times. Report mean / SE / 95% CI.

This is a *methodological* CI (sensitivity to lag selection), not a sampling
CI over trajectories — the latter would need per-trajectory CLIP saved at
all lags, which we don't have except for elderly gs=7.5.

Output:
  results_shat14_bootstrap.csv  — one row per (prompt, gs)
"""
import csv
import glob
import json
import os
from pathlib import Path
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(SCRIPT_DIR, "results")

PROMPTS = [
    ("cat",       os.path.join(RESULTS, "clip_da_mp_p0_cat_gs{}_dt0p005.json")),
    ("mountains", os.path.join(RESULTS, "clip_da_mp_p1_mountains_gs{}_dt0p005.json")),
    ("flowers",   os.path.join(RESULTS, "clip_da_mp_p2_flowers_gs{}_dt0p005.json")),
    ("street",    os.path.join(RESULTS, "clip_da_mp_p3_street_gs{}_dt0p005.json")),
    ("food",      os.path.join(RESULTS, "clip_da_mp_p4_food_gs{}_dt0p005.json")),
    ("elderly",   os.path.join(RESULTS, "clip_da_mp_p5_elderly_gs{}_dt0p005.json")),
    ("car",       os.path.join(RESULTS, "clip_da_mp_p6_car_gs{}_dt0p005.json")),
    ("person",    os.path.join(RESULTS, "clip_da_sweep_gs{}_dt0p005.json")),
    ("bat",       os.path.join(RESULTS, "clip_da_ext_p0_bat_gs{}_dt0p005.json")),
    ("crane",     os.path.join(RESULTS, "clip_da_ext_p1_crane_gs{}_dt0p005.json")),
    ("mouse",     os.path.join(RESULTS, "clip_da_ext_p2_mouse_gs{}_dt0p005.json")),
    ("seal",      os.path.join(RESULTS, "clip_da_ext_p3_seal_gs{}_dt0p005.json")),
    ("jaguar",    os.path.join(RESULTS, "clip_da_ext_p4_jaguar_gs{}_dt0p005.json")),
    ("palm",      os.path.join(RESULTS, "clip_da_ext_p5_palm_gs{}_dt0p005.json")),
]
GS_KEYS = ["1p0", "3p0", "7p5", "15p0"]
GS_VALUES = [1.0, 3.0, 7.5, 15.0]
DT = 0.005
DELTA_FIXED20 = 20 * DT      # 0.1
T = 2000
N = 200
NF = 1.0 / N                 # 1/N noise floor

B = 2000
RNG = np.random.default_rng(20260504)

OUT_CSV = Path(__file__).resolve().parent / "results_shat14_bootstrap.csv"


def load_fixed20_rhosq(path):
    d = json.load(open(path))
    rho_sq = np.asarray(d["rho_sq_all"], dtype=np.float64)
    lags = np.asarray(d["snapshot_lags_all"], dtype=np.int64)
    fixed20_set = set(range(20, T + 1, 20))
    mask = np.fromiter((int(l) in fixed20_set for l in lags), bool, len(lags))
    return rho_sq[mask], lags[mask], d.get("hit_floor_fixed20"), d.get("ssa_fixed20")


def shat_truncate_at_floor(rho):
    """Match the canonical Ŝ_fixed20 estimator: sum until first lag where ρ² < 1/N."""
    below = rho < NF
    if below.any():
        cut = int(np.argmax(below))
        return DELTA_FIXED20 * float(rho[:cut].sum())
    return DELTA_FIXED20 * float(rho.sum())


def shat_no_trunc(rho):
    """Plain Riemann sum (no truncation) — used inside bootstrap so the
    sum scale is preserved when the lag ordering is randomised."""
    return DELTA_FIXED20 * float(rho.sum())


rows = []
print(f"{'prompt':<10} {'gs':>4}  {'point':>8} {'mean':>8} {'se':>7} {'ci_lo':>8} {'ci_hi':>8}  {'hit_floor'}")
print("-" * 80)

for label, tmpl in PROMPTS:
    for gk, gv in zip(GS_KEYS, GS_VALUES):
        rho, lags, hit_floor, point_saved = load_fixed20_rhosq(tmpl.format(gk))
        L = len(rho)
        # Bootstrap: resample lag indices with replacement, recompute Ŝ.
        # Truncation-at-noise-floor on a permuted index sequence isn't
        # well-defined (the curve is no longer monotonic), so we use the
        # plain Riemann sum and rescale to match the saved truncated point
        # estimate (preserves the variance shape, anchors the scale).
        idx = RNG.integers(0, L, size=(B, L))
        rho_resampled = rho[idx]                       # (B, L)
        shat_b = DELTA_FIXED20 * rho_resampled.sum(axis=1)
        # Reference: Ŝ from full set without truncation
        shat_full = DELTA_FIXED20 * float(rho.sum())
        # Anchor mean(boot) ≈ shat_full by construction; report scaled to
        # the canonical (truncation-respecting) point estimate so the CI
        # appears around the value the table reports.
        scale = (point_saved / shat_full) if (point_saved and shat_full) else 1.0
        shat_b *= scale
        mean = float(shat_b.mean())
        se = float(shat_b.std(ddof=1))
        lo, hi = np.percentile(shat_b, [2.5, 97.5])
        rows.append({
            "prompt": label, "w": gv,
            "shat_point": point_saved,
            "shat_boot_mean": mean,
            "shat_boot_se": se,
            "shat_boot_ci_lo": float(lo),
            "shat_boot_ci_hi": float(hi),
            "hit_floor_fixed20": bool(hit_floor) if hit_floor is not None else None,
            "B": B, "n_fixed20_lags": L,
        })
        print(f"{label:<10} {gv:>4}  {point_saved:>8.3f} {mean:>8.3f} {se:>7.3f} "
              f"{lo:>8.3f} {hi:>8.3f}  {hit_floor}")

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\n[saved] {OUT_CSV}")
