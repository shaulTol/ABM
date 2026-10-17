#!/usr/bin/env python3
"""
Resolve the MC-margin discrepancy for the SDXL elderly DA1 cell at L* = 2000.

Two published margins exist for the SAME observed lambda_1 = 9.9619e-3:

  Null A ("8.1x")  project notes / mc_dip_validation:
      pipeline-matched ANISOTROPIC null. 17 GPU replicates each sampled
      N(mu, Sigma_hat_latent) initial latents (SVD of the observed x0), ran the
      REAL SDXL Langevin (T=2000, dt=0.0025), VAE-decoded and CLIP-embedded
      z0 and z_tau. The 850 resulting pairs are pooled; a bootstrap of
      B = 10,000 draws of n = 200 forms C_b = z_tau^T z0 / 200, symmetrises and
      keeps the top-10 eigenvalues BY MAGNITUDE. "null max" = max over the
      10,000 bootstrap draws of the rank-1 magnitude.
      Source: experiments/sdxl/results/sdxl_null_anisotropic_combined.npz,
              .../sdxl_anisotropic_null_pvalues.json,
              older_versions/experiments/mc_dip_validation/
                  sdxl_iso_null_via_late_t/compute_bootstrap_pvalue.py

  Null B ("2.1x")  fig_elderly_da.py spectrum-figure recipe, used in
      compute.py's per-lag column: an ANALYTIC GAUSSIAN OU SURROGATE in CLIP
      space, x_tau = exp(-tau/2) z0 + sqrt(1-exp(-tau)) zeta with
      z0, zeta ~ N(0, C_hat(0)); 200 independent replicates, seed 42.
      "null max"/"null p95" = over the 200 replicates.

This script determines whether the gap is caused by the NULL CONSTRUCTION or by
the number of replicates, by:

  A. reproducing Null A exactly (same pool, same B, same seed);
  B. re-running Null A's IDENTICAL bootstrap procedure on a Null-B-generated
     pool of the same size (850 pairs) -- construction varies, procedure fixed;
  C. a replicate-count control on Null B: running max of |lambda_1| over
     R = 200, 1000, 5000 independent replicates.

Output: mc_reconciliation.json (read and rendered by compute.py).
"""

import os
import json
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "sdxl", "results"))

NULL_A_POOL = os.path.join(RESULTS_DIR, "sdxl_null_anisotropic_combined.npz")
NULL_A_JSON = os.path.join(RESULTS_DIR, "sdxl_anisotropic_null_pvalues.json")
DA_NPZ = os.path.join(RESULTS_DIR, "clip_da_elderly_gs7p5_fc_dt0p0025.npz")
CTAU_FULL = os.path.join(RESULTS_DIR, "elderly_gs7p5_n200_C_tau_full.npz")

TAU_STAR = 5.0           # L* = 2000 steps x dt = 0.0025
N_RESAMPLE = 200
B_BOOT = 10000
BOOT_SEED = 20260502     # exactly the seed used by compute_bootstrap_pvalue.py
POOL_SEED = 777          # seed for the synthetic Null-B pool
REP_SEED = 42            # seed used by fig_elderly_da.py / compute.py
N_TOP = 10


def top_abs_eigs(C, k=N_TOP):
    Csym = 0.5 * (C + C.T)
    ev = np.linalg.eigvalsh(Csym)
    order = np.argsort(-np.abs(ev))
    return ev[order[:k]]


def bootstrap_pool(z0, z_tau, b_boot=B_BOOT, n=N_RESAMPLE, seed=BOOT_SEED, tag=""):
    """Null A's bootstrap procedure, applied to any (z0, z_tau) pool."""
    mu0 = z0.mean(axis=0)
    mut = z_tau.mean(axis=0)
    z0c = z0 - mu0
    ztc = z_tau - mut
    rng = np.random.default_rng(seed)
    out = np.zeros((b_boot, N_TOP))
    t0 = time.time()
    for b in range(b_boot):
        idx = rng.integers(0, z0.shape[0], size=n)
        C = ztc[idx].T @ z0c[idx] / n
        out[b] = top_abs_eigs(C)
        if (b + 1) % 2000 == 0:
            print(f"    {tag} {b+1}/{b_boot} ({time.time()-t0:.0f}s)")
    return out


def main():
    print("Loading observed and null-pool data...")
    da = np.load(DA_NPZ, allow_pickle=True)
    lam1_obs = float(np.asarray(da["eigenvalues_full_at_tau_star"])[0])
    print(f"  observed lambda_1 at L*=2000: {lam1_obs:.6e}")

    stored = json.load(open(NULL_A_JSON))
    stored_max = stored["pvalues"]["lambda_1"]["null_max_abs"]
    stored_ratio = lam1_obs / stored_max
    print(f"  stored Null A null_max_abs = {stored_max:.6e} -> ratio "
          f"{stored_ratio:.2f}x")

    # ---- A: reproduce Null A exactly ---------------------------------------
    pool = np.load(NULL_A_POOL)
    z0A = pool["z0"].astype(np.float64)
    ztA = pool["z_tau"].astype(np.float64)
    print(f"\n[A] Reproducing Null A bootstrap (pool {z0A.shape}, B={B_BOOT})...")
    bootA = bootstrap_pool(z0A, ztA, tag="A")
    A_max = float(np.abs(bootA[:, 0]).max())
    A_mean_abs = float(np.abs(bootA[:, 0]).mean())
    A_mean_signed = float(bootA[:, 0].mean())
    A_p95 = float(np.percentile(np.abs(bootA[:, 0]), 95))
    A_ratio = lam1_obs / A_max
    A_reproduced = bool(abs(A_max - stored_max) / stored_max < 1e-6)
    print(f"  null_max_abs = {A_max:.6e} (stored {stored_max:.6e}, "
          f"reproduced={A_reproduced})")
    print(f"  ratio obs/null_max = {A_ratio:.2f}x")

    # ---- B: same procedure, Null-B construction ----------------------------
    if os.path.exists(CTAU_FULL):
        cd = np.load(CTAU_FULL, mmap_mode="r")
        C0 = np.asarray(cd["C0"]).astype(np.float64)
        c0_source = os.path.basename(CTAU_FULL)
    else:
        raise SystemExit("C(0) source missing")
    ev0, evec0 = np.linalg.eigh(C0)
    ev0 = np.maximum(ev0, 0.0)
    sqrtC0 = evec0 * np.sqrt(ev0)[np.newaxis, :]
    lam1_C0 = float(ev0.max())

    rho = np.exp(-TAU_STAR / 2.0)
    noise = np.sqrt(max(1.0 - np.exp(-TAU_STAR), 1e-20))
    N_big = z0A.shape[0]

    rng = np.random.default_rng(POOL_SEED)
    d = C0.shape[0]
    z0B = rng.standard_normal((N_big, d)) @ sqrtC0.T
    zB_noise = rng.standard_normal((N_big, d)) @ sqrtC0.T
    ztB = rho * z0B + noise * zB_noise
    print(f"\n[B] Null-B pool ({N_big} OU pairs, rho={rho:.4f}) through Null A's "
          f"bootstrap (B={B_BOOT})...")
    bootB = bootstrap_pool(z0B, ztB, tag="B")
    B_max = float(np.abs(bootB[:, 0]).max())
    B_mean_abs = float(np.abs(bootB[:, 0]).mean())
    B_mean_signed = float(bootB[:, 0].mean())
    B_p95 = float(np.percentile(np.abs(bootB[:, 0]), 95))
    B_ratio = lam1_obs / B_max
    print(f"  null_max_abs = {B_max:.6e} -> ratio {B_ratio:.2f}x")

    # ---- C: replicate-count control on Null B ------------------------------
    print("\n[C] Replicate-count control on Null B (independent replicates)...")
    R_MAX = 5000
    checkpoints = [200, 1000, 5000]
    rngc = np.random.RandomState(REP_SEED)
    running_max = 0.0
    running = []
    tops = np.empty(R_MAX)
    t0 = time.time()
    for i in range(R_MAX):
        z0 = rngc.randn(N_RESAMPLE, d) @ sqrtC0.T
        zn = rngc.randn(N_RESAMPLE, d) @ sqrtC0.T
        xt = rho * z0 + noise * zn
        mu = z0.mean(axis=0)
        C = (xt - mu).T @ (z0 - mu) / N_RESAMPLE
        e = top_abs_eigs(C, k=1)[0]
        tops[i] = e
        running_max = max(running_max, abs(float(e)))
        if (i + 1) in checkpoints:
            running.append({
                "n_replicates": i + 1,
                "null_max_abs": running_max,
                "null_p95_abs": float(np.percentile(np.abs(tops[:i + 1]), 95)),
                "ratio_obs_over_null_max": lam1_obs / running_max,
            })
            print(f"    R={i+1:5d}  max={running_max:.4e}  "
                  f"ratio={lam1_obs/running_max:.2f}x  ({time.time()-t0:.0f}s)")

    monotone_up = all(running[i]["null_max_abs"] <= running[i + 1]["null_max_abs"]
                      for i in range(len(running) - 1))
    ratio_shrinks = all(running[i]["ratio_obs_over_null_max"] >=
                        running[i + 1]["ratio_obs_over_null_max"]
                        for i in range(len(running) - 1))

    out = {
        "question": "Why does the same observed lambda_1 = 9.96e-3 give 8.1x in "
                    "the project notes and ~2.1x in the rebuttal lag table?",
        "observed_lambda1_at_Lstar": lam1_obs,
        "tau_star": TAU_STAR,
        "null_A_pipeline_anisotropic": {
            "description": "17 real-SDXL Langevin replicates from "
                           "N(mu, Sigma_hat_latent) initial latents, VAE+CLIP "
                           "embedded; 850 pooled (z0, z_tau) pairs; bootstrap "
                           "B=10000 draws of n=200; top-10 eigenvalues by "
                           "magnitude; 'null max' = max over the 10000 draws of "
                           "the rank-1 magnitude (per rank, not across ranks).",
            "pool_file": os.path.relpath(NULL_A_POOL, RESULTS_DIR),
            "N_big": int(N_big),
            "n_resample": N_RESAMPLE,
            "B_boot": B_BOOT,
            "seed": BOOT_SEED,
            "stored_null_max_abs": stored_max,
            "recomputed_null_max_abs": A_max,
            "exactly_reproduced": A_reproduced,
            "null_mean_signed": A_mean_signed,
            "null_mean_abs": A_mean_abs,
            "null_p95_abs": A_p95,
            "ratio_obs_over_null_max": A_ratio,
            "ratio_obs_over_null_p95": lam1_obs / A_p95,
        },
        "null_B_analytic_OU_surrogate": {
            "description": "fig_elderly_da.py recipe: x_tau = exp(-tau/2) z0 + "
                           "sqrt(1-exp(-tau)) zeta with z0, zeta ~ N(0, C_hat(0)) "
                           "in CLIP space; no SDXL dynamics. Independent "
                           "replicates, not a bootstrap.",
            "C0_source": c0_source,
            "lambda1_of_C0": lam1_C0,
            "rho_built_into_null": float(rho),
            "expected_systematic_top_eigenvalue_rho_times_lambda1_C0":
                float(rho * lam1_C0),
            "as_used_in_lag_table_200_reps_seed42": {
                "null_p95": 4.1089e-3,
                "null_max": 4.6495e-3,
                "ratio_obs_over_null_p95": lam1_obs / 4.1089e-3,
                "ratio_obs_over_null_max": lam1_obs / 4.6495e-3,
            },
            "under_null_A_bootstrap_procedure": {
                "N_big": int(N_big),
                "n_resample": N_RESAMPLE,
                "B_boot": B_BOOT,
                "seed": BOOT_SEED,
                "pool_seed": POOL_SEED,
                "null_max_abs": B_max,
                "null_mean_signed": B_mean_signed,
                "null_mean_abs": B_mean_abs,
                "null_p95_abs": B_p95,
                "ratio_obs_over_null_max": B_ratio,
            },
        },
        "replicate_count_control": {
            "description": "Null B, independent replicates, running max of "
                           "|lambda_1|. If replicate count explained the gap, "
                           "more replicates would shrink the null max; it cannot, "
                           "because a running maximum is non-decreasing.",
            "seed": REP_SEED,
            "checkpoints": running,
            "null_max_is_monotone_nondecreasing_in_R": monotone_up,
            "ratio_monotone_nonincreasing_in_R": ratio_shrinks,
        },
        "verdict": {
            "cause": "null construction, not replicate count",
            "both_internally_correct": True,
            "explanation":
                "Null A's pooled null has essentially decorrelated by tau* "
                f"(null_mean_signed = {A_mean_signed:.3e}), so its lambda_1 is "
                "pure finite-sample fluctuation. Null B BUILDS IN a lag-tau "
                f"coupling of rho = exp(-tau*/2) = {rho:.4f} in every direction, "
                "including the top direction of C_hat(0); its systematic "
                f"contribution alone is rho * lambda_1(C_hat(0)) = "
                f"{rho * lam1_C0:.3e}, already about {rho * lam1_C0 / stored_max:.1f}x "
                "Null A's entire null maximum, before any fluctuation is added. "
                "Holding the bootstrap procedure fixed and swapping only the "
                f"construction moves the null max from {A_max:.3e} (A) to "
                f"{B_max:.3e} (B), a factor of {B_max / A_max:.1f}, which accounts "
                "for the discrepancy.",
            "ratio_null_B_over_null_A_same_procedure": B_max / A_max,
            "apples_to_apples_margin_against_null_A": A_ratio,
            "apples_to_apples_margin_against_null_B": B_ratio,
        },
    }

    path = os.path.join(SCRIPT_DIR, "mc_reconciliation.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
