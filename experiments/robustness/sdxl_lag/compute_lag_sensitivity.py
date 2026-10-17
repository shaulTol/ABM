#!/usr/bin/env python3
"""
P8: SDXL DA lag-sensitivity table (rebuttal, read-only analysis).

Question: is the elderly-prompt DA1 direction (w=7.5, dt=0.0025) stable across
the stored lag grid, or is it specific to L* = 2000?

This script is READ-ONLY on stored results. No simulation, no GPU, no new SDXL
sampling. The only model loaded is the REAL CLIP text encoder
(openai/clip-vit-large-patch14) on CPU, used solely to rebuild the two text
probes exactly as `experiments/sdxl/fig_elderly_da.py` does:

    positive = "a photograph of a man"
    negative = "a photograph of a woman"
    axis     = normalize(normalize(emb_pos) - normalize(emb_neg))

Common evaluation set (critical): centered `z0_reembed` from
`clip_da_elderly_gs7p5_fc_z_tau.npz`, i.e. the re-embedded INITIAL samples, the
same set Appendix H.7 / `fig_elderly_da.py` use for the reported |rho| = 0.90.
Every DA direction v1(L) is scored on those same samples:

    score_L = (z0_reembed - mean(z0_reembed)) @ v1(L)

`z_tau` is used ONLY for a separately labelled stationarity sensitivity.

Outputs:
  RESULTS.md    compact rebuttal table + supplementary tables
  results.json  machine-readable

Usage:
  python experiments/robustness/sdxl_lag/compute_lag_sensitivity.py
"""

import os
import sys
import json
import argparse

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
SDXL_DIR = os.path.join(PROJECT_ROOT, "experiments", "sdxl")
RESULTS_DIR = os.path.join(SDXL_DIR, "results")
sys.path.insert(0, PROJECT_ROOT)

DA_NPZ = os.path.join(RESULTS_DIR, "clip_da_elderly_gs7p5_fc_dt0p0025.npz")
DA_JSON = os.path.join(RESULTS_DIR, "clip_da_elderly_gs7p5_fc_dt0p0025.json")
Z_NPZ = os.path.join(RESULTS_DIR, "clip_da_elderly_gs7p5_fc_z_tau.npz")
TRAJ_JSON = os.path.join(RESULTS_DIR, "traj_avg_elderly_gs7p5_dt0p0025.json")
CTAU_FULL_NPZ = os.path.join(RESULTS_DIR, "elderly_gs7p5_n200_C_tau_full.npz")

DT = 0.0025
L_STAR = 2000

# 8 approximately log-spaced lags from the stored grid; L* = 2000 included.
LAGS = [1, 20, 60, 160, 400, 800, 1400, 2000]

CLIP_MODEL = "openai/clip-vit-large-patch14"
GENDER_POS = "a photograph of a man"
GENDER_NEG = "a photograph of a woman"

MC_REPS = 200
MC_SEED = 42

ANGLE_TOL_DEG = 1e-3       # check 1 tolerance
RHO_TARGET = 0.90          # check 2 target
RHO_TOL = 0.03
LAM1_TARGET = 9.96e-3      # check 4 target
LAM1_RTOL = 0.01


# ----------------------------------------------------------------------------
# gender axis (REAL CLIP text encoder, exactly as fig_elderly_da.py)
# ----------------------------------------------------------------------------

def build_gender_axis():
    """Rebuild the fixed gender probe axis with the real CLIP text encoder."""
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    model = CLIPModel.from_pretrained(CLIP_MODEL)
    model.eval()
    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)

    def _encode(text):
        with torch.no_grad():
            out = model.get_text_features(
                **tok(text, return_tensors="pt", padding=True, truncation=True))
            # transformers <= 4.x returns a tensor; 5.x returns a
            # BaseModelOutputWithPooling whose pooler_output is the same
            # projected text embedding.
            emb = out if torch.is_tensor(out) else out.pooler_output
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.squeeze(0).cpu().numpy().astype(np.float64)

    axis = _encode(GENDER_POS) - _encode(GENDER_NEG)
    axis = axis / np.linalg.norm(axis)
    return axis


# ----------------------------------------------------------------------------
# matched Monte Carlo null (same recipe as fig_elderly_da.py make_spectrum_figure)
# ----------------------------------------------------------------------------

def mc_null_lambda1(C0, N, tau_phys, n_reps=MC_REPS, seed=MC_SEED):
    """Matched OU null: top eigenvalue distribution of the symmetrised C(tau)
    estimator when the data are Gaussian with covariance C0 and pure OU decay
    (no genuine slow direction). Returns (95th pctl, max) of lambda_1."""
    rng = np.random.RandomState(seed)
    d = C0.shape[0]
    ev, evec = np.linalg.eigh(C0)
    ev = np.maximum(ev, 0.0)
    sqrt_C0 = evec * np.sqrt(ev)[np.newaxis, :]

    rho = np.exp(-tau_phys / 2.0)
    noise_scale = np.sqrt(max(1.0 - np.exp(-tau_phys), 1e-20))

    top = np.empty(n_reps)
    for i in range(n_reps):
        z0 = rng.randn(N, d) @ sqrt_C0.T
        zn = rng.randn(N, d) @ sqrt_C0.T
        x_tau = rho * z0 + noise_scale * zn
        mu = z0.mean(axis=0)
        z0c = z0 - mu
        xc = x_tau - mu
        C_raw = (xc.T @ z0c) / N
        C_sym = 0.5 * (C_raw + C_raw.T)
        top[i] = np.linalg.eigvalsh(C_sym)[-1]
    return float(np.percentile(top, 95)), float(top.max())


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mc", action="store_true",
                    help="skip the matched Monte Carlo null (slow-ish)")
    ap.add_argument("--mc-all-lags", action="store_true", default=True,
                    help="also compute the matched MC null at every reported lag "
                         "(supplementary table)")
    args = ap.parse_args()

    print("Loading stored results (read-only)...")
    da = np.load(DA_NPZ, allow_pickle=True)
    zz = np.load(Z_NPZ)
    da_json = json.load(open(DA_JSON))
    traj = json.load(open(TRAJ_JSON))

    snapshot_lags = np.asarray(da["snapshot_lags"]).astype(int)
    lam_plus_profile = np.asarray(da["freeconv_lambda_plus_profile"]).astype(float)
    lam2_profile = np.asarray(da["freeconv_lambda2_profile"]).astype(float)

    z0 = zz["z0_reembed"].astype(np.float64)      # (N, D) re-embedded initial samples
    z_tau = zz["z_tau"].astype(np.float64)
    N, D = z0.shape
    z0c = z0 - z0.mean(axis=0)                    # centered ONCE, common eval set

    checks = {}

    # ---- check 5: lag arrays / free-conv profile compatibility -------------
    eig_lags = sorted(int(k.split("lag")[1]) for k in da.files
                      if k.startswith("eigvecs_lag"))
    val_lags = sorted(int(k.split("lag")[1]) for k in da.files
                      if k.startswith("eigenvalues_lag"))
    c5_len = (len(lam_plus_profile) == len(snapshot_lags) ==
              len(lam2_profile) == len(eig_lags) == len(val_lags))
    c5_order = (eig_lags == list(snapshot_lags)) and (val_lags == list(snapshot_lags))
    c5_mono = bool(np.all(np.diff(snapshot_lags) > 0))
    # independent consistency probe: stored lambda2 profile equals |eig|-sorted
    # second eigenvalue at each lag (that is how it was written)
    lam2_recomp = []
    for L in snapshot_lags:
        e = np.asarray(da[f"eigenvalues_lag{L}"]).astype(float)
        lam2_recomp.append(np.sort(np.abs(e))[::-1][1])
    lam2_max_abs_err = float(np.max(np.abs(np.array(lam2_recomp) - lam2_profile)))
    c5 = bool(c5_len and c5_order and c5_mono and lam2_max_abs_err < 1e-8)
    checks["check5_lag_array_compatibility"] = {
        "passed": c5,
        "n_snapshot_lags": int(len(snapshot_lags)),
        "n_lambda_plus_profile": int(len(lam_plus_profile)),
        "n_eigvec_keys": len(eig_lags),
        "n_eigenvalue_keys": len(val_lags),
        "lags_strictly_increasing": c5_mono,
        "keys_match_snapshot_lags": bool(c5_order),
        "lambda2_profile_max_abs_err_vs_recompute": lam2_max_abs_err,
    }
    print(f"  check5 lag/profile compatibility: {'PASS' if c5 else 'FAIL'}")

    missing = [L for L in LAGS if L not in set(snapshot_lags)]
    if missing:
        raise SystemExit(f"requested lags not in stored grid: {missing}")

    # ---- reference direction at L* ----------------------------------------
    v_star = np.asarray(da[f"eigvecs_lag{L_STAR}"])[:, 0].astype(np.float64)
    v_star = v_star / np.linalg.norm(v_star)

    # ---- check 3: eigendecompose C_tau_star -------------------------------
    C_star = np.asarray(da["C_tau_star"]).astype(np.float64)
    C_star_sym = 0.5 * (C_star + C_star.T)
    ev_s, evec_s = np.linalg.eigh(C_star_sym)
    order = ev_s.argsort()[::-1].copy()
    v_from_C = evec_s[:, order[0]]
    abs_ip = float(abs(np.dot(v_from_C, v_star)))
    sign_inv_angle = float(np.degrees(np.arccos(min(1.0, abs_ip))))
    lam1_from_C = float(ev_s[order[0]])
    c3 = bool(abs_ip > 0.999)
    checks["check3_C_tau_star_eigvec"] = {
        "passed": c3,
        "abs_inner_product": abs_ip,
        "sign_invariant_angle_deg": sign_inv_angle,
        "lambda1_from_C_tau_star": lam1_from_C,
        "criterion": "abs inner product > 0.999 (sign-invariant)",
    }
    print(f"  check3 C_tau_star top eigvec: {'PASS' if c3 else 'FAIL'} "
          f"(|<v,v*>|={abs_ip:.6f}, angle={sign_inv_angle:.4f} deg)")

    # ---- check 4: lambda1(2000) -------------------------------------------
    lam1_star = float(np.asarray(da[f"eigenvalues_lag{L_STAR}"])[0])
    c4 = bool(abs(lam1_star - LAM1_TARGET) / LAM1_TARGET < LAM1_RTOL)
    checks["check4_lambda1_at_Lstar"] = {
        "passed": c4,
        "lambda1_2000": lam1_star,
        "target": LAM1_TARGET,
        "rel_error": float(abs(lam1_star - LAM1_TARGET) / LAM1_TARGET),
    }
    print(f"  check4 lambda1(2000)={lam1_star:.5e}: {'PASS' if c4 else 'FAIL'}")

    # ---- gender axis -------------------------------------------------------
    print(f"Building gender axis with real CLIP text encoder ({CLIP_MODEL}, CPU)...")
    axis = build_gender_axis()
    gender_score = z0c @ axis            # fixed across all lags

    # Axis provenance validation (not one of the five mandatory checks):
    # the pipeline stored cos(da1_eigvec, gender_axis) using the SAME two
    # strings but the opposite orientation (its DEFAULT_PROBE_PAIRS uses
    # woman - man; fig_elderly_da.py uses man - woman). Magnitudes must agree.
    da1_stored = np.asarray(da["da1_eigvec"]).astype(np.float64)
    da1_stored = da1_stored / np.linalg.norm(da1_stored)
    cos_recomp = float(abs(np.dot(da1_stored, axis)))
    cos_stored = float(abs(da_json["da1_cosines"]["gender"]))
    axis_ok = bool(abs(cos_recomp - cos_stored) < 1e-5)
    checks_axis = {
        "passed": axis_ok,
        "abs_cos_recomputed": cos_recomp,
        "abs_cos_stored_in_json": cos_stored,
        "note": "orientation differs by a sign (pipeline probe is woman - man, "
                "fig_elderly_da.py probe is man - woman); |cos| must match",
    }
    print(f"  axis provenance |cos| recomputed={cos_recomp:.7f} vs stored="
          f"{cos_stored:.7f}: {'PASS' if axis_ok else 'FAIL'}")

    # ---- trajectory-averaged rho^2 ----------------------------------------
    rho_sq_traj = {int(k): float(v) for k, v in traj["rho_sq_traj"].items()}
    rho_sq_simple = {int(k): float(v) for k, v in traj["rho_sq_simple"].items()}

    # ---- per-lag table -----------------------------------------------------
    rows = []
    for L in LAGS:
        i = int(np.where(snapshot_lags == L)[0][0])
        evals = np.asarray(da[f"eigenvalues_lag{L}"]).astype(float)  # value-sorted desc
        v = np.asarray(da[f"eigvecs_lag{L}"])[:, 0].astype(np.float64)
        v = v / np.linalg.norm(v)
        ip = float(np.dot(v, v_star))
        flipped = ip < 0
        if flipped:
            v = -v
            ip = -ip
        angle = float(np.degrees(np.arccos(min(1.0, max(-1.0, ip)))))

        score = z0c @ v
        rho, pval = spearmanr(score, gender_score)

        lam1, lam2 = float(evals[0]), float(evals[1])
        lam_plus = float(lam_plus_profile[i])
        rows.append({
            "L": int(L),
            "tau_phys": float(L * DT),
            "lambda1": lam1,
            "lambda2": lam2,
            "lambda2_over_lambda1": lam2 / lam1,
            "sign_flipped_vs_Lstar": bool(flipped),
            "angle_to_v1_Lstar_deg": angle,
            "spearman_rho_gender": float(rho),
            "abs_spearman_rho_gender": float(abs(rho)),
            "spearman_pvalue": float(pval),
            "rho_sq_traj_averaged": rho_sq_traj.get(L, None),
            "rho_sq_simple_pair": rho_sq_simple.get(L, None),
            "lambda_plus_analytic": lam_plus,
            "lambda1_over_lambda_plus": lam1 / lam_plus,
            "above_analytic_reference": bool(lam1 > lam_plus),
            "noise_dominated_analytic": bool(lam1 <= lam_plus),
        })

    # ---- check 1 and 2 at L* ----------------------------------------------
    row_star = [r for r in rows if r["L"] == L_STAR][0]
    c1 = bool(row_star["angle_to_v1_Lstar_deg"] < ANGLE_TOL_DEG)
    checks["check1_angle_zero_at_Lstar"] = {
        "passed": c1,
        "angle_deg": row_star["angle_to_v1_Lstar_deg"],
        "tolerance_deg": ANGLE_TOL_DEG,
    }
    c2 = bool(abs(row_star["abs_spearman_rho_gender"] - RHO_TARGET) <= RHO_TOL)
    checks["check2_gender_rho_at_Lstar"] = {
        "passed": c2,
        "abs_rho": row_star["abs_spearman_rho_gender"],
        "target": RHO_TARGET,
        "tolerance": RHO_TOL,
        "reported_paper_rho_at_peak_gap_lag_1400":
            da_json["da1_correlations"]["gender"]["rho"],
    }
    print(f"  check1 angle at L*={row_star['angle_to_v1_Lstar_deg']:.3e} deg: "
          f"{'PASS' if c1 else 'FAIL'}")
    print(f"  check2 |rho|(L*)={row_star['abs_spearman_rho_gender']:.4f}: "
          f"{'PASS' if c2 else 'FAIL'}")

    all_pass = all(v["passed"] for v in checks.values())

    # ---- matched Monte Carlo null -----------------------------------------
    mc = {}
    if not args.skip_mc:
        if os.path.exists(CTAU_FULL_NPZ):
            cdat = np.load(CTAU_FULL_NPZ, mmap_mode="r")
            C0 = np.asarray(cdat["C0"]).astype(np.float64)
            N_mc = int(cdat["N"])
            mc_source = os.path.basename(CTAU_FULL_NPZ)
        else:
            C0 = (z0c.T @ z0c) / N
            N_mc = N
            mc_source = "recomputed from centered z0_reembed"
        print(f"Matched MC null (C0 from {mc_source}, {MC_REPS} reps, seed {MC_SEED})...")
        for L in (LAGS if args.mc_all_lags else [L_STAR]):
            p95, mx = mc_null_lambda1(C0, N_mc, L * DT)
            mc[str(L)] = {"lambda1_mc_null_p95": p95, "lambda1_mc_null_max": mx}
            print(f"  L={L:5d}  MC p95={p95:.4e}  max={mx:.4e}")
        for r in rows:
            m = mc.get(str(r["L"]))
            if m is not None:
                r["mc_null_p95"] = m["lambda1_mc_null_p95"]
                r["mc_null_max"] = m["lambda1_mc_null_max"]
                r["lambda1_over_mc_p95"] = r["lambda1"] / m["lambda1_mc_null_p95"]
                r["mc_rejected"] = bool(r["lambda1"] > m["lambda1_mc_null_p95"])
                r["noise_dominated_mc"] = not r["mc_rejected"]

    # A lag is treated as noise-dominated if DA1 fails ANY available noise
    # reference at that lag (conservative).
    for r in rows:
        r["noise_dominated"] = bool(r["noise_dominated_analytic"]
                                    or r.get("noise_dominated_mc", False))

    # supplementary: overlap of v1(L) with PC1 of the same evaluation set,
    # which explains the short-lag behaviour (C(tau) -> C(0) as tau -> 0)
    cov0 = (z0c.T @ z0c) / N
    ev0, evec0 = np.linalg.eigh(cov0)
    pc1 = evec0[:, ev0.argsort()[::-1].copy()[0]]
    for r in rows:
        v = np.asarray(da[f"eigvecs_lag{r['L']}"])[:, 0].astype(np.float64)
        v = v / np.linalg.norm(v)
        r["abs_cos_with_pc1"] = float(abs(np.dot(v, pc1)))

    # ---- stationarity sensitivity (z_tau) — clearly separate ---------------
    ztc = z_tau - z_tau.mean(axis=0)
    gender_score_ztau = ztc @ axis
    ztau_rows = []
    for L in LAGS:
        v = np.asarray(da[f"eigvecs_lag{L}"])[:, 0].astype(np.float64)
        v = v / np.linalg.norm(v)
        if np.dot(v, v_star) < 0:
            v = -v
        rho, _ = spearmanr(ztc @ v, gender_score_ztau)
        ztau_rows.append({"L": int(L), "abs_spearman_rho_gender_ztau": float(abs(rho))})

    # ---- summary over lags above the noise reference -----------------------
    def _band(sel, key):
        ang = max(r["angle_to_v1_Lstar_deg"] for r in sel)
        at = [r["L"] for r in sel if r["angle_to_v1_Lstar_deg"] == ang][0]
        rv = [r["abs_spearman_rho_gender"] for r in sel]
        return {
            key + "_n_lags": len(sel),
            key + "_lags": [int(r["L"]) for r in sel],
            key + "_max_meaningful_angle_deg": float(ang),
            key + "_max_meaningful_angle_at_L": int(at),
            key + "_gender_abs_rho_min": float(min(rv)),
            key + "_gender_abs_rho_max": float(max(rv)),
        }

    meaningful = [r for r in rows if not r["noise_dominated"]]
    analytic_only = [r for r in rows if not r["noise_dominated_analytic"]]
    summary = {
        "n_lags_reported": len(rows),
        "n_lags_above_noise_reference": len(meaningful),
        "n_lags_noise_dominated": len(rows) - len(meaningful),
        "noise_reference_used": ("conservative: analytic free-conv edge AND "
                                 "matched MC null (95th pctl)"),
        "max_meaningful_angle_deg": max(r["angle_to_v1_Lstar_deg"]
                                        for r in meaningful),
        "gender_abs_rho_at_Lstar": row_star["abs_spearman_rho_gender"],
        "all_sanity_checks_passed": all_pass,
    }
    summary.update(_band(meaningful, "conservative"))
    summary.update(_band(analytic_only, "analytic_edge_only"))
    summary["max_meaningful_angle_at_L"] = summary[
        "conservative_max_meaningful_angle_at_L"]
    summary["gender_abs_rho_min"] = summary["conservative_gender_abs_rho_min"]
    summary["gender_abs_rho_max"] = summary["conservative_gender_abs_rho_max"]

    out = {
        "workstream": "P8: SDXL DA lag-sensitivity",
        "sources": {
            "da_npz": os.path.relpath(DA_NPZ, PROJECT_ROOT),
            "da_json": os.path.relpath(DA_JSON, PROJECT_ROOT),
            "z_npz": os.path.relpath(Z_NPZ, PROJECT_ROOT),
            "traj_json": os.path.relpath(TRAJ_JSON, PROJECT_ROOT),
            "C_tau_full_npz_for_mc": (os.path.relpath(CTAU_FULL_NPZ, PROJECT_ROOT)
                                      if os.path.exists(CTAU_FULL_NPZ) else None),
        },
        "config": {
            "prompt": da_json["prompt"],
            "guidance_scale": da_json["guidance_scale"],
            "dt": DT,
            "T_steps": da_json["T"],
            "N": N,
            "embed_dim": D,
            "L_star": L_STAR,
            "tau_star_phys": L_STAR * DT,
            "clip_model": CLIP_MODEL,
            "gender_probe_positive": GENDER_POS,
            "gender_probe_negative": GENDER_NEG,
            "gender_axis_route": "recomputed with real CLIP text encoder "
                                 "(axis not stored in NPZ/JSON; only cosines "
                                 "and correlations were stored)",
            "evaluation_set": "centered z0_reembed (re-embedded initial samples)",
            "mc_reps": MC_REPS,
            "mc_seed": MC_SEED,
            "freeconv_gamma": float(da["freeconv_gamma"]),
            "freeconv_sigma_f_sq": float(da["freeconv_sigma_f_sq"]),
            "freeconv_first_cross_idx": int(da["freeconv_first_cross_idx"]),
        },
        "sanity_checks": checks,
        "axis_provenance_validation": checks_axis,
        "rows": rows,
        "monte_carlo_null": mc,
        "stationarity_sensitivity_z_tau": ztau_rows,
        "summary": summary,
    }

    rec_path = os.path.join(SCRIPT_DIR, "mc_reconciliation.json")
    if os.path.exists(rec_path):
        out["mc_reconciliation"] = json.load(open(rec_path))
        print("Merged mc_reconciliation.json (run reconcile_mc_null.py to refresh)")
    else:
        print("NOTE: mc_reconciliation.json absent; run reconcile_mc_null.py first")

    json_path = os.path.join(SCRIPT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {json_path}")

    write_markdown(out)
    print(f"Saved: {os.path.join(SCRIPT_DIR, 'RESULTS.md')}")

    if not all_pass:
        print("\n*** ONE OR MORE MANDATORY SANITY CHECKS FAILED ***")
        sys.exit(2)


def fmt(x, spec="{:.4g}"):
    return "n/a" if x is None else spec.format(x)


def write_markdown(out):
    rows = out["rows"]
    s = out["summary"]
    c = out["config"]
    ck = out["sanity_checks"]

    L = []
    A = L.append
    A("# P8: SDXL DA lag-sensitivity (elderly prompt, w = 7.5, dt = 0.0025)\n")
    A("Read-only re-analysis of stored SDXL DA results. No new simulation, no GPU, "
      "no new SDXL sampling. The only model loaded is the real CLIP text encoder "
      "(`openai/clip-vit-large-patch14`) on CPU, used to rebuild the two fixed text "
      "probes of `experiments/sdxl/fig_elderly_da.py`.\n")
    A(f"- Prompt: `{c['prompt']}`, guidance `{c['guidance_scale']}`, "
      f"`dt = {c['dt']}`, `T = {c['T_steps']}` steps, N = {c['N']}, d = {c['embed_dim']}.")
    A(f"- Evaluation set (common to every lag): **{c['evaluation_set']}**; "
      "`score_L = (z0_reembed - mean) @ v1(L)`.")
    A(f"- Gender axis: `normalize(normalize(\"{c['gender_probe_positive']}\") - "
      f"normalize(\"{c['gender_probe_negative']}\"))`; {c['gender_axis_route']}.")
    A(f"- Reference direction: `v1(L* = {c['L_star']})`, "
      f"tau* = {c['tau_star_phys']} (dimensionless OU time).\n")

    A("## Sanity checks\n")
    A("| # | Check | Result | Value |")
    A("|---|-------|--------|-------|")
    A(f"| 1 | angle(v1(2000), v1(L*)) = 0 | "
      f"{'PASS' if ck['check1_angle_zero_at_Lstar']['passed'] else 'FAIL'} | "
      f"{ck['check1_angle_zero_at_Lstar']['angle_deg']:.2e} deg |")
    A(f"| 2 | common-set gender \\|rho\\| at L* ~ 0.90 | "
      f"{'PASS' if ck['check2_gender_rho_at_Lstar']['passed'] else 'FAIL'} | "
      f"{ck['check2_gender_rho_at_Lstar']['abs_rho']:.4f} |")
    A(f"| 3 | eig(C_tau_star) top vector matches eigvecs_lag2000[:,0] "
      f"(sign-invariant) | "
      f"{'PASS' if ck['check3_C_tau_star_eigvec']['passed'] else 'FAIL'} | "
      f"\\|<v,v*>\\| = {ck['check3_C_tau_star_eigvec']['abs_inner_product']:.6f} "
      f"({ck['check3_C_tau_star_eigvec']['sign_invariant_angle_deg']:.3f} deg) |")
    A(f"| 4 | lambda1(2000) ~ 9.96e-3 | "
      f"{'PASS' if ck['check4_lambda1_at_Lstar']['passed'] else 'FAIL'} | "
      f"{ck['check4_lambda1_at_Lstar']['lambda1_2000']:.5e} |")
    A(f"| 5 | lag arrays and free-conv profiles compatible | "
      f"{'PASS' if ck['check5_lag_array_compatibility']['passed'] else 'FAIL'} | "
      f"{ck['check5_lag_array_compatibility']['n_snapshot_lags']} lags, "
      f"{ck['check5_lag_array_compatibility']['n_lambda_plus_profile']} "
      f"lambda_plus values, strictly increasing |")
    av = out["axis_provenance_validation"]
    A(f"| + | (extra) recomputed gender axis reproduces the stored "
      f"cos(DA1, gender) magnitude | {'PASS' if av['passed'] else 'FAIL'} | "
      f"\\|cos\\| = {av['abs_cos_recomputed']:.7f} vs stored "
      f"{av['abs_cos_stored_in_json']:.7f} |")
    A("")
    A("The extra row confirms the CLIP text axis was rebuilt exactly. Note the "
      "pipeline's stored probe is oriented `woman - man` while "
      "`fig_elderly_da.py` uses `man - woman`; the two differ by a sign only, "
      "which is irrelevant here because all correlations are reported as "
      "\\|rho\\|.\n")

    A("## Main table\n")
    A("| L | tau = L dt | lambda1 | lambda2 | lambda2/lambda1 | angle to v1(L*) [deg] "
      "| gender \\|rho\\| | traj-avg rho^2 | lambda1 > lambda_+ (analytic) "
      "| matched MC, Null B (OU surrogate) |")
    A("|---:|---:|---:|---:|---:|---:|---:|---:|:--|:--|")
    for r in rows:
        traj = ("n/a (not stored)" if r["rho_sq_traj_averaged"] is None
                else f"{r['rho_sq_traj_averaged']:.4f}")
        mcs = "n/a"
        if "mc_rejected" in r:
            mcs = (f"yes ({r['lambda1_over_mc_p95']:.1f}x p95)"
                   if r["mc_rejected"] else "no")
            if r["L"] == L_STAR and out.get("mc_reconciliation"):
                aa = out["mc_reconciliation"]["null_A_pipeline_anisotropic"]
                mcs += (f"<br>**Null A: yes, "
                        f"{aa['ratio_obs_over_null_max']:.1f}x null max, "
                        f"p_emp <= 1e-4**")
        nd = " **NOISE-DOMINATED**" if r["noise_dominated"] else ""
        ang = (f"({r['angle_to_v1_Lstar_deg']:.1f})" if r["noise_dominated"]
               else f"{r['angle_to_v1_Lstar_deg']:.1f}")
        A(f"| {r['L']}{nd} | {r['tau_phys']:.3f} | {r['lambda1']:.4e} | "
          f"{r['lambda2']:.4e} | {r['lambda2_over_lambda1']:.3f} | "
          f"{ang} | "
          f"{r['abs_spearman_rho_gender']:.3f} | {traj} | "
          f"{'yes' if r['above_analytic_reference'] else 'no'} "
          f"({r['lambda1_over_lambda_plus']:.0f}x) | {mcs} |")
    A("")
    A("Rows marked **NOISE-DOMINATED** fail at least one available noise "
      "reference at that lag; their angles are shown in parentheses and are "
      "NOT interpreted.\n")
    A("Notes.")
    A("- `lambda1`, `lambda2` are the top two eigenvalues of the symmetrised "
      "C(tau), sorted by value (not by absolute value).")
    A("- Angles are computed after sign alignment against `v1(L*)` "
      "(flip if the inner product is negative).")
    A("- Trajectory-averaged rho^2 is stored only on the fixed-20 lag sub-grid "
      "(L = 20, 40, ..., 2000); it does not exist for L = 1 and is marked `n/a`.")
    A("- `lambda_+` is the stored analytic free-convolution edge "
      "(`freeconv_lambda_plus_profile`) aligned with `snapshot_lags`. This is an "
      "analytic null reference under its theorem's assumptions, not a proven "
      "bound for this data.")
    A("- The matched Monte Carlo column is **Null B**: an analytic Gaussian OU "
      f"surrogate with the empirical C(0), {c['mc_reps']} replicates, seed "
      f"{c['mc_seed']}, using the `fig_elderly_da.py` spectrum-figure recipe. It "
      "is NOT the null behind the '8.1x' figure in the project notes (that is "
      "**Null A**, the pipeline-matched anisotropic null, available only at L*). "
      "See the reconciliation section below. The column reports rejection "
      "against the Null B 95th percentile, which is stable in replicate count "
      "(4.11e-3 / 4.04e-3 / 4.16e-3 at R = 200 / 1000 / 5000); the Null B "
      "*maximum* is not, so margins are quoted against the percentile.")
    A("")

    A("## Summary\n")
    A("Two noise references are available and they disagree, so both are "
      "reported. The analytic free-convolution edge places DA1 above the null "
      "at every lag; the matched Monte Carlo null (same OU recipe as the paper's "
      "spectrum figure) does not, because at short lag C(tau) is still close to "
      "C(0) and a pure-OU null with the empirical C(0) already produces a top "
      "eigenvalue of comparable size. The conservative reading uses both.\n")
    A(f"- Lags reported: {s['n_lags_reported']}; passing both references: "
      f"{s['n_lags_above_noise_reference']}; noise-dominated under at least one "
      f"reference: {s['n_lags_noise_dominated']}.")
    A(f"- **Conservative (both references), lags "
      f"{s['conservative_lags']}: maximum meaningful angle to `v1(L*)` = "
      f"{s['conservative_max_meaningful_angle_deg']:.1f} deg** (at L = "
      f"{s['conservative_max_meaningful_angle_at_L']}); "
      f"**gender |rho| range {s['conservative_gender_abs_rho_min']:.3f} to "
      f"{s['conservative_gender_abs_rho_max']:.3f}**.")
    A(f"- Analytic edge only (all "
      f"{s['analytic_edge_only_n_lags']} lags): maximum angle "
      f"{s['analytic_edge_only_max_meaningful_angle_deg']:.1f} deg (at L = "
      f"{s['analytic_edge_only_max_meaningful_angle_at_L']}); gender |rho| range "
      f"{s['analytic_edge_only_gender_abs_rho_min']:.3f} to "
      f"{s['analytic_edge_only_gender_abs_rho_max']:.3f}.")
    A(f"- At L* itself the common evaluation set gives gender |rho| = "
      f"{s['gender_abs_rho_at_Lstar']:.3f}, reproducing the reported ~0.90.")
    A("")
    A("**Verdict.** DA1 is not lag-invariant across the whole stored grid: it "
      "rotates monotonically as the lag grows and only settles at long lag. "
      "Within the lags where DA1 is above both noise references "
      f"(tau >= {min(r['tau_phys'] for r in rows if not r['noise_dominated']):.1f}), "
      "the direction is stable to within "
      f"{s['conservative_max_meaningful_angle_deg']:.0f} deg of `v1(L*)` and the "
      "gender interpretation is stable "
      f"({s['conservative_gender_abs_rho_min']:.2f}-"
      f"{s['conservative_gender_abs_rho_max']:.2f}). "
      "At the shortest lag (L = 1) DA1 is essentially PC1 "
      f"(|cos| = {rows[0]['abs_cos_with_pc1']:.2f}), which is expected because "
      "C(tau) -> C(0) as tau -> 0, and it is also the regime the matched MC null "
      "cannot separate from noise; the PC1 overlap then collapses and settles "
      f"near {rows[-1]['abs_cos_with_pc1']:.2f} at L*, matching the reported "
      "|cos(DA1, PC1)| = 0.514. The headline result is therefore a long-lag "
      "property, not an artefact of the single choice L* = 2000, but it is also "
      "not lag-free: reporting DA1 at a short lag would give a materially "
      "different direction and a much weaker gender correlation.")
    A("")

    rec = out.get("mc_reconciliation")
    if rec:
        a = rec["null_A_pipeline_anisotropic"]
        b = rec["null_B_analytic_OU_surrogate"]
        bb = b["under_null_A_bootstrap_procedure"]
        v = rec["verdict"]
        A("## Reconciliation of the two MC margins at L* (8.1x vs 2.1x)\n")
        A("Both numbers are computed from the SAME observed "
          f"lambda_1 = {rec['observed_lambda1_at_Lstar']:.4e}. The disagreement "
          "is entirely in the NULL, and it is caused by the null CONSTRUCTION, "
          "not by the number of replicates. Both are internally correct for "
          "their own null.\n")
        A("| | Null A (pipeline anisotropic) | Null B (analytic OU surrogate) |")
        A("|---|---|---|")
        A("| What is simulated | N(mu, Sigma_hat_latent) initial latents pushed "
          "through the REAL SDXL Langevin (T=2000, dt=0.0025), VAE-decoded and "
          "CLIP-embedded | closed-form OU in CLIP space: "
          "x_tau = exp(-tau/2) z0 + sqrt(1-exp(-tau)) zeta, "
          "z0, zeta ~ N(0, C_hat(0)); no SDXL, no VAE, no CLIP |")
        A(f"| Replicates | 17 GPU runs pooled to N_big = {a['N_big']} pairs, then "
          f"bootstrap B = {a['B_boot']} draws of n = {a['n_resample']} "
          f"| {c['mc_reps']} independent replicates of n = 200 |")
        A("| 'null max' means | max over the 10,000 bootstrap draws of the "
          "rank-1 eigenvalue BY MAGNITUDE (per rank, not across ranks) "
          "| max over the replicates of lambda_1 |")
        A(f"| null_max_abs | {a['recomputed_null_max_abs']:.4e} "
          f"| {b['as_used_in_lag_table_200_reps_seed42']['null_max']:.4e} |")
        A(f"| null_mean_signed | {a['null_mean_signed']:.4e} | "
          f"{bb['null_mean_signed']:.4e} (matched-procedure variant, see 3) |")
        A(f"| obs / null_max | **{a['ratio_obs_over_null_max']:.2f}x** | "
          f"{b['as_used_in_lag_table_200_reps_seed42']['ratio_obs_over_null_max']:.2f}x |")
        A("")
        A("**1. Which null is which.** The '8.1x' in the project notes comes "
          "from `experiments/sdxl/results/sdxl_anisotropic_null_pvalues.json` "
          "(`pvalues.lambda_1.null_max_abs`), produced by "
          "`older_versions/experiments/mc_dip_validation/sdxl_iso_null_via_late_t/"
          "compute_bootstrap_pvalue.py` from the pool "
          "`sdxl_null_anisotropic_combined.npz`. It is the **anisotropic, "
          "pipeline-matched** null, not an iso-Gaussian one: the notes' phrase "
          "'ISO-GAUSSIAN floor' is a mislabel. (A genuinely iso-Gaussian "
          "late-t null exists as a separate, differently-constructed "
          "experiment in the same folder; it is not the source of 8.1x.) "
          "'null max' is the maximum over the 10,000 bootstrap draws of the "
          "rank-1 magnitude, i.e. over draws, per rank.")
        A(f"\n**2. Both are internally correct.** Re-running "
          f"`compute_bootstrap_pvalue.py`'s procedure on the stored pool with "
          f"its own seed ({a['seed']}) reproduces "
          f"null_max_abs = {a['recomputed_null_max_abs']:.6e} against the stored "
          f"{a['stored_null_max_abs']:.6e} "
          f"({'exact match' if a['exactly_reproduced'] else 'MISMATCH'}), giving "
          f"{a['ratio_obs_over_null_max']:.2f}x. Neither number is stale; they "
          "answer different questions.")
        A(f"\n**3. Apples-to-apples.** Holding Null A's bootstrap procedure "
          f"completely fixed (same N_big = {bb['N_big']}, n = {bb['n_resample']}, "
          f"B = {bb['B_boot']}, seed = {bb['seed']}) and swapping ONLY the pool "
          f"construction to Null B moves the null maximum from "
          f"{a['recomputed_null_max_abs']:.4e} to {bb['null_max_abs']:.4e}, a "
          f"factor of {v['ratio_null_B_over_null_A_same_procedure']:.2f}, and the "
          f"margin from {a['ratio_obs_over_null_max']:.2f}x to "
          f"{bb['ratio_obs_over_null_max']:.2f}x. Construction alone accounts for "
          "the entire discrepancy.")
        A(f"\n**Why.** Null A's pool has essentially decorrelated by tau* "
          f"(null_mean_signed = {a['null_mean_signed']:.3e}), so its lambda_1 is "
          f"pure finite-sample fluctuation. Null B BUILDS IN a lag coupling "
          f"rho = exp(-tau*/2) = {b['rho_built_into_null']:.4f} in every "
          f"direction, including the top direction of C_hat(0). Its systematic "
          f"term alone is rho * lambda_1(C_hat(0)) = "
          f"{b['expected_systematic_top_eigenvalue_rho_times_lambda1_C0']:.3e}, "
          f"already ~{b['expected_systematic_top_eigenvalue_rho_times_lambda1_C0'] / a['recomputed_null_max_abs']:.1f}x "
          "Null A's ENTIRE null maximum before any fluctuation is added. Null B "
          "is therefore the far more conservative null: it asks 'is DA1 slower "
          "than a uniform-rate OU?', while Null A asks 'is DA1 above pure "
          "sampling noise?'.")
        rc = rec["replicate_count_control"]
        A("\n**4. Replicate count is not the cause, and the direction refutes "
          "it.** A running maximum cannot decrease, so more replicates can only "
          "raise a null max and lower the margin. Null B, independent "
          f"replicates, seed {rc['seed']}:\n")
        A("| replicates R | Null B null_max_abs | obs / null_max |")
        A("|---:|---:|---:|")
        for cp in rc["checkpoints"]:
            A(f"| {cp['n_replicates']} | {cp['null_max_abs']:.4e} | "
              f"{cp['ratio_obs_over_null_max']:.2f}x |")
        A(f"\nMonotone non-decreasing in R: "
          f"{rc['null_max_is_monotone_nondecreasing_in_R']}; margin monotone "
          f"non-increasing: {rc['ratio_monotone_nonincreasing_in_R']}. Going "
          "from 200 to 10,000 replicates would push Null B's margin further "
          "BELOW 1.6x, never up to 8.1x. Sample size is refuted as the "
          "explanation.\n")
        A("**What to quote.** Against the pipeline-matched anisotropic null "
          f"(Null A, the paper's null): **{a['ratio_obs_over_null_max']:.1f}x "
          f"the null maximum** over 10,000 bootstrap draws, "
          f"p_emp <= 1e-4. Against the conservative uniform-rate OU surrogate "
          f"(Null B): **{b['as_used_in_lag_table_200_reps_seed42']['ratio_obs_over_null_p95']:.1f}x "
          "the 95th percentile** at 200 replicates. Always name the null. Null A "
          "is available ONLY at L* = 2000, because the stored null pool contains "
          "z0 and z_tau at tau* alone; that is why the per-lag column uses "
          "Null B.\n")

    A("## Limitations\n")
    A(f"- The free-convolution stopping rule never actually triggered on this "
      f"run (`freeconv_first_cross_idx = {c['freeconv_first_cross_idx']}`), so "
      f"L* = {c['L_star']} is the fallback value equal to the full trajectory "
      f"length T, not a detected crossing. Lag sensitivity therefore cannot be "
      f"probed beyond tau = {c['tau_star_phys']}: nothing longer was stored.")
    A("- Trajectory-averaged rho^2 is stored only on the fixed-20 sub-grid, so "
      "L = 1 has no value. Its stored profile is also strongly non-monotone in "
      "L (it dips to ~1e-4 near L = 600 and rises again), so the column is "
      "reported as-is and not used to rank lags.")
    A("- The analytic free-convolution edge is a null reference under its "
      "theorem's assumptions; it is not a proven bound for this data, and here "
      "it is roughly an order of magnitude more permissive than the matched "
      "Monte Carlo null at the same lag. Do not read the 15x-21x margins as "
      "significance.")
    A(f"- The matched Monte Carlo null uses {c['mc_reps']} replicates at a "
      f"single seed ({c['mc_seed']}); its 95th percentile carries Monte Carlo "
      "error, and rows near 1.0x (L = 400) should be treated as borderline.")
    A("- The two MC margins (8.1x and ~2.1x) are RESOLVED above: different "
      "nulls, both internally correct. The residual risk is presentational -- "
      "any quoted margin must name its null. Null A exists only at L*, so the "
      "per-lag noise marking rests on Null B plus the analytic edge.")
    A("- Null A itself rests on only 17 independent GPU replicates (850 "
      "trajectories) resampled 10,000 times; the bootstrap draws share those 17 "
      "simulations, so its p-value is conservative with respect to "
      "between-simulation variance, as its own source file notes. Its "
      "'p_emp <= 1e-4' is a bootstrap resolution limit, not evidence of a "
      "1e-4 tail.")
    A("- Single prompt, single guidance scale, single dt, N = 200. No "
      "cross-seed replication of the SDXL run itself is available, so the "
      "angles carry no sampling uncertainty from the generative side.")
    A("- The gender axis had to be recomputed (only cosines/correlations were "
      "stored, not the axis). Agreement with the stored cosine to 7 significant "
      "figures confirms it is the same axis.")
    A("")

    A("## Supplementary: overlap of v1(L) with PC1 of the same evaluation set\n")
    A("| L | tau | \\|cos(v1(L), PC1)\\| | angle to v1(L*) [deg] |")
    A("|---:|---:|---:|---:|")
    for r in rows:
        A(f"| {r['L']} | {r['tau_phys']:.3f} | {r['abs_cos_with_pc1']:.3f} | "
          f"{r['angle_to_v1_Lstar_deg']:.1f} |")
    A("")

    A("## Supplementary: stationarity sensitivity on z_tau (NOT the headline)\n")
    A("Same directions, but both the projection and the gender score are computed "
      "on centered `z_tau` (the samples at the end of the Langevin run) instead of "
      "the re-embedded initial samples. Reported only as a sensitivity; the paper's "
      "reported correlation is the `z0_reembed` one above.\n")
    A("| L | gender \\|rho\\| on z_tau |")
    A("|---:|---:|")
    for r in out["stationarity_sensitivity_z_tau"]:
        A(f"| {r['L']} | {r['abs_spearman_rho_gender_ztau']:.3f} |")
    A("")

    with open(os.path.join(SCRIPT_DIR, "RESULTS.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
