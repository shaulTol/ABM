"""
Bootstrap CI for the elderly DA1↔gender Spearman correlation.

Uses the existing free-conv τ*=2000, dt=0.0025, N=200 elderly run
(SDXL Base 1024×1024, w=7.5).

Inputs (auto-located):
  - clip_da_elderly_gs7p5_fc_z_tau.npz    → z0_reembed (200,768) + z_tau (200,768)
  - clip_da_elderly_gs7p5_fc_dt0p0025.npz → da1_eigvec (saved point-estimate),
                                            mu_clip
Gender axis: CLIP-ViT-L/14 text directions
  "a photograph of a woman" − "a photograph of a man"

Outputs (this dir):
  - results.json
  - bootstrap_histogram.png
  - RESULTS.md
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ELDERLY_DA = HERE.parent
RESULTS_DA = ELDERLY_DA / "results_da_pcavda"

Z_TAU_NPZ = HERE / "clip_da_elderly_gs7p5_fc_z_tau.npz"
PE_NPZ = RESULTS_DA / "clip_da_elderly_gs7p5_fc_dt0p0025.npz"
PE_JSON = RESULTS_DA / "clip_da_elderly_gs7p5_fc_dt0p0025.json"

print(f"[paths]")
print(f"  z_tau:          {Z_TAU_NPZ}")
print(f"  point-estimate: {PE_NPZ}")
print(f"  pe-json:        {PE_JSON}")

assert Z_TAU_NPZ.exists(), f"missing: {Z_TAU_NPZ}"
assert PE_NPZ.exists(),    f"missing: {PE_NPZ}"
assert PE_JSON.exists(),   f"missing: {PE_JSON}"

zt = np.load(Z_TAU_NPZ, allow_pickle=True)
z0 = np.asarray(zt["z0_reembed"]).astype(np.float64)   # (200, 768)
z_tau = np.asarray(zt["z_tau"]).astype(np.float64)     # (200, 768)
N, D = z0.shape
print(f"[data] z0={z0.shape}  z_tau={z_tau.shape}  (N={N}, D={D})")
assert N == 200 and D == 768

pe = np.load(PE_NPZ, allow_pickle=True)
saved_da1 = np.asarray(pe["da1_eigvec"]).astype(np.float64)
saved_da1 /= np.linalg.norm(saved_da1)
mu_clip   = np.asarray(pe["mu_clip"]).astype(np.float64)  # global mean from original run
print(f"[saved] da1_eigvec norm={np.linalg.norm(saved_da1):.6f}  mu_clip[:3]={mu_clip[:3]}")

with open(PE_JSON) as f:
    pe_json = json.load(f)
saved_rho = float(pe_json["da1_correlations"]["gender"]["rho"])
print(f"[saved] paper |ρ(DA1, gender)| = {abs(saved_rho):.4f}  (signed: {saved_rho:+.4f})")

# ── Gender axis (CLIP text directions) ─────────────────────────────────
print("\n[clip] loading openai/clip-vit-large-patch14 (text encoder only)…")
import torch
from transformers import CLIPModel, CLIPTokenizer

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"[clip] device={device}")
tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    torch_dtype=torch.float32,
).to(device).eval()

@torch.no_grad()
def encode_text(text):
    """Encode a text string into the CLIP shared embedding space (768-D, L2-norm).

    transformers v5's `get_text_features` no longer applies the text_projection
    (verified empirically: gives cos≈0.012 against the saved gender axis,
    versus the correct ~0.568). Use text_model → pooler_output → text_projection
    explicitly, which matches transformers v4 behaviour.
    """
    inp = tok([text], return_tensors="pt", padding=True, truncation=True).to(device)
    out = clip_model.text_model(**inp)
    pooled = out.pooler_output             # EOS-token pooled hidden state
    e = clip_model.text_projection(pooled) # project into shared CLIP space
    e = e / e.norm(dim=-1, keepdim=True)
    return e.cpu().numpy().squeeze(0).astype(np.float64)

woman = encode_text("a photograph of a woman")
man   = encode_text("a photograph of a man")
gender_axis = woman - man
gender_axis /= (np.linalg.norm(gender_axis) + 1e-12)
print(f"[clip] gender_axis ‖·‖={np.linalg.norm(gender_axis):.6f}, "
      f"first 3 dims={gender_axis[:3]}")

gender_labels = z_tau @ gender_axis  # (N,)
print(f"[gender] labels range=[{gender_labels.min():+.4f}, {gender_labels.max():+.4f}]")

# Free GPU memory; we don't need CLIP again.
del clip_model
if device == "cuda":
    torch.cuda.empty_cache()


# ── Helpers ───────────────────────────────────────────────────────────
def c_sym(z0_b, z_tau_b, mu):
    """Symmetrized lag-τ covariance Ĉ_sym(τ) on a centered ensemble."""
    z0_c = z0_b - mu
    zt_c = z_tau_b - mu
    Nb = z0_b.shape[0]
    C = (zt_c.T @ z0_c) / Nb
    return 0.5 * (C + C.T)


def leading_eig(C):
    """Top eigenpair of a symmetric matrix."""
    w, V = np.linalg.eigh(C)        # ascending
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    return w, V


# ── Point estimate: replicate ─────────────────────────────────────────
print("\n=== Point estimate ===")
# Try the saved mu_clip first (canonical: averaged across all 111 lags).
mu_local = (z0.mean(axis=0) + z_tau.mean(axis=0)) / 2.0

variants = [
    ("saved_mu_clip", mu_clip),
    ("local_(z0+z_tau)/2", mu_local),
]
best_pe = None
for tag, mu in variants:
    C = c_sym(z0, z_tau, mu)
    w, V = leading_eig(C)
    da1 = V[:, 0].copy()
    # Sign-align so ⟨da1, saved_da1⟩ > 0
    if da1 @ saved_da1 < 0:
        da1 = -da1
    proj = z_tau @ da1
    from scipy.stats import spearmanr
    rho, p = spearmanr(proj, gender_labels)
    cos_with_saved = float(da1 @ saved_da1)
    gap = float(w[0] / w[1])
    print(f"  [{tag}] λ1={w[0]:+.5f} λ2={w[1]:+.5f} gap={gap:.3f}"
          f"  cos(da1, saved)={cos_with_saved:.5f}  ρ={rho:+.4f} (|ρ|={abs(rho):.4f})")
    if best_pe is None or abs(abs(rho) - abs(saved_rho)) < abs(abs(best_pe[3]) - abs(saved_rho)):
        best_pe = (tag, da1, w, rho, p, gap, cos_with_saved, mu)

tag, point_da1, point_eigs, point_rho, point_p, point_gap, point_cos, mu_used = best_pe
print(f"  → using {tag}  |ρ|={abs(point_rho):.4f}  cos(da1,saved)={point_cos:.4f}")

if abs(abs(point_rho) - abs(saved_rho)) > 0.02:
    raise RuntimeError(
        f"Point estimate |ρ|={abs(point_rho):.4f} differs from saved "
        f"|ρ|={abs(saved_rho):.4f} by more than 0.02."
    )
print(f"  ✓ point estimate replicated (within 0.02 of saved {abs(saved_rho):.4f})")

# ── Bootstrap ─────────────────────────────────────────────────────────
print("\n=== Bootstrap (B=2000) ===")
B = 2000
rng = np.random.default_rng(20260503)
abs_rhos = np.empty(B, dtype=np.float64)
gaps     = np.empty(B, dtype=np.float64)
sign_flips = 0
nans = 0

from scipy.stats import spearmanr

for b in range(B):
    idx = rng.integers(0, N, size=N)  # with replacement
    z0_b = z0[idx]
    zt_b = z_tau[idx]
    # Recompute mu on the resampled ensemble (consistent with the per-replicate
    # estimator). We chose `mu_used` matching the point-estimate variant above.
    if tag == "saved_mu_clip":
        mu_b = mu_clip
    else:
        mu_b = (z0_b.mean(axis=0) + zt_b.mean(axis=0)) / 2.0
    C_b = c_sym(z0_b, zt_b, mu_b)
    w_b, V_b = leading_eig(C_b)
    v1 = V_b[:, 0].copy()
    if v1 @ point_da1 < 0:
        v1 = -v1
        sign_flips += 1
    proj_fixed = z_tau @ v1  # project the FIXED original embeddings
    rho_b, _ = spearmanr(proj_fixed, gender_labels)
    if not np.isfinite(rho_b):
        nans += 1
        rho_b = 0.0
    abs_rhos[b] = abs(rho_b)
    gaps[b]     = w_b[0] / w_b[1] if w_b[1] != 0 else np.inf
    if (b + 1) % 250 == 0:
        print(f"  {b+1:5d}/{B}  median |ρ|={np.median(abs_rhos[:b+1]):.4f}  "
              f"median gap={np.median(gaps[:b+1]):.3f}  flips={sign_flips}")

print(f"\n[summary] nans={nans}  sign_flips={sign_flips}/{B} ({100*sign_flips/B:.1f}%)")

# Sanity checks
assert nans == 0, f"{nans} bootstrap replicates produced NaN ρ"
median_gap = float(np.median(gaps))
assert median_gap > 1.2, f"median gap {median_gap:.3f} not > 1.2"
print(f"✓ all {B} replicates finite, median λ1/λ2={median_gap:.3f}")

# ── Stats ─────────────────────────────────────────────────────────────
mean_abs = float(np.mean(abs_rhos))
se_abs   = float(np.std(abs_rhos, ddof=1))
lo, hi   = np.percentile(abs_rhos, [2.5, 97.5])
frac_above_08 = float((abs_rhos > 0.8).mean())

print(f"\n=== Bootstrap CI ===")
print(f"  point   |ρ| = {abs(point_rho):.4f}")
print(f"  mean    |ρ| = {mean_abs:.4f}")
print(f"  SE      |ρ| = {se_abs:.4f}")
print(f"  95% CI      = [{lo:.4f}, {hi:.4f}]")
print(f"  P(|ρ| > 0.8) = {frac_above_08:.4f}")
print(f"  median gap   = {median_gap:.3f}  (range [{gaps.min():.3f}, {gaps.max():.3f}])")

# ── Save deliverables ────────────────────────────────────────────────
results = {
    "point_estimate_abs_rho": float(abs(point_rho)),
    "point_estimate_signed_rho": float(point_rho),
    "saved_paper_signed_rho": float(saved_rho),
    "point_da1_cos_with_saved": float(point_cos),
    "point_eigenvalue_gap_lambda1_over_lambda2": float(point_gap),
    "B": B,
    "bootstrap_mean_abs_rho": mean_abs,
    "bootstrap_se_abs_rho": se_abs,
    "bootstrap_ci95_abs_rho": [float(lo), float(hi)],
    "fraction_abs_rho_gt_0p8": frac_above_08,
    "bootstrap_median_lambda1_over_lambda2": median_gap,
    "bootstrap_min_lambda1_over_lambda2": float(gaps.min()),
    "bootstrap_max_lambda1_over_lambda2": float(gaps.max()),
    "bootstrap_sign_flip_count": int(sign_flips),
    "bootstrap_sign_flip_rate": float(sign_flips) / B,
    "nans": int(nans),
    "mu_variant_used": tag,
    "rng_seed": 20260503,
    "inputs": {
        "z_tau_npz": str(Z_TAU_NPZ),
        "point_estimate_npz": str(PE_NPZ),
        "point_estimate_json": str(PE_JSON),
    },
    "configuration": {
        "T": int(pe_json["T"]),
        "dt": float(pe_json["dt"]),
        "n_traj": int(pe_json["n_traj"]),
        "tau_star_lag": 2000,
        "embed_dim": 768,
        "model": pe_json.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0"),
        "prompt": pe_json.get("prompt", "a portrait of an elderly person"),
        "guidance_scale": float(pe_json.get("guidance_scale", 7.5)),
        "clip_model": "openai/clip-vit-large-patch14",
        "gender_axis": "normalize(CLIP('a photograph of a woman') - CLIP('a photograph of a man'))",
    },
}
out_json = HERE / "results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[saved] {out_json}")

# ── Histogram ────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.hist(abs_rhos, bins=40, color="#3a7bd5", edgecolor="white", linewidth=0.6)
ax.axvline(abs(point_rho), color="black", linestyle="--", linewidth=1.4,
           label=f"point |ρ|={abs(point_rho):.3f}")
ax.axvline(lo, color="0.4", linestyle=":", linewidth=1.0)
ax.axvline(hi, color="0.4", linestyle=":", linewidth=1.0,
           label=f"95% CI [{lo:.3f}, {hi:.3f}]")
ax.set_xlabel(r"$|\rho|$  (Spearman, DA1 proj. vs. gender)")
ax.set_ylabel("count")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
out_png = HERE / "bootstrap_histogram.png"
fig.savefig(out_png, dpi=200)
plt.close(fig)
print(f"[saved] {out_png}")

# ── RESULTS.md ────────────────────────────────────────────────────────
latex = (
    f"Bootstrap over the $N=200$ trajectories ($B={B}$, DA direction "
    f"recomputed per replicate, sign-aligned to the point-estimate DA1) "
    f"gives $|\\rho| = {abs(point_rho):.2f}$ "
    f"[95\\% CI: {lo:.2f}, {hi:.2f}]."
)

md = f"""# Elderly DA1↔gender bootstrap CI

Point |ρ| = {abs(point_rho):.4f} (paper: {abs(saved_rho):.4f}). Bootstrap mean = {mean_abs:.4f}, SE = {se_abs:.4f}, 95% CI = [{lo:.4f}, {hi:.4f}].
B = {B} replicates, all finite; sign-flip rate {sign_flips/B*100:.1f}%, median λ₁/λ₂ = {median_gap:.2f}; P(|ρ|>0.8) = {frac_above_08:.3f}.
DA recomputed per resample, sign-aligned to point-estimate DA1; projection on the same fixed N=200 final-snapshot embeddings; gender labels = projection onto CLIP-ViT-L/14 text direction normalize("a photograph of a woman" − "a photograph of a man").

## LaTeX line (ready to paste)

> {latex}

## Files
- `results.json` — full numerical summary
- `bootstrap_histogram.png` — histogram of bootstrap |ρ|
- `run_bootstrap.py` — this script
"""
out_md = HERE / "RESULTS.md"
out_md.write_text(md)
print(f"[saved] {out_md}")

print("\n=== LaTeX line ===")
print(latex)
