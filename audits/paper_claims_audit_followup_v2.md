# Paper Claims Audit — Follow-up v2

**Date:** 2026-05-03

---

## Task A: Table 2 |r| Consistency

**All 10 rows use `scipy.stats.pearsonr` with `abs()`. Angles are raw radians treated as scalars (not circular). No Spearman or circular correlation anywhere.**

| Row | Statistic | Array A (projection) | Array B (angle) | N | Script:line |
|-----|-----------|---------------------|-----------------|---|-------------|
| DA1 | Pearson \|r\| | `centered @ eigh(C_τ)[:,-1]` (linear, 45D) | ψ (rad) | 750k | `save_alanine_C_tau_h_half_10k.py:165` |
| DA2 | Pearson \|r\| | `centered @ eigh(C_τ)[:,-2]` (linear, 45D) | φ (rad) | 750k | `save_alanine_C_tau_h_half_10k.py:166` |
| TICA1 | Pearson \|r\| | `centered @ tica_evecs[:,0]` (linear, 45D) | ψ (rad) | 750k | `compute_da_comparison.py:366-368` |
| TICA2 | Pearson \|r\| | `centered @ tica_evecs[:,1]` (linear, 45D) | φ (rad) | 750k | `compute_da_comparison.py:366-368` |
| PCA1 | Pearson \|r\| | `centered @ pca_evecs[:,0]` (linear, 45D) | ψ/φ (rad) | 750k | `compute_da_comparison.py:379-381` |
| PCA2 | Pearson \|r\| | `centered @ pca_evecs[:,1]` (linear, 45D) | ψ/φ (rad) | 750k | `compute_da_comparison.py:379-381` |
| SM1 | Pearson \|r\| | `embed[:,0]` (**nonlinear** NN embedding) | ψ (rad) | 750k | `run_spectral_map.py:146-149` |
| SM2 | Pearson \|r\| | `embed[:,1]` (**nonlinear** NN embedding) | φ (rad) | 750k | `run_spectral_map.py:146-149` |
| DMψ₂ | Pearson \|r\| | `psi_vecs[:,1]` (kernel eigvec) | ψ (rad) | **2k** | `run_diffusion_maps.py:141-143` |
| DMψ₃ | Pearson \|r\| | `psi_vecs[:,2]` (kernel eigvec) | φ (rad) | **2k** | `run_diffusion_maps.py:141-143` |

### Inconsistencies Found

1. **Sample size mismatch**: DA/TICA/PCA/SM use all 750k frames. DM uses only 2k subsampled frames (kernel matrix is N²). Correlations are noisier for DM; the 5-seed averaging partially compensates.

2. **Projection type mismatch**: DA/TICA/PCA produce linear projections in 45D distance space. SM produces nonlinear NN embedding coordinates. DM produces kernel eigenvector coordinates. All are reported as "|r(ψ)|" without distinguishing linear vs nonlinear — methodologically defensible (each method's best 1D summary vs dihedral), but worth noting.

3. **Circular angle treatment**: All methods treat φ/ψ as raw scalars for Pearson. Since these are periodic angles in [-π,π], Pearson |r| is not a proper circular-linear correlation measure. However, this is **uniform across all rows** — no inter-method bias. Using circular correlation would change all values equally.

**No inconsistency in the statistic itself** — all rows use identical `abs(pearsonr(projection, angle)[0])`.

---

## Task B: SDXL Eigenvalue Sort Fix

**Changed** `eigvals.abs().argsort(descending=True)` → `eigvals.argsort(descending=True)` in 4 locations:

| File | Line | Context |
|------|------|---------|
| `sdxl_components.py` | 253 | Main eigendecomposition in lag loop |
| `sdxl_components.py` | 286 | Free-conv stopping (previous lag) |
| `sdxl_components.py` | 366 | Fallback when free-conv stop never triggers |
| `compute_sdxl_ssa_v2.py` | 406 | Older SSA v2 computation |

**DA1/DA2 unchanged**: Top-2 eigenvalues at τ*=2000 are 0.00996, 0.00512 — both positive and well-separated from |λ_min|=0.00344. The abs-sort and value-sort give identical ordering for the top-2. Difference only affects rank 4+ (e.g., negative λ=-0.00344 previously ranked above positive λ=0.00311).

---

## Task C: Dispersion Values and dt Bug

### Recomputed values (via `regenerate_all.py --only dispersion`)

| Distribution | SSA (stdout) | H (stdout) | Paper Ŝ | Paper H |
|---|---|---|---|---|
| GMM | 0.86 (S_old) | 2.11 nats | 1.822 | 5.14 |
| Blob | 0.20 (S_old) | 2.84 nats | 1.022 | 6.06 |

SSA stdout shows **S_old** (mean of ρ²), not Ŝ (Riemann sum). Paper Ŝ values verified from raw ρ data in `dispersion_rho_dt01.csv` (Ŝ_GMM=1.825, Ŝ_blob=1.023, matching paper within 0.003).

Entropy remains unresolved: code gives H in nats (2.11/2.84), paper claims 5.14/6.06 — no committed code reproduces these.

### The `dt=` bug explained

**Root cause**: `compute_1d_ssa()` (line 204) calls `compute_vector_ssa_score()` without `dt=`. Inside, `aggregate_ssa(rho_sq, dt=None, normalize='riemann')` hits the fallback (line 324): `return float(rho_sq.mean())` — returning S_old instead of Ŝ.

**What it affects: both stdout AND figure annotations.**
- Stdout: prints `SSA=0.86` / `SSA=0.20` (S_old, wrong)
- PNG annotation: `ax.text(...)` at line 288-289 shows `SSA = 0.86` / `SSA = 0.20` on `fig_dispersion_vs_fragmentation.png` (wrong)

**However**: The standalone script `figures/make_fig_dispersion_vs_fragmentation.py` (the 2D version) does NOT have this bug — it correctly passes `dt=` to `aggregate_ssa()` at line 135. If the paper figure was generated from this standalone script, the PNG in the paper is correct.

**Fix**: Add `dt=dt` to the `compute_vector_ssa_score()` call in `compute_1d_ssa()`.

---

## Task D: Planted Spikes Dip Behavior

**Diagnosis: Explanation (a) — DA1 mixes ~8 effective bimodal axes via CLT → unimodal marginal.**

The 20 planted axes all have **identical** double-well parameters (c=3.0, σ_small=0.3), so the top 20 eigenvalues of C_sym(τ*) are nearly degenerate:

- λ₁/λ₂ = 1.019 (20-fold degeneracy)
- DA1 lies 95.4% within the planted subspace but is **62° from the nearest individual planted axis**
- DA1 is a linear combination of ~8 effective bimodal components (effective dim = 1/Σw_j⁴ ≈ 8.1)

Projecting onto DA1 gives `proj = Σ_j w_j X_j` where each X_j is bimodal (±3). This is a mixture of 2²⁰ ≈ 1M Gaussian components whose means follow a CLT bell curve → unimodal.

**Paper does NOT claim DA1 is bimodal for k=20.** The planted spikes experiment claims:
- **Spike counting**: detected count matches planted k (panel e)
- **Subspace recovery**: DA top-20 aligns 94.8% with planted top-20 (panel f)
- **Sample complexity** (panel d)

No claim about 1D bimodality of individual eigenvectors. The dip failure is expected and not a problem.

**For k=1**: DA1 aligns with the single planted axis (14.4° error) → strongly bimodal (dip p≈0). This confirms the method works when there's a non-degenerate leading direction.
