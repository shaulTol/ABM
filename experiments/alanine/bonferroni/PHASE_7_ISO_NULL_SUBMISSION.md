# Phase 7: Isotropic Gaussian Null — Submission Log

**Date:** 2026-05-02
**Purpose:** Run MC null test with isotropic Gaussian null N(0, σ²_f · I_d) instead of matched-covariance N(0, C₀), to test whether DA2 passes under a fairer null.

---

## 1. Motivation

Phase 4 showed that the N(0, C₀) null inherits the data's anisotropy: C₀ has λ₁/λ₂ = 1.91×, giving the null Gaussian two slowly-decorrelating directions. The null's λ₂ stays inflated because the OU process along PC2 decorrelates at the same rate as along PC1 — there is no barrier structure to create direction-dependent mixing times.

Phase 6 showed the 45D KDE has **zero barrier** along DA2 (< 0.01 kT). The canonical Langevin mixes freely along DA2, so its λ₂ decays fast. But the N(0, C₀) null's λ₂ also stays large (because the null's PC2 variance is nearly as large as PC1 variance). This makes it hard for observed λ₂ to exceed the null.

The isotropic null N(0, σ²_f · I_d) removes this anisotropy. All directions have equal variance σ²_f = tr(C₀)/d, so the null's eigenvalues are not dominated by a 2D subspace.

---

## 2. Isotropic Null Parameters

| Parameter | Value |
|---|---|
| Null distribution | N(0, σ²_f · I₄₅) |
| σ²_f = tr(C₀)/d | **9.171e-04** |
| σ_f | 3.028e-02 |
| d | 45 |
| N (KDE centers) | 2000 |
| h (KDE bandwidth) | 0.0091 |
| dt (Langevin step) | 0.1 |
| lag (steps) | 100,000 |
| phys_lag (ps) | 10,000 |
| n_eigvals stored | 10 |
| Total replicates | 100 (10 tasks × 10 reps) |
| Seed offset | 20000 + task_id × 10 |

---

## 3. Smoke Test (Local, CPU)

Ran 1 replicate at lag=100 steps (10 ps) to verify the script:

```
sigma_f_sq = tr(C0)/d = 9.171438e-04
sigma_f = 3.028438e-02
NULL: N(0, sigma_f^2 * I_45) — ISOTROPIC

Rep 1/1 (seed=20000):
  λ₁=0.001341, λ₂=0.001296, gap=1.03, time=16s
```

Saved npz contains: null_eigenvalues, sigma_f_sq, null_type='isotropic'. All correct.

The near-unit gap (1.03×) at short lag is expected for isotropic Gaussian — no direction is preferred.

---

## 4. Cluster Submission

| Field | Value |
|---|---|
| Job ID | **44661668** |
| Array | 0-9 (10 tasks) |
| Partition | catfish (gpu:l4:1) |
| Walltime | 4 hours |
| Memory | 32G |
| Status at submission | All 10 tasks RUNNING |

**Script:** `experiments/mc_dip_validation/run_alanine_null_iso.py`
**Sbatch:** `experiments/mc_dip_validation/cluster/run_alanine_null_iso.sh`

**Output files:**
- NPZ: `results/alanine_null_iso_lag100k_task{000-009}.npz`
- Logs: `results/ala_null_iso_44661668_{0-9}.out/.err`

**Expected completion:** ~10-20 min per task (10 reps × ~60-120s each). All tasks should finish within 30 min.

---

## 5. Analysis Plan (After Completion)

1. Load all 100 replicates from 10 task files.
2. Compute null λ₁ and λ₂ distributions (median, 2.5th, 97.5th percentiles).
3. Compare observed eigenvalues at τ=10,000 ps (λ₁=1.21e-2, λ₂=6.10e-4) against isotropic null.
4. Compute per-direction p-values: p_k = (1 + #{null_k ≥ obs_k}) / (1 + 100).
5. Apply Bonferroni (m=2) and BH corrections.

**Key question:** Does DA2 pass under the isotropic null?

---

## 6. Comparison with Matched-Covariance Null

| Null type | What it tests | DA2 result (Phase 1) |
|---|---|---|
| N(0, C₀) | Slower than matched-covariance Gaussian? | **FAIL** (p=0.980 at 2000ps) |
| N(0, σ²_f · I) | Slower than isotropic Gaussian? | **PENDING** |

The isotropic null is a weaker null (easier to beat), testing whether the observed λ₂ exceeds what an isotropic Gaussian would produce. If DA2 passes under this null but not under N(0, C₀), it means the φ mode's autocovariance is above the isotropic baseline but below what C₀'s anisotropy produces.
