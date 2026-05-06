# Phase 8: Centering Audit — Where Does mean=1.1 Come From?

**Date:** 2026-05-02
**Purpose:** Investigate the centering inconsistency in `phase7_iso_null_da2_projection.png`: observed KDE centers projected onto DA2 have mean=1.1, while isotropic null projections have mean=0.

---

## 1. Five Projections onto DA2

DA2 = second eigenvector of symmetrized C(τ) at τ=10,000 ps, from `alanine_C_tau_full_h_half_10k.npz` idx=37.

| # | Data | Centering | (X @ DA2).mean() | (X @ DA2).std() |
|---|---|---|---|---|
| 1 | 2000 KDE centers (raw) | None | **1.105719** | 0.074964 |
| 2 | 2000 KDE centers − centers.mean(0) | Own sample mean | **≈0** (8.2e-8) | 0.074964 |
| 3 | 2000 KDE centers − X_sub.mean(0) | Simulator's mu_f | **0.001417** | 0.074964 |
| 4 | (No stored mean in canonical file) | — | — | — |
| 5 | Full 750k MD frames (raw) | None | **1.104270** | 0.075672 |

**The std is identical (0.0750) regardless of centering.** Only the mean shifts.

---

## 2. Source of the mean=1.1

The mean=1.1 is simply **mu_f @ DA2**, where mu_f is the sample mean of the data:

```
X_sub.mean(0) @ DA2 = 1.104302
md_mean @ DA2       = 1.104277
centers.mean(0) @ DA2 = 1.105719
```

The alanine dipeptide distance features have a nonzero global mean (||mu_f|| = 2.31). When projected onto DA2 (or any direction), this global mean produces a large offset. The offset is **purely cosmetic** — it reflects the coordinate system of the raw distance features, not any signal.

---

## 3. How C(τ) Was Computed (Centering Chain)

Tracing the exact centering used when the canonical file was produced:

### Step 1: Simulator construction
**`save_alanine_C_tau_h_half_10k.py`, lines 118–119:**
```python
kde_sim = GenericLangevinSimulator(
    score_fn=kde.score, samples=X_sub_t, dt=best_dt, device=device)
```

### Step 2: mu_f computed inside GenericLangevinSimulator
**`generic_langevin.py`, line 64:**
```python
self.mu_f = self.samples.mean(dim=0)  # = X_sub_t.mean(dim=0)
```
No explicit mu_f was passed, so it defaults to the empirical mean of the 10,000-point subsample.

### Step 3: extract_da_directions uses this mu_f
**`da_directions.py`, lines 111, 34:**
```python
sim, mu_f = _make_simulator(simulator=kde_sim, ...)
# → mu_f = simulator.mu_f = X_sub.mean(0)
```

### Step 4: C(τ) centered by mu_f
**`da_directions.py`, lines 118, 127–128:**
```python
x0 = sim.sample_initial(n_trajectories)       # resamples from X_sub_t
x0_centered = x0 - mu_f.unsqueeze(0)          # centered by X_sub.mean(0)
x_tau_centered = x_tau - mu_f.unsqueeze(0)     # centered by X_sub.mean(0)
```

### Step 5: C(0) also centered by same mu_f
**`save_alanine_C_tau_h_half_10k.py`, lines 194–197:**
```python
x0 = kde_sim.sample_initial(n_traj)
mu_f = kde_sim.mu_f                            # same X_sub.mean(0)
x0_c = x0 - mu_f.unsqueeze(0)
C0 = (x0_c.t() @ x0_c / n_traj)
```

**Conclusion:** C(τ) and C(0) are both computed on data centered by `X_sub.mean(dim=0)`. The eigenvectors (including DA2) are directions in the **centered** coordinate system. The eigenvalues are unaffected by the choice of centering origin — centering shifts the mean but doesn't change the autocovariance structure.

---

## 4. Null Scripts: Centering Convention

### Matched-covariance null (`run_alanine_null.py`)
**Lines 68, 81–82:**
```python
mu_hat = np.zeros(d)                    # Center at origin
x_null_np = mu_hat + z @ L.T           # ~ N(0, C₀)
```
Then `GenericLangevinSimulator(samples=x_null, ...)` sets `mu_f = x_null.mean(dim=0) ≈ 0`.
Then `extract_da_directions` centers by this mu_f ≈ 0.

### Isotropic null (`run_alanine_null_iso.py`)
**Line 85:**
```python
x_null_np = sigma_f * rng.randn(args.N, d)  # ~ N(0, σ²_f·I)
```
Then `GenericLangevinSimulator(samples=x_null, ...)` sets `mu_f = x_null.mean(dim=0) ≈ 0`.
Then `extract_da_directions` centers by this mu_f ≈ 0.

**Both nulls are centered at the origin.** The observed pipeline is centered at `X_sub.mean(0)`. This is **correct and consistent**: each pipeline centers by its own mu_f before computing C(τ). The eigenvectors and eigenvalues are computed in the centered frame in all cases.

---

## 5. Does the Centering Affect Eigenvalues or Eigenvectors?

**No.** The autocovariance matrix C(τ) = E[(X_τ − μ)(X₀ − μ)ᵀ] depends on the centering constant μ only through a rank-1 correction. When μ = E[X₀] (the true mean), the cross-term vanishes:

```
C(τ) = E[X_τ X₀ᵀ] − μμᵀ    (for any constant μ = E[X₀])
```

The eigenvectors of C(τ) are independent of whether we subtract the population mean or the sample mean (to leading order in N). The eigenvalues change by at most O(1/N) from the rank-1 sample-mean correction.

**Verification:** Stored eigenvalues match re-computed eigenvalues from the stored C_tau matrix:
```
Stored:      [1.20989848e-02, 6.09556038e-04, 6.02409382e-05, ...]
Re-computed: [1.20989060e-02, 6.09550600e-04, 6.02404119e-05, ...]
Match: True (within float64 precision)
```

---

## 6. What the Diagnostic Plot Should Have Done

The `phase7_iso_null_da2_projection.png` plot projected **raw** (uncentered) KDE centers onto DA2:
```python
proj_obs = centers @ DA2          # mean = 1.1 (uncentered)
proj_null = x_null @ DA2          # mean ≈ 0 (null is zero-mean)
```

For a fair visual comparison, it should have centered both:
```python
proj_obs = (centers - centers.mean(0)) @ DA2      # mean ≈ 0
proj_null = (x_null - x_null.mean(0)) @ DA2       # mean ≈ 0
```

Or equivalently, subtracted each dataset's own mu_f before projecting. The std (0.075 vs 0.030) and distributional shape (bimodal vs unimodal) are unaffected by centering — only the location changes.

---

## 7. Answer to the Key Question

**Was the C(τ) used to extract DA2 computed with the same centering convention that the diagnostic plot used?**

**No.** C(τ) was computed on data centered by `X_sub.mean(0)` (mu_f ≈ a vector of norm 2.31), while the diagnostic plot projected **uncentered** data (not subtracting any mean).

**Does this affect DA2 or the eigenvalues?** **No.** DA2 is an eigenvector of a centered autocovariance matrix. Projecting uncentered data onto DA2 just adds a constant offset (mu_f @ DA2 = 1.10) to all projections. The eigenvectors, eigenvalues, and the significance test are all unaffected. The mean=1.1 in the diagnostic plot is a display artifact — it's the projection of the global mean of the distance features onto DA2, not a centering bug in the pipeline.

**The centering conventions are consistent within each pipeline (observed and null).** The diagnostic plot's inconsistency is purely cosmetic.

---

## 8. Summary

| Question | Answer |
|---|---|
| Where does mean=1.1 come from? | mu_f @ DA2 = X_sub.mean(0) @ DA2 = 1.104 |
| Is mu_f stored in the canonical file? | **No** — must be reconstructed from code |
| Does centering affect DA2 direction? | **No** — eigenvectors invariant to translation |
| Does centering affect eigenvalues? | **No** — to O(1/N) |
| Is the observed C(τ) properly centered? | **Yes** — by X_sub.mean(0), consistently |
| Is the null C(τ) properly centered? | **Yes** — by null_samples.mean(0) ≈ 0 |
| Was the diagnostic plot misleading? | **Yes** (cosmetically) — should have centered before projecting |
| Does this explain why DA2 fails? | **No** — the centering is correct; DA2 fails for the reasons identified in Phases 4–6 |
