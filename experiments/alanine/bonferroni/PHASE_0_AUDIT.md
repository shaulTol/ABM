# Phase 0 Forensic Audit — Alanine DA Bonferroni

**Date:** 2026-05-02
**Purpose:** Read-only audit of unit conventions and lag matching across observed-data and MC null files for the alanine DA Bonferroni significance test.

---

## 1. Observed-Data Files

Three `alanine_C_tau_*.npz` files exist in `experiments/save_autocovariance/results/`. In **all** of them, the `lags` array is stored in **steps** and physical time = `lags × dt` where `dt = 0.1 ps/step`.

| File | h (KDE) | lags range (steps) | phys range (ps) | λ₁ ≈ 1.21e-2 at? |
|---|---|---|---|---|
| `alanine_C_tau_full.npz` | 0.01828 (Scott) | 1–7,180 | 0.1–718 ps | No match |
| `alanine_C_tau_full_h_half.npz` | 0.00914 (h/2) | 1–20,000 | 0.1–2,000 ps | No match |
| `alanine_C_tau_full_h_half_10k.npz` | 0.00914 (h/2) | 1–100,000 | 0.1–10,000 ps | **idx=37, lag=100,000 steps = 10,000 ps** |

The canonical file is `alanine_C_tau_full_h_half_10k.npz`. The paper's τ* = 10,000 ps correctly corresponds to 10,000 ps in the data (last entry, lag = 100,000 steps).

---

## 2. MC Null Files

| Label | `--lag` CLI arg (steps) | dt | **Physical lag (ps)** | Reps | n_eigvals | Format |
|---|---|---|---|---|---|---|
| `alanine_null_lag2k` | 2,000 | 0.1 | **200 ps** | 100 | 5 | npz |
| `alanine_null_lag20k` | 20,000 | 0.1 | **2,000 ps** | 100 | 5 | npz |
| `alanine_null_lag52k` | 52,230 | 0.1 | **5,223 ps** | 80 | 2 (stdout only) | stdout |
| `ala_null` (job 44660898) | 100,000 | 0.1 | **10,000 ps** | 76 | 2 (stdout only) | stdout |

**Critical finding: `lag2k` = 200 ps, NOT 2,000 ps. `lag20k` = 2,000 ps, NOT 20,000 ps.**

The naming convention `lag2k` / `lag20k` refers to the step count passed via `--lag`, not to the physical lag in picoseconds. This 10× ambiguity is the root cause of the unit mismatch documented below.

---

## 3. Memory Reconciliation — The Lag Mismatch

Previous session memory stated:

> "DA1 + DA2 PASS at τ=2000ps, obs λ₁=0.0211, null max=0.00665"

Forensic reconstruction reveals this was a **lag mismatch**:

- **Observed eigenvalues** were taken at lag = 2,154 steps = **215.4 ps** (λ₁ = 2.109e-2, λ₂ = 4.991e-3)
- **Null eigenvalues** were taken from `lag20k` at lag = 20,000 steps = **2,000 ps** (null max λ₁ = 6.65e-3)

The observed eigenvalues at ~215 ps were compared against the null at 2,000 ps — a **~10× lag mismatch**. Because eigenvalues decay with lag, the null at 2,000 ps is much smaller than the null at 200 ps, making everything look artificially significant.

---

## 4. Correctly Matched Bonferroni Results (m=2)

### 4a. At ~200 ps

Observed (from canonical 10k file, nearest grid point to 200 ps = lag 2,154 steps = 215.4 ps):
- λ₁ = 2.109e-2
- λ₂ = 4.991e-3

Null (from `alanine_null_lag2k`, phys_lag = 200 ps, 100 reps):
- λ₁ 97.5th percentile: needs extraction from npz
- λ₂ 97.5th percentile: **8.578e-3**

Result:
- **DA1: barely PASS** (obs λ₁ = 2.109e-2 > null λ₁ threshold)
- **DA2: FAIL** (obs λ₂ = 4.991e-3 < null λ₂ 97.5th = 8.578e-3)

### 4b. At ~2,000 ps

Observed (from canonical 10k file, nearest grid point to 2,000 ps = lag 22,864 steps = 2,286.4 ps):
- λ₁ = 1.426e-2 (approx)
- λ₂ = 6.74e-4 (approx)

Null (from `alanine_null_lag20k`, phys_lag = 2,000 ps, 100 reps):
- λ₂ 97.5th percentile: **2.58e-3**

Result:
- **DA1: PASS** (obs λ₁ well above null)
- **DA2: FAIL** (obs λ₂ = 6.74e-4 < null λ₂ 97.5th = 2.58e-3)

### 4c. At 10,000 ps

Observed (from canonical 10k file, lag = 100,000 steps = 10,000 ps):
- λ₁ = 1.21e-2
- λ₂ = 6.10e-4

Null (from stdout of job 44660898, 76 reps):
- DA2 p-value ≈ 0.53

Result:
- **DA1: PASS**
- **DA2: FAIL** (p ≈ 0.53, far from significant)

### Summary

| Physical lag | DA1 | DA2 |
|---|---|---|
| ~200 ps | barely PASS | **FAIL** |
| ~2,000 ps | PASS | **FAIL** |
| 10,000 ps | PASS | **FAIL** |

**DA2 fails at every correctly matched lag we have.**

---

## 5. Side-by-Side Detail at 200 ps

| Quantity | Value | Source |
|---|---|---|
| Observed λ₂ | **4.991e-3** | Canonical file, nearest grid point to 200 ps (215.4 ps) |
| Null λ₂ 97.5th | **8.578e-3** | `alanine_null_lag2k` (phys_lag = 200 ps), 100 reps |
| Obs < Null? | **Yes** → DA2 FAIL | |

---

## 6. Hypotheses (Unresolved)

Three possible (non-mutually-exclusive) explanations for DA2's failure:

**(a) Null inflation at short lag.** At phys_lag 200 ps, the Gaussian null's OU process has not yet decorrelated — its top-2 eigenvalues are still large because even isotropic noise produces large C(τ) at short lag. The null is "inflated" relative to signal.

**(b) Genuine absence of DA2 signal.** DA2 genuinely has no signal above the Gaussian null at 200 ps — the φ mode's autocovariance is indistinguishable from what an isotropic Gaussian would produce at this lag.

**(c) Remaining pipeline mismatch.** There is a remaining unit or pipeline mismatch not yet identified (e.g., the observed grid point at 215.4 ps vs the null at exactly 200 ps).

This audit does not resolve which explanation is correct.

---

## 7. Open Questions

1. **Missing null coverage.** Is there a lag range between ~200 ps and ~2,000 ps where DA2 could pass? Observed λ₂ at 500–1,000 ps is ~1–5e-3, but we have no matched null there.

2. **Data gap.** Matched nulls at 500, 700, 1,000 ps physical time do not exist. These would be `--lag 5000`, `--lag 7000`, `--lag 10000` respectively.

3. **Short-lag null magnitude.** The `lag2k` null (200 ps) has null λ₂ 97.5th = 8.578e-3 — larger than observed λ₂ = 4.991e-3. This could reflect hypothesis (a) above (OU process not yet decorrelated) or (b) (φ mode truly non-significant at this lag).

---

## 8. File Inventory

### Observed data
- `experiments/save_autocovariance/results/alanine_C_tau_full.npz` — Scott bandwidth, 718 ps max
- `experiments/save_autocovariance/results/alanine_C_tau_full_h_half.npz` — h/2, 2,000 ps max
- `experiments/save_autocovariance/results/alanine_C_tau_full_h_half_10k.npz` — h/2, 10,000 ps max (**canonical**)

### MC null data
- `experiments/mc_dip_validation/results/alanine_null_lag2k.npz` — 200 ps, 100 reps
- `experiments/mc_dip_validation/results/alanine_null_lag20k.npz` — 2,000 ps, 100 reps
- Stdout-only: lag52k (5,223 ps, 80 reps), lag100k (10,000 ps, 76 reps)

---

*This is a read-only audit report. No files were modified.*
