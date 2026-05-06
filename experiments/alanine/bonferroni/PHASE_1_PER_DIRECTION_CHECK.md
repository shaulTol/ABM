# Phase 1: Per-Direction Significance — Bonferroni vs BH vs Uncorrected

**Date:** 2026-05-02
**Purpose:** Determine whether DA2's failure is due to multiplicity correction (Bonferroni too conservative) or absence of detectable signal.

---

## Method

For each (phys_lag, direction) pair, we compute:

1. **Observed λ_k** from the canonical file `alanine_C_tau_full_h_half_10k.npz` at the nearest grid point.
2. **Per-direction p-value**: empirical p = (1 + #{null λ_k ≥ obs λ_k}) / (1 + n_reps), comparing obs λ_k against the null distribution of the **same index** k.
3. **Conservative p-value**: obs λ_k against null λ₁ (largest null eigenvalue). Tests whether DA_k exceeds the noisiest direction the null produces.
4. **Bonferroni m=3**: per-direction p × 3, capped at 1.
5. **BH q-value**: Benjamini-Hochberg across 3 directions at each phys_lag.

**Null files** (concatenated from 10 SLURM array tasks each, 10 reps/task = 100 total):
- `alanine_null_lag2k_task*.npz`: `--lag 2000` steps → phys_lag = **200 ps**, shape (100, 5)
- `alanine_null_lag20k_task*.npz`: `--lag 20000` steps → phys_lag = **2,000 ps**, shape (100, 5)

Directions tested: DA1, DA2, DA3 (m=3). Significance threshold: α = 0.05.

---

## Results at phys_lag = 200 ps

Observed grid point: idx=24, lag (steps) = 2,154, **phys_lag = 215.4 ps**.

⚠ **Grid mismatch**: observed at 215.4 ps vs null at 200.0 ps (Δ = 15.4 ps). Since eigenvalues decay with lag, the observed eigenvalues at 215.4 ps are slightly *smaller* than they would be at exactly 200 ps. This makes the test slightly conservative (biased against detection).

| direction | obs λ | grid lag (ps) | per-dir p | cons p (vs null λ₁) | Bonf m=3 p | BH q | per-dir PASS? | BH PASS? |
|---|---|---|---|---|---|---|---|---|
| DA1 | 2.109e-02 | 215.4 | 0.0099 | 0.0099 | 0.0297 | 0.0149 | PASS | PASS |
| DA2 | 4.991e-03 | 215.4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **FAIL** | **FAIL** |
| DA3 | 3.727e-04 | 215.4 | 0.0099 | 1.0000 | 0.0297 | 0.0149 | PASS | PASS |

Null distribution at phys_lag = 200 ps (100 reps):

| null statistic | λ₁ | λ₂ | λ₃ |
|---|---|---|---|
| mean | 1.819e-02 | 7.644e-03 | 1.723e-04 |
| 95th pctl | 1.980e-02 | 8.463e-03 | 2.332e-04 |
| 97.5th pctl | 2.009e-02 | 8.578e-03 | 2.381e-04 |
| max | 2.047e-02 | 9.085e-03 | 2.658e-04 |

---

## Results at phys_lag = 2,000 ps

Observed grid point: idx=32, lag (steps) = 22,855, **phys_lag = 2,285.5 ps**.

⚠ **Grid mismatch**: observed at 2,285.5 ps vs null at 2,000.0 ps (Δ = 285.5 ps). Same conservative bias as above — observed eigenvalues at the later lag are smaller.

| direction | obs λ | grid lag (ps) | per-dir p | cons p (vs null λ₁) | Bonf m=3 p | BH q | per-dir PASS? | BH PASS? |
|---|---|---|---|---|---|---|---|---|
| DA1 | 1.782e-02 | 2285.5 | 0.0099 | 0.0099 | 0.0297 | 0.0297 | PASS | PASS |
| DA2 | 6.740e-04 | 2285.5 | 0.9802 | 1.0000 | 1.0000 | 0.9802 | **FAIL** | **FAIL** |
| DA3 | 7.810e-05 | 2285.5 | 0.4158 | 1.0000 | 1.0000 | 0.6238 | FAIL | FAIL |

Null distribution at phys_lag = 2,000 ps (100 reps):

| null statistic | λ₁ | λ₂ | λ₃ |
|---|---|---|---|
| mean | 4.868e-03 | 1.645e-03 | 7.357e-05 |
| 95th pctl | 6.203e-03 | 2.390e-03 | 1.038e-04 |
| 97.5th pctl | 6.555e-03 | 2.578e-03 | 1.172e-04 |
| max | 6.652e-03 | 3.013e-03 | 1.432e-04 |

---

## Interpretation

**DA2 fails even the uncorrected per-direction test.** This is not a multiplicity-correction problem.

At phys_lag = 200 ps, DA2's observed λ₂ = 4.991e-3 has per-direction p = 1.000 — every single null replicate produced a λ₂ ≥ obs λ₂. The null mean λ₂ = 7.644e-3 is *53% larger* than the observed value. DA2 is not borderline; it is squarely inside the null distribution. BH cannot salvage a p-value of 1.000.

At phys_lag = 2,000 ps, DA2's observed λ₂ = 6.740e-4 has per-direction p = 0.980 — only 2 of 100 null replicates fell below it. Again, deep inside the null. BH q = 0.980.

**The answer is (b): DA2 has no detectable signal above the Gaussian null at either lag.** The φ-direction's autocovariance, while real in the sense that DA2 correlates with φ (|r| = 0.87), does not produce an eigenvalue that exceeds what an isotropic Gaussian of matching (N, d, σ²_f) would produce by chance. The multiplicity correction is irrelevant — even testing DA2 in complete isolation, with no correction at all, it fails decisively (p ≥ 0.98).

**Surprising finding: DA3 passes at 200 ps.** At phys_lag = 200 ps, DA3's observed λ₃ = 3.727e-4 has per-direction p = 0.0099 (obs > all 100 null λ₃ replicates). However, DA3 fails the conservative test (vs null λ₁), meaning it is above the *third*-eigenvalue noise floor but not above the *first*-eigenvalue noise floor. At phys_lag = 2,000 ps, DA3 loses significance entirely (p = 0.42).

**DA1 is robust.** It passes all tests — per-direction, conservative (vs null λ₁), Bonferroni, and BH — at both lags. DA1 is the only direction with a genuine spike above the full null distribution.

---

## Summary Table

| phys_lag (ps) | DA1 per-dir | DA1 BH | DA2 per-dir | DA2 BH | DA3 per-dir | DA3 BH |
|---|---|---|---|---|---|---|
| 200 | **PASS** (0.010) | **PASS** (0.015) | FAIL (1.000) | FAIL (1.000) | **PASS** (0.010) | **PASS** (0.015) |
| 2,000 | **PASS** (0.010) | **PASS** (0.030) | FAIL (0.980) | FAIL (0.980) | FAIL (0.416) | FAIL (0.624) |

**Conclusion:** DA2's failure is not due to Bonferroni conservatism. DA2 does not clear even the per-direction noise floor (p ≥ 0.98 at both lags). The framework detects exactly one spike (DA1 = ψ) as significant; the second mode (DA2 = φ) is indistinguishable from the Gaussian null at the sample sizes and dimensions of this dataset (N = 2,000, d = 45).
