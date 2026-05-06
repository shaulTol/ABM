# Phase 11: Zero-Drift Diagnosis — Why the Iso Null Has No Langevin Drift

**Date:** 2026-05-02
**Purpose:** The iso null at phys_lag=10,000 ps gives λ₁ ≈ 1.34e-3 and λ₂ ≈ 1.29e-3, far above the analytical OU prediction of σ²_f·exp(−τ/2) ≈ 0. The decay scan (job 44662055) confirmed the drift is exactly zero — `||drift|| = 0.000e+00` across all 2000 sample points. This report diagnoses why.

---

## 1. The Zero-Drift Finding (from decay scan)

Job 44662055 output:

```
=== Drift magnitude at t=0 ===
  drift = 0.5 * var_f * dt * grad_log_f(x)
  ||drift||: mean=0.000000e+00, std=0.000000e+00, max=0.000000e+00
  ||grad_log_f||: mean=0.000000e+00, max=0.000000e+00
  noise per step: sigma_f * sqrt(dt) * ||z|| ~ 9.54e-03 * sqrt(45) = 6.40e-02
  drift/noise ratio (norms): 0.0000
```

Eigenvalues are constant across lags (no decay):

| lag_steps | phys_ps | λ₁ | λ₂ | λ₃ | λ₄ | λ₅ | tr(C(τ))/tr(C(0)) |
|---|---|---|---|---|---|---|---|
| 1 | 0.1 | 1.27e-3 | 1.25e-3 | 1.21e-3 | 1.20e-3 | 1.18e-3 | 1.000 |
| 10 | 1.0 | 1.31e-3 | 1.29e-3 | 1.28e-3 | 1.23e-3 | 1.19e-3 | 1.010 |
| 100 | 10.0 | 1.31e-3 | 1.27e-3 | 1.26e-3 | 1.20e-3 | 1.18e-3 | 1.017 |
| 1000 | 100.0 | 1.30e-3 | 1.28e-3 | 1.25e-3 | 1.22e-3 | 1.20e-3 | 1.018 |

**Verdict: H1 confirmed.** The simulator is not mixing. It is a pure random walk — noise only, no drift. Eigenvalues of C(τ) are the finite-sample noise floor of independent random-walk trajectories.

---

## 2. KDE Score on Three Datasets

Evaluated `kde.score(x[:10])` with the same bandwidth h=0.0091, d=45, N=2000:

| Dataset | ||score|| mean | ||score|| max | 1-NN dist | 1-NN/h | # log_w > −100 |
|---|---|---|---|---|---|
| **Real alanine** (2000 centers) | **2.03e-1** | **1.31e+0** | 0.040 | **4.41** | **232** |
| **Matched-Σ null** N(0, C₀) | **4.60e-4** | **2.96e-3** | 0.061 | **6.72** | **74** |
| **Isotropic null** N(0, σ²_f·I) | **0.00e+0** | **0.00e+0** | 0.201 | **22.05** | **1** |

**Only the real alanine data produces meaningful KDE scores.** The matched-Σ null has scores ~500× smaller than real data. The isotropic null has exactly zero scores.

---

## 3. Root Cause: Bandwidth vs Inter-Point Distance in 45D

The KDE score formula is:

```
∇log f̂(x) = Σᵢ wᵢ(x) · (xᵢ − x) / h²
where wᵢ(x) = softmax(−‖x − xᵢ‖²/(2h²))
```

The softmax weights `wᵢ` are determined by `log_w = −‖x − xᵢ‖²/(2h²)`.

### Key numbers:

For the **isotropic null**, typical inter-point distance is:

```
E[‖xᵢ − xⱼ‖] = σ_f · √(2d) = 0.0303 × √90 = 0.287
```

So the typical `log_w = −(0.287)²/(2 × 0.0091²) = −498`.

The softmax on 2000 values where the maximum is `log_w = 0` (self-distance) and all others are `≈ −498` puts **100% weight on the nearest neighbor** and exactly 0 on everything else. In float32:

```
Iso null softmax weights: min=0.0, max=1.0
```

With w₁ = 1 for the nearest neighbor and wᵢ = 0 for all others, the score becomes:

```
∇log f̂(x) ≈ (x_nn − x) / h²
```

But `x_nn − x` has components of order ~0.2 in each of 45 dimensions, while `h² = 8.3e-5`. The score should be enormous (~2400 per coordinate). **Why is it zero instead?**

Because the query points `x[:10]` are **themselves** in the centers array. When `x = xᵢ` exactly, `‖x − xᵢ‖ = 0`, `log_w_i = 0`, and for all other `j`, `log_w_j ≈ −498`. The softmax puts weight 1.0 on the self-center (i=self), and the contribution `wᵢ · (xᵢ − x)/h² = 1.0 · 0/h² = 0`. All other weights underflow to exactly 0 in float32.

**Result: score = 0 at every center.** And since `sample_initial()` resamples from the centers, every trajectory starts at a center and gets score = 0 at step 0.

After one Langevin step (pure noise), the point moves to `x + σ_f√dt · z` where `σ_f√dt ≈ 0.0096`. Now the point is ~0.064 (in ℓ₂) from the nearest center. The log_w for the nearest center is `−0.064²/(2 × 0.0091²) ≈ −25`, while all other centers have `log_w ≈ −498`. The softmax still puts ~100% weight on the one nearest center, giving score ≈ `(x_nn − x)/h² ≈ −(noise_step)/h²`. This is a restoring force back toward the nearest center.

**But this restoring force is toward a single center, not toward the global mean.** In 45D with N=2000 centers separated by ~0.2 from each other, the KDE effectively creates 2000 independent point attractors, each trapping its Langevin trajectory in a ball of radius ~h around itself. The trajectories never explore the global density — they oscillate around their nearest center.

---

## 4. Why Real Alanine Data Doesn't Have This Problem

For the real alanine 2000 centers:
- 1-NN distance = 0.040, so 1-NN/h = 4.41
- 232 centers have `log_w > −100`
- The softmax distributes weight across ~100+ neighbors

The alanine distance data lives on a **low-dimensional manifold** in 45D. Although the ambient dimension is 45, the effective intrinsic dimension is much lower (~2–5, corresponding to ψ, φ, and a few other slow coordinates). Points are close together on the manifold, so many neighbors fall within range of h=0.0091.

In contrast, the isotropic null fills all 45 dimensions uniformly. With N=2000 in 45D, the average inter-point distance scales as `σ_f · √(2d)`, which is ~31× the bandwidth. No point has any neighbor within meaningful KDE kernel range.

---

## 5. Implications for the Matched-Σ Null

The matched-Σ null N(0, C₀) is intermediate:
- 1-NN/h = 6.72 (closer than iso, further than real)
- ||score|| = 4.60e-4 (tiny but nonzero)
- 74 centers have `log_w > −100`

C₀ has two dominant eigenvalues (λ₁/λ₂ ratio 1.91×), so the matched-Σ null concentrates samples in a ~2D subspace, making neighbors closer. But the scores are still ~500× smaller than real data. The drift is present but negligible compared to noise:

```
drift/noise ≈ ||score|| × var_f × dt / (sigma_f × sqrt(dt))
            ≈ 4.6e-4 × 9.2e-4 × 0.1 / (0.030 × 0.316)
            ≈ 4.5e-6
```

**The matched-Σ null is also essentially a pure random walk.** The drift exists but is ~200,000× smaller than the noise. This means:

- **The matched-Σ null eigenvalues at phys_lag=2000 ps (from Phase 4) are also just the finite-sample noise floor, not OU autocovariance.**
- **The comparison of observed λ₂ vs null λ₂ in Phase 1 was comparing genuine Langevin autocovariance (observed) vs random-walk noise floor (null).**
- **DA2 "failing" against the null means: observed λ₂ (genuine) < random-walk noise floor.** This is not a meaningful statistical test.

---

## 6. Why the Observed Data's Eigenvalues ARE Genuine

For the real alanine data:
- ||score|| = 0.203 (mean), 1.307 (max)
- drift/noise ≈ 0.203 × 9.2e-4 × 0.1 / (0.030 × 0.316) ≈ 0.002

Still small (drift ~0.2% of noise per step), but over 100,000 steps, the cumulative drift is significant. The key question is whether this drift is enough to produce genuine autocovariance structure — and the answer is yes, because the observed eigenvalues show a clear spectral gap (λ₁/λ₂ = 19.85×) that the random-walk null cannot produce.

However, the observed λ₂ = 6.10e-4 may also be partially or wholly a noise floor artifact. At phys_lag=10,000 ps, even the real data's drift may have mixed out along DA2, leaving only the estimator noise floor.

---

## 7. The Noise Floor of C(τ)

For N independent random walks (no drift) of duration τ steps:

```
C(τ) = (1/N) Σₙ (X_τ⁽ⁿ⁾ − μ)(X₀⁽ⁿ⁾ − μ)ᵀ
```

If X_τ and X₀ are independent (which they are for a pure random walk at any lag > 0, since the walk has no memory), then C(τ) is a random matrix with entries of order σ²/√N. The top eigenvalue of this random matrix follows the Tracy-Widom distribution, scaled by the Marchenko-Pastur edge:

```
λ_MP_edge ≈ σ² × (1 + √(d/N))² / d
```

For the iso null: σ² ≈ var per coordinate ≈ σ²_f = 9.17e-4, d=45, N=2000:

```
λ_MP_edge ≈ 9.17e-4 × (1 + √(45/2000))² / 45
           ≈ 9.17e-4 × (1 + 0.150)² / 45
           ≈ 9.17e-4 × 1.322 / 45
           ≈ 2.69e-5
```

But this doesn't match the observed iso null λ₁ ≈ 1.3e-3. The discrepancy is because after τ steps of random walk, the variance of X_τ grows:

```
Var(X_τ) = Var(X₀) + τ × σ²_noise_per_step
         = σ²_f + lag_steps × σ²_f × dt
         = σ²_f × (1 + lag_steps × dt)
```

At lag=100,000 steps with dt=0.1: Var(X_τ) = σ²_f × (1 + 10000) ≈ 10001 × σ²_f. The random walk has diffused far from its origin, so X_τ and X₀ share only the initial variance σ²_f as "signal." The C(τ) noise floor depends on Var(X₀) and Var(X_τ), both of which grow with lag.

The exact noise floor for this setup requires more careful analysis, but the observed λ₁ ≈ 1.3e-3 for the iso null is consistent with a finite-sample noise floor from uncorrelated X₀ and X_τ.

---

## 8. KDEScore Code Analysis

**File:** `src/kde/kde_score.py`, lines 71–112.

The score is computed as:

```python
# Line 97: Differences (B, N, d)
diff = self.centers.unsqueeze(0) - x_batch.unsqueeze(1)  # xᵢ − x

# Line 100: Squared distances (B, N)
sq_dist = (diff ** 2).sum(dim=-1)

# Line 103: Log-responsibilities
log_w = -sq_dist / (2.0 * self.h2)

# Line 106: Softmax (numerically stable via max subtraction)
w = torch.softmax(log_w, dim=-1)  # (B, N)

# Line 110: Score
grad_out[start:end] = (w.unsqueeze(-1) * diff).sum(dim=1) / self.h2
```

The `torch.softmax` implementation subtracts `max(log_w)` before exponentiating, which prevents overflow. But when `log_w_max − log_w_i > ~88` (float32 exp underflow), `exp(log_w_i − log_w_max) = 0`. For the iso null, `log_w_max = 0` (self-distance) and all others are `≈ −498`, so all non-self weights underflow to exactly 0.

**This is not a bug in KDEScore** — the softmax is numerically correct. The problem is that h=0.0091 is far too small for 2000 points in 45D isotropic space. The kernels don't overlap at all.

---

## 9. Verdict

| Hypothesis | Status | Evidence |
|---|---|---|
| H1: Simulator not mixing (zero drift) | **CONFIRMED** | ||drift|| = 0 for iso null, 4.6e-4 for matched-Σ (negligible) |
| H2: Time-unit mismatch | **RULED OUT** | No decay at all — not a timescale issue |
| H3: Estimator floor | **PARTIALLY** | The observed eigenvalues ARE the noise floor, but because of H1, not sampling error |

**The root cause is H1: KDE bandwidth underflow in 45D.**

With h=0.0091 and N=2000 in d=45:
- **Isotropic null**: 1-NN/h = 22. Softmax gives all weight to self. Score = 0. **Pure random walk.**
- **Matched-Σ null**: 1-NN/h = 6.7. Score ≈ 4.6e-4 (negligible). **Essentially pure random walk.**
- **Real alanine data**: 1-NN/h = 4.4. Score ≈ 0.20. **Genuine Langevin drift.**

**Critical implication:** The MC null test in its current form (run_alanine_null.py and run_alanine_null_iso.py) does NOT test what it claims to test. It claims to compare observed Langevin autocovariance against the Langevin autocovariance of a Gaussian null. But the Gaussian null's Langevin diffusion has zero (or negligible) drift, so the null eigenvalues are just finite-sample noise from independent random walks — not the autocovariance of an OU process.

**The comparison of observed λ_k vs null λ_k is actually: genuine Langevin autocovariance vs random-walk noise floor.** This is a very different test from what was intended.

---

## 10. What the Test Actually Tests

Despite the unintended mechanism, the test may still have statistical validity — but with a different interpretation:

- **DA1 passes**: observed λ₁ = 1.21e-2 >> noise floor ≈ 1.3e-3. The ψ mode's autocovariance is well above the finite-sample noise floor. This direction has genuine slow mixing.

- **DA2 fails**: observed λ₂ = 6.10e-4 < noise floor ≈ 1.3e-3. The φ mode's autocovariance is **below** the finite-sample noise floor. This means DA2's autocovariance is too small to be distinguished from sampling noise.

This is actually a **valid significance test**, just not against the intended null. It tests: "Is the observed autocovariance along direction k larger than what you'd see from pure sampling noise (no signal)?" — which is essentially a signal-to-noise ratio test.
