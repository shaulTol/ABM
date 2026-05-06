# Diffusion Maps Baseline on Alanine Dipeptide

## Bandwidth selection

Log-log criterion on N=2000 subsample (seed=42): ε* = 0.0655.
Also tested ε*/2 = 0.0328 and 2ε* = 0.131.

## Key result: ordering matches DA/TICA

At all three bandwidths, ψ₂ ↔ ψ (slow dihedral) and ψ₃ ↔ φ (fast dihedral).
This is the same ordering as DA1 ↔ ψ, DA2 ↔ φ, and TICA1 ↔ ψ, TICA2 ↔ φ.

**Diffusion maps recovers the correct slow/fast ordering.** This is expected:
with α=1, DM approximates the Fokker-Planck generator, which should recover
the same metastable modes as TICA and DA.

## Correlations (mean ± SD across 5 seeds)

| | ε*/2 (0.033) | ε* (0.066) | 2ε* (0.131) |
|---|---|---|---|
| ψ₂ \|r(ψ)\| | **0.676 ± 0.023** | 0.615 ± 0.023 | 0.581 ± 0.023 |
| ψ₂ \|r(φ)\| | 0.420 ± 0.015 | 0.523 ± 0.012 | 0.565 ± 0.010 |
| ψ₃ \|r(φ)\| | **0.721 ± 0.019** | 0.659 ± 0.020 | 0.630 ± 0.019 |
| ψ₃ \|r(ψ)\| | 0.262 ± 0.014 | 0.348 ± 0.014 | 0.386 ± 0.014 |

Smaller ε (less smoothing) gives cleaner separation between the two dihedrals.
At ε*/2: ψ₂ is predominantly ψ (0.676 vs 0.420) and ψ₃ is predominantly φ
(0.721 vs 0.262).

## Comparison with DA and TICA

| Method | Slow mode \|r(ψ)\| | Fast mode \|r(φ)\| |
|---|---|---|
| DA1 (h/2, τ=10000ps) | **0.672 ± 0.001** | 0.351 |
| TICA1 (τ_lag=10) | ~0.67 | ~0.35 |
| DM ψ₂ (ε*/2) | 0.676 ± 0.023 | 0.420 |
| DA2 (h/2, τ=10000ps) | 0.069 | **0.873 ± 0.000** |
| TICA2 (τ_lag=10) | ~0.07 | ~0.84 |
| DM ψ₃ (ε*/2) | 0.262 | 0.721 |

DM ψ₂ matches DA1/TICA1 on the slow mode (|r(ψ)| ≈ 0.67) but has more
φ leakage (0.42 vs 0.35). DM ψ₃ captures the fast mode less purely than
DA2/TICA2 (|r(φ)| = 0.72 vs 0.87) with substantial ψ leakage (0.26 vs 0.07).

The higher cross-contamination in DM likely reflects the fact that DM works
with a fixed kernel bandwidth while DA/TICA use lag-dependent autocovariance
which naturally separates timescales.

## Robustness

- Ordering ψ₂↔ψ, ψ₃↔φ is correct at all 3 bandwidths and all 5 seeds
- Standard deviations across seeds are small (0.01-0.02)
- Smaller ε gives purer separation; larger ε mixes the dihedrals more

## Correlations computed on same-sample

All correlations computed on the same 2000 subsampled frames used for the
eigenvector computation (no Nyström out-of-sample extension).
