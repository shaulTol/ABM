# Alanine Dipeptide — KDE-DA Experiment (Paper Version)

All DA results in the paper use **KDE score estimation** (not GMM). This document records every parameter, result, and file path.

## Method

**Score estimation**: Isotropic Gaussian KDE with Scott's bandwidth rule.
- Kernel: `K_h(u) ∝ exp(-||u||²/(2h²))` — single scalar bandwidth h for all dimensions.
- Bandwidth selection: Scott's rule `h = std × N^(-1/(d+4))`.
- No adaptivity (not Abramson, not balloon/k-NN).

**DA extraction**: Eigendecomposition of the lag-τ autocovariance matrix C(τ), computed from synthetic Langevin trajectories driven by the KDE score.
- `C(τ) = (1/N) Σₙ (X_τ⁽ⁿ⁾ − μ_f)(X₀⁽ⁿ⁾ − μ_f)ᵀ`, symmetrized, eigendecomposed.
- DA1 = leading eigenvector (max metastability direction).
- DA2 = second eigenvector at its own optimal lag.

**Lag selection**: Peak eigenvalue gap criterion.
- DA1 optimal lag: τ* = argmax_τ (λ₁/λ₂)
- DA2 optimal lag: τ* = argmax_τ (λ₂/λ₃)
- 28 geometric checkpoints from phys_t=0.1 to 2000.

## Parameters

| Parameter | Value |
|---|---|
| Feature space | 45D heavy-atom pairwise distances |
| Data source | MDShare `alanine-dipeptide-3x250ns-heavy-atom-distances.npz` |
| Total frames | 750,000 (3×250ns, frame interval = 1ps) |
| KDE centers | 2,000 (subsampled from 10k subsample of full data) |
| KDE bandwidth h (Scott) | 0.0183 |
| σ_f² (empirical) | 0.0009 |
| dt (Langevin) | 0.1 |
| Number of trajectories | 2,000 |
| Convergence checkpoints | 28 (geometric, phys_t 0.1–2000) |
| Max Langevin steps | 20,000 |
| Seed | 42 |
| Device | GPU (L40S, salmon partition) |
| TICA lag | 10 frames = 10 ps (standard convention) |
| TICA whitening ε | 1e-6 (deeptime default) |
| Ground truth | φ, ψ backbone dihedrals |

## Definitive Results (Paper Numbers)

### DA vs TICA vs PCA — Correlation with Dihedral Angles

| Method | |r(ψ)| | |r(φ)| | Notes |
|---|---|---|---|
| **DA1 (KDE, τ*=717.9)** | **0.6762** | 0.3352 | Slow mode (ψ) |
| **DA2 (KDE, τ*=33.2)** | 0.2133 | **0.7537** | Fast mode (φ) |
| TICA1 (lag=10ps) | 0.7534 | 0.2647 | Slow mode |
| TICA2 (lag=10ps) | 0.2004 | 0.5234 | Fast mode |
| PCA1 | 0.5628 | 0.5949 | Max variance (mixed) |
| PCA2 | 0.4065 | 0.5984 | 2nd variance (mixed) |

### Key Comparisons

| Comparison | DA | TICA | Winner |
|---|---|---|---|
| Slow mode (ψ) | DA1 \|r(ψ)\|=0.676 | TICA1 \|r(ψ)\|=0.753 | TICA slightly better |
| Fast mode (φ) | DA2 \|r(φ)\|=**0.754** | TICA2 \|r(φ)\|=0.523 | **DA much better (+0.23)** |

### Optimal Lags and Eigenvalue Gaps

| Direction | τ* (physical time) | Eigenvalue gap at τ* |
|---|---|---|
| DA1 (λ₁/λ₂) | 717.9 | 118.0× |
| DA2 (λ₂/λ₃) | 33.2 | 34.1× |

### Geometric Relationships

| Quantity | Value |
|---|---|
| angle(DA1, PCA1) | 28.5° |
| angle(DA1, TICA1) | 78.6° |

### Full Gap Profile (28 checkpoints)

| phys_t | gap12 (λ₁/λ₂) | gap23 (λ₂/λ₃) |
|---|---|---|
| 0.1 | 1.9 | 13.1 |
| 0.2 | 1.9 | 13.3 |
| 0.3 | 1.9 | 13.7 |
| 0.5 | 1.9 | 14.0 |
| 0.7 | 1.9 | 14.4 |
| 1.0 | 1.9 | 15.0 |
| 1.5 | 1.9 | 15.7 |
| 2.1 | 1.9 | 16.5 |
| 3.0 | 1.9 | 18.4 |
| 4.2 | 1.9 | 20.2 |
| 6.0 | 1.9 | 25.3 |
| 8.4 | 2.0 | 24.5 |
| 11.9 | 2.0 | 29.8 |
| 16.7 | 2.1 | 28.2 |
| 23.6 | 2.2 | 31.5 |
| **33.2** | 2.4 | **34.1** ← DA2 peak |
| 46.7 | 2.8 | 32.6 |
| 65.7 | 3.3 | 28.6 |
| 92.5 | 5.5 | 29.8 |
| 130.1 | 8.0 | 20.2 |
| 183.1 | 12.3 | 26.2 |
| 257.7 | 25.5 | 12.7 |
| 362.6 | 65.2 | 3.5 |
| 510.2 | 94.1 | 2.0 |
| **717.9** | **118.0** ← DA1 peak | 1.4 |
| 1010.1 | 35.2 | 3.1 |
| 1421.4 | 25.1 | 3.8 |
| 2000.0 | 6.5 | 5.4 |

## GMM Cross-Validation

GMM-DA was computed separately (4-component GMM, analytical score, dt=0.0001). The two methods agree:

| Direction | GMM |r| | KDE |r| (at GMM lag) | GMM↔KDE |cos| |
|---|---|---|---|
| DA1 (ψ) | 0.678 | 0.667 | 0.997 |
| DA2 (φ) | 0.814 | 0.810 | 0.994 |

Note: KDE at its own optimal lag gives slightly different numbers (DA1: 0.676 at τ*=718 vs 0.667 at τ*=1010; DA2: 0.754 at τ*=33 vs 0.810 at τ*=92.5). The GMM lag happens to be better for DA2 because the GMM's sharper score delays mode mixing, giving more time for DA2 to separate.

## SLURM Jobs

| Job | Date | Script | What | Runtime |
|---|---|---|---|---|
| 44378957 | 2026-03-31 | `compute_alanine_kde_da.py` | KDE-DA at GMM lags (cross-validation) | 62 min |
| 44395687 | 2026-04-05 | `compute_alanine_kde_gap_profile.py` | Full KDE convergence profile (28 checkpoints) | 108 min |

## Result Files

| File | Contents |
|---|---|
| `results/alanine_kde_gap_profile.npz` | KDE convergence profile: gap12, gap23, eigenvectors at each of 28 checkpoints, optimal eigenvalues/eigenvectors for DA1 and DA2 |
| `results/alanine_da_results_kde.npz` | KDE-DA at GMM-selected lags (cross-validation only) |
| `results/alanine_da_results_gmm_only.npz` | GMM-DA results + TICA + PCA (job 44317525) |

## Paper Figures

| Figure | Path | Contents | Size | DPI |
|---|---|---|---|---|
| fig_alanine_da_tica.png | `~/github/thesis/figures/` and `~/github/ssa-paper/figures/` | 2-panel Ramachandran: (a) DA1-KDE projection at τ*=718, (b) TICA1 projection | 8×4 in | 300 |
| fig_alanine_eigenvalue_gap.png | `~/github/thesis/figures/` and `~/github/ssa-paper/figures/` | gap12 and gap23 vs physical time from KDE convergence profile | 5×3 in | 300 |

Generated by: `experiments/da_directions/make_paper_figures.py`

## Scripts

| Script | Purpose |
|---|---|
| `compute_alanine_kde_da.py` | KDE-DA at fixed lags (GMM cross-validation) |
| `compute_alanine_kde_gap_profile.py` | Full KDE convergence profile (paper-version results) |
| `compute_da_comparison.py` | Original GMM-DA + TICA + PCA computation |
| `make_paper_figures.py` | Generate all paper figures (loads KDE gap profile) |
| `cluster/run_da_alanine_kde.sh` | SLURM for cross-validation run |
| `cluster/run_kde_gap_profile.sh` | SLURM for convergence profile run |
