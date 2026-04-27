# Noise Floor Experiments — Log

## 2026-04-13

### Parallel Ellipses — DONE
- d=2, N=10000, dt=0.01, 24 geometric checkpoints (1–1600)
- **τ* = 1600** (permutation floor, m=1) — largest lag tested, λ₁ still above floor
- λ₁(τ*) = 1.30 > perm_floor = 0.50
- λ₂(τ*) = -0.18 < perm_floor → confirms only 1 significant direction
- DA1 angle from y-axis: **6.9°** (correct — mode separation is along y)
- Analytic floor way off (σ_f² = 17.5 due to large x-variance) — permutation floor is correct
- Old gap-max lag was 800; noise-floor criterion goes to 1600 (entire grid)
- **Figure**: `figures/fig_parallel_ellipses_noise_floor.png`
- **Scatter**: `figures/fig_parallel_ellipses_noise_floor_scatter.png`
- **JSON**: `experiments/noise_floor/parallel_ellipses/noise_floor_ellipses.json`

### SDXL Square — DONE (from cached data, conservative floor)
- d=768, N=200, γ=3.84 (severely underdetermined)
- σ_f² = 0.000340 (from CLIP embeddings)
- **Permutation floor (x0 vs shuffled x0) = 0.0535** — exceeds ALL eigenvalues at ALL lags
- Max λ₁ across all lags = 0.024 (at lag=10) — well below floor
- **No lag found where signal exceeds noise floor**
- This is expected: with d/N = 3.84, the random matrix noise dominates
- Analytic floor = 0.002–0.003 (much lower — formula not valid for non-isotropic, high-γ regime)
- **Implication**: DA with N=200, d=768 is in the noise-dominated regime. Need N >> d for reliable eigendecomposition. Either reduce d (PCA preprocess) or increase N.
- **Figure**: `figures/fig_sdxl_noise_floor.png`
- **JSON**: `experiments/noise_floor/sdxl/noise_floor_sdxl.json`

### Alanine Dipeptide — SUBMITTED (job 44456636)
- d=45, N=2000, γ=0.0225, KDE score (h=0.018, 2000 centers), dt=0.1
- 28 geometric checkpoints (1–20000 steps = 0.1–2000 ps)
- Submitted to salmon partition (GPU), should take ~1hr
- Old results: DA1 at τ*=718ps (gap-max), DA2 at τ*=33ps
- Awaiting results for noise-floor τ* comparison

### MP Figures — DONE (updated)
- Added analytic floor (Eq. 7) line to fig_mp_eigenvalue_trajectories.png
- Added floor reference to fig_mp_eigenvalue_floor.png panels
- All 3 figures regenerated in `figures/`
