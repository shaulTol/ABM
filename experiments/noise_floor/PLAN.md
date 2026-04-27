# Noise Floor Lag Selection — Experiment Plan

## Goal
Re-run DA experiments using a principled noise-floor-based lag selection criterion (permutation test) instead of the gap-maximization heuristic.

## Permutation Floor (operative criterion)
At each lag τ, destroy temporal correlation by shuffling trajectory indices: pair X_τ^(n) with X_0^(π(n)) for a random permutation π. Compute Ĉ_perm(τ), take its top eigenvalue. Repeat 20 times, take the 95th percentile as the empirical floor. This is the ground-truth threshold — it preserves the marginal distribution while destroying temporal structure.

**Lag selection**: τ* = largest lag where λ_1(τ), ..., λ_m(τ) ALL exceed the permutation floor at that lag.

## Analytic Floor (sanity check only)
```
λ_floor(τ) = exp(-τ/2) · σ_f² · (1 + √(d/N))² + σ_f² · √(d/N) · √(1 - exp(-τ))
```
Exact for isotropic Gaussian null. Plotted alongside permutation floor as a reference — should roughly agree for parallel ellipses, serve as rough lower bound for alanine/SDXL. NOT used for τ* selection.

## Directory Structure
```
experiments/noise_floor/
  PLAN.md
  EXPERIMENT_LOG.md
  noise_floor_utils.py          — shared: permutation floor, analytic floor, plotting
  alanine/
    run_alanine_noise_floor.py  — re-run Langevin, compute eigenvalues + permutation floor
    noise_floor_alanine.json    — results
  parallel_ellipses/
    run_ellipses_noise_floor.py — fresh run with permutation floor
    noise_floor_ellipses.json   — results
  sdxl/
    run_sdxl_noise_floor.py     — analyze cached data OR NEEDS_GPU.md
    noise_floor_sdxl.json       — results (if possible from cached data)
  mp_figures/
    generate_mp_figures.py      — MP baseline + eigenvalue floor + trajectories
```

## Figures (all saved to `figures/`)
- `fig_alanine_noise_floor.png` — λ_1,2,3(τ) + permutation floor + analytic floor vs τ
- `fig_parallel_ellipses_noise_floor.png` — same format
- `fig_mp_baseline.png` — MP law for 4 γ values (already done)
- `fig_mp_eigenvalue_floor.png` — C(τ) spectra at 10 lags (already done, update with analytic floor)
- `fig_mp_eigenvalue_trajectories.png` — λ_1 vs median vs ρ(τ) vs analytic floor (update)

## Experiments

### Exp 1: Parallel Ellipses (run first — fast, validates the method)
- d=2, means (0,±3), cov diag(25,1), weights (0.5,0.5), N=10000, dt=0.01
- Geometric checkpoints from τ=1 to τ=1600
- At each checkpoint: eigendecompose Ĉ(τ), compute permutation floor (20 shuffles, 95th pctile)
- m=1: find τ* where λ_1 > perm floor. Verify λ_2 < perm floor (only 1 significant direction).
- Verify DA1 at τ* points along y-axis (angle < 10°)
- Compare to old gap-max result (lag=800)

### Exp 2: Alanine Dipeptide (needs re-run for permutation floor)
- d=45, N=2000, KDE score (h=0.018, 2000 centers), dt=0.1
- Same 28-checkpoint geometric grid as existing experiment
- At each checkpoint: eigendecompose + permutation floor
- m=2: find τ* where both λ_1 and λ_2 > perm floor
- Compare DA1/DA2 correlations (|r(ψ)|, |r(φ)|) to old gap-max results
- Old: DA1 at τ*=718ps (|r(ψ)|=0.676), DA2 at τ*=33ps (|r(φ)|=0.754)
- Note: This needs GPU or long CPU run (~1hr). Check if can run locally.

### Exp 3: SDXL DA (from cached data if possible)
- d=768, N=200, cached eigenvalues at 20 lags in clip_da_autoresearch_square.npz
- Only top 10 eigenvalues saved — sufficient for m=1 floor check
- Need σ_f² from CLIP embeddings and raw trajectory pairs for permutation test
- If raw data unavailable → NEEDS_GPU.md with exact requirements

### Exp 4: MP Figures (update existing)
- Add analytic floor line to fig_mp_eigenvalue_trajectories.png
- Add analytic floor dashed line to fig_mp_eigenvalue_floor.png panels
