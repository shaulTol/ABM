# Task 3: GMM Sweep Repeatability

## Method

Each of 4 sweep configurations was rerun with 10 independent 
random seeds. The seeds control both torch and numpy RNG, affecting 
Langevin trajectory initial conditions and noise. MI is analytical 
(deterministic given GMM parameters) so only SSA has stochastic 
variation.

Langevin parameters: n_trajectories=10,000, T=1,000, dt=0.01, 
max_lag=500, n_samples=2,000.

## Selected Sweeps

- **sep_2d**: 2D Bimodal: Varying Separation
- **var_2d**: 2D Bimodal: Varying Variance
- **weight_2d**: 2D Bimodal: Varying Weight Asymmetry
- **nmodes_fixdist_2d**: 2D Line: Varying Modes (Fixed Distance=4.0)

## Results: Coefficient of Variation

CV = SD/mean across 10 seeds, summarized over all parameter values:

| Sweep | CV mean | CV median | CV max | Max SD |
|-------|---------|-----------|--------|--------|
| sep_2d | 0.0045 | 0.0046 | 0.0077 | 0.0024 |
| var_2d | 0.0054 | 0.0054 | 0.0077 | 0.0024 |
| weight_2d | 0.0096 | 0.0087 | 0.0185 | 0.0044 |
| nmodes_fixdist_2d | 0.0041 | 0.0045 | 0.0049 | 0.0034 |

## Assessment

All CVs < 5%. SSA estimates are highly reproducible 
across seeds. The sweep curves in the paper are representative 
of the population mean within narrow error bands.
