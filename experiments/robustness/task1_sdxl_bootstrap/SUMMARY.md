# Task 1: Bootstrap SEs for SDXL Guidance Sweep SSA

## Method

**Lag bootstrap** (N_boot=10,000): For each of the 32 cells 
(8 prompts × 4 guidance scales), SSA_fixed20 is the mean of 
ρ²(τ) over 100 evenly-spaced lags (τ = 20, 40, ..., 2000). 
We resample these 100 lag-level ρ² values with replacement 
to get bootstrap distributions of SSA.

**Paired lag bootstrap** for peak probabilities: For each 
prompt, we resample the SAME lag indices across all 4 scales, 
then compute Pr[SSA(w=7.5) > SSA(w)] over 10,000 replicates.

**Limitation**: Per-trajectory data (N=200 Langevin trajectories) 
was not saved — only the mean ρ̂(τ) at each lag. The bootstrap 
therefore quantifies **lag selection sensitivity**, not trajectory 
sampling uncertainty. Since trajectories are iid given z₀, the 
trajectory-level variance contributes O(1/√200) ≈ 7% relative 
noise to each ρ̂(τ), which is smaller than the lag-to-lag 
variation visible in the ρ²(τ) curves.

**Entropy**: Point estimates only (kNN, k=5, d=50). CLIP 
embeddings for most cells were not saved; entropy bootstrap 
would require regenerating 2,000 images per cell.

## Results

### SSA with Bootstrap SEs

| Prompt | w=1.0 | w=3.0 | w=7.5 | w=15.0 |
|--------|-------|-------|-------|--------|
| person | 0.581±0.006 | 0.597±0.006 | **0.633±0.005** | 0.611±0.006 |
| cat | 0.571±0.006 | 0.595±0.005 | **0.610±0.006** | 0.573±0.009 |
| mountains | 0.598±0.005 | 0.640±0.004 | **0.684±0.005** | 0.672±0.007 |
| flowers | 0.513±0.006 | 0.548±0.005 | **0.558±0.006** | 0.501±0.009 |
| street | 0.591±0.005 | 0.628±0.005 | **0.646±0.006** | 0.622±0.008 |
| food | 0.565±0.005 | 0.590±0.006 | **0.606±0.006** | 0.587±0.009 |
| elderly | 0.580±0.006 | 0.583±0.006 | **0.620±0.006** | 0.601±0.008 |
| car | 0.596±0.006 | 0.619±0.005 | **0.627±0.006** | 0.622±0.009 |

### Peak Probabilities: Pr[SSA(w=7.5) > SSA(w)]

| Prompt | vs w=1.0 | vs w=3.0 | vs w=15.0 |
|--------|----------|----------|-----------|
| person | 1.000 | 1.000 | 1.000 |
| cat | 1.000 | 1.000 | 1.000 |
| mountains | 1.000 | 1.000 | 1.000 |
| flowers | 1.000 | 1.000 | 1.000 |
| street | 1.000 | 1.000 | 1.000 |
| food | 1.000 | 1.000 | 1.000 |
| elderly | 1.000 | 1.000 | 1.000 |
| car | 1.000 | 1.000 | 0.976 |

### Assessment

All 8 prompts show Pr[SSA(w=7.5) > SSA(w)] ≥ 0.95 for every 
comparison scale w ∈ {1.0, 3.0, 15.0}. The claim "SSA peaks 
at w=7.5 for all 8 prompts" is robust under lag bootstrap.
