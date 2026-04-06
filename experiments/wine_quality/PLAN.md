# DA vs PCA on Wine Quality (Red + White)

## Goal
Demonstrate that DA finds the red/white wine separation direction that PCA
misses, because the red/white split isn't the max-variance axis.

## Dataset
- UCI Wine Quality: ~1599 red + ~4898 white = ~6497 samples, 11 chemical features
- Download from UCI ML repository
- Red/white label kept for validation only (not used in DA/PCA computation)
- Drop quality score column, use only 11 continuous features
- Standardize (zero mean, unit variance)

## Method

### PCA
- Standard PCA, extract PC1

### DA
1. KDE score: Scott's bandwidth × 2 on 11D standardized features, 2000 centers
2. dt stability test
3. Canonical diffusion: N=5000 trajectories
4. Lag selection: checkpoints [1,2,4,8,16,32,64,128,256,512,1024,2048],
   argmax lambda_1/lambda_2
5. Extract DA1

## Outputs
- `PLAN.md` — this file
- `results.json` — all numerical results
- `projected_histograms.png` — 2×1: PC1 vs DA1 projections, colored red/white
- `loading_comparison.png` — side-by-side bar chart
- `eigenvalue_gap_profile.png` — gap ratio vs lag
- `eigenvalue_decay.png` — eigenvalue magnitudes vs lag

## Validation metrics (for both PC1 and DA1)
- SSA score (1D canonical diffusion)
- Bimodality coefficient (Sarle's)
- Point-biserial correlation with red/white label
- Dip test p-value
