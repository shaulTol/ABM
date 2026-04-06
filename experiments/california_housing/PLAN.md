# DA vs PCA on California Housing

## Goal
Demonstrate that DA (Dominant Autocorrelation) finds a meaningful subpopulation
separation direction that PCA misses. The key idea: PCA finds the max-variance
axis, while DA finds the axis along which the distribution is most multimodal
(slowest-mixing under canonical diffusion).

## Dataset
- `sklearn.datasets.fetch_california_housing`: 20,640 samples, 8 features
- Standardize to zero mean, unit variance

## Method

### PCA
- Standard PCA on standardized data, extract PC1

### DA
1. **Score estimation**: KDE with Scott's bandwidth on all 20,640 samples
   (only 8D, so full KDE is feasible). Use 5000 random centers to keep
   score evaluation tractable.
2. **Canonical diffusion**: GenericLangevinSimulator with:
   - dt stability test (try 0.1, 0.01, 0.001)
   - N=10,000 trajectories
3. **Lag selection**: Geometric checkpoints [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
   select tau* via argmax lambda_1/lambda_2 gap ratio
4. **Extract DA1**: Leading eigenvector of C(tau*)

### Comparison
- Cosine similarity between PC1 and DA1
- Feature loadings bar chart
- Projected histograms (colored by median income above/below median)
- SSA of 1D projections onto PC1 vs DA1
- Eigenvalue gap profile across lags

## Outputs
All saved to `experiments/california_housing/`:
- `results.json` — numerical results
- `projected_histograms.png` — 2x1 histogram figure
- `loading_comparison.png` — side-by-side loading bar chart
- `eigenvalue_gap_profile.png` — gap ratio vs lag
- `eigenvalue_decay.png` — eigenvalue magnitudes vs lag

## SSA on 1D projections
For the 1D projected data (onto PC1 or DA1), we run 1D canonical diffusion
(KDE score in 1D) and compute the scalar SSA = (1/L) sum rho(tau)^2.
This directly measures how multimodal the projection is.
