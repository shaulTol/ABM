# Whitening-Variant Experiment Suite

Replicates key paper experiments with **whitening** applied before Langevin simulation.

## Motivation

The paper's canonical diffusion uses scalar A₀ = σ²_f I_d. The whitening variant first
transforms each distribution f to identity covariance (X̃ = Σ^{-1/2}(X - μ)), then runs
standard isotropic Langevin with σ²_f̃ ≈ 1. This is mathematically equivalent to
A₀ = Cov_f(X) but avoids per-step covariance multiplications.

## Structure

```
experiments_whitening/
├── shared/
│   ├── whitening.py                     # WhiteningTransform class
│   ├── canonical_diffusion_whitened.py   # GenericLangevinSimulator wrapper
│   └── tests/test_whitening.py          # Unit tests
├── exp01_gmm_sweeps/          # 4 sweeps × 30 points × 10 seeds (10D, N=200)
├── exp02_random_gmm/          # 50 random bimodal GMMs (10D, N=200)
├── exp04_dispersion/          # 2D blob vs parallel ellipses (smoke test)
├── exp05_sdxl_guidance/       # 3 prompts × 4 guidance scales (768D CLIP)
├── exp06_sdxl_elderly_da/     # Elderly DA1 with whitened CLIP embeddings
├── exp07_alanine/             # 45D pairwise-distance KDE-DA
└── exp10_planted_spikes/      # d=200, k=20 planted metastable directions
```

## Whitening Procedure

For analytic Σ_f (GMM experiments): `W = Σ_f^{-1/2}` via eigendecomposition.
For empirical Σ̂ (SDXL, alanine): Ledoit-Wolf shrinkage before inversion.

Score in whitened coordinates:
```
∇log f̃(x̃) = W_inv^T · ∇log f(W_inv · x̃ + μ)
```

## Compute Budget

| Experiment | Partition | GPU-hours | Notes |
|------------|-----------|-----------|-------|
| Exp 1 (GMM sweeps) | catfish | ~0 (CPU) | ~3h wall |
| Exp 2 (Random GMM) | catfish | ~0 (CPU) | ~2h wall |
| Exp 4 (Dispersion) | local | 0 | minutes |
| Exp 5 (SDXL guidance) | salmon | 180-360 | 12 parallel jobs |
| Exp 6 (SDXL elderly DA) | salmon | 30 | 1 job, after Exp 5 |
| Exp 7 (Alanine) | catfish | ~1 | ~1h wall |
| Exp 10 (Planted spikes) | catfish | ~2 | ~2h wall |

## Running

```bash
# Local smoke test (Exp 4, CPU only):
python experiments_whitening/exp04_dispersion/run.py

# Cluster (synthetic experiments):
sbatch experiments_whitening/exp01_gmm_sweeps/slurm.sh
sbatch experiments_whitening/exp02_random_gmm/slurm.sh
sbatch experiments_whitening/exp07_alanine/slurm.sh

# Cluster (SDXL, needs salmon L40S):
sbatch experiments_whitening/exp05_sdxl_guidance/slurm.sh  # job array 0-11
sbatch experiments_whitening/exp06_sdxl_elderly_da/slurm.sh

# Planted spikes (local or cluster):
python experiments_whitening/exp10_planted_spikes/run.py
```

## Dependency Order

```
shared/whitening.py + tests ──→ Exp 1,2,4,10,7 (synthetic, parallel)
                             ──→ Exp 5 (SDXL guidance)
                                     ──→ Exp 6 (elderly DA, needs Exp 5's W)
```
