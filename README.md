# Measuring and Decomposing Mode Separation via the Canonical Diffusion

Code repository for the paper. Implements **SSA** (Sum of Squared
Autocorrelations), a scalar multimodality measure, and **DA** (Dominant
Autocorrelation) directions, a directional decomposition ordered by
metastability.

## Algorithm Overview

Given samples $X$ from an unknown density $f$, the method works in three stages:

**Stage 1 — Canonical diffusion** (Algorithm 1, `cli/save_lags.py`).
Run the isotropic Langevin SDE

$$dX = \tfrac{\sigma_f^2}{2} \nabla \log f(X)\, dt + \sigma_f\, dW$$

where $\sigma_f^2 = \mathrm{tr}\,\mathrm{Cov}(X)/d$. Collect $N$ trajectories, each of length $T$.
Compute the symmetrized autocovariance $C_{\mathrm{sym}}(\tau) = (C(\tau) + C(\tau)^\top)/2$
at a grid of lags.

**Stage 2a — SSA** (`cli/compute_ssa.py`).
At each lag compute the trace-normalized autocorrelation
$\hat\rho(\tau_\ell) = \mathrm{tr}\,\hat C(\tau_\ell)\,/\,\mathrm{tr}\,\hat C(0)$.
Truncate at the stopping lag $T_{\max}$ where $\hat\rho(\tau)^2$ hits the $1/N$ noise floor. Sum:

$$\hat{S} = \Delta t \sum_{\ell=0}^{L_{\max}} \hat\rho(\ell\,\Delta t)^2$$

This is a scalar in $[0, \infty)$ that is zero for unimodal distributions and grows with mode separation.

**Stage 2b — DA directions** (`cli/compute_da.py`).
Pick $\tau^*$ where the second eigenvalue crosses the noise floor. The eigenvectors
of $C_{\mathrm{sym}}(\tau^*)$ are the DA directions, ordered by metastability. DA1 points
along the slowest-mixing axis of $f$.

### Code ↔ Paper Mapping

| Paper | Code | Description |
|-------|------|-------------|
| Algorithm 1 | `cli/save_lags.py` | Langevin simulation + $C_{\mathrm{sym}}(\tau)$ computation |
| Eq. 3 (SSA) | `cli/compute_ssa.py` | Riemann-sum estimator with noise floor |
| Def. 3 (DA) | `cli/compute_da.py` | Eigenvectors of $C_{\mathrm{sym}}(\tau^*)$ |
| Theorem 2 (noise floor) | `src/noise_floor/free_conv_utils.py` | Free-convolution upper edge $\lambda_+(\tau)$ |
| Langevin integrator | `src/simulation/langevin.py` | Euler-Maruyama SDE solver |
| KDE score estimator | `src/kde/kde_score.py` | $\nabla \log \hat{f}$ for tabular data |
| DDPM score (SDXL) | `experiments/sdxl/sdxl_components.py` | Tweedie denoiser as score oracle |

## Setup

```bash
pip install -r requirements.txt
# For SDXL experiments: GPU required (L40S 48GB recommended)
# Additional: pip install latent_clip open_clip_torch==3.3.0 diffusers==0.30.0
```

### Cluster Configuration

SLURM scripts use placeholders. Replace them before running:

| Placeholder | Meaning | Example |
|-------------|---------|---------|
| `<CLUSTER_USER>` | Your cluster username | `jdoe` |
| `<CLUSTER_HOST>` | SSH jump host for tunnel | `gateway.university.edu` |
| `<PROJECT_DIR>` | Absolute path to this repo on cluster | `/scratch/jdoe/ssa-paper` |
| `<CONDA_ENV>` | Absolute path to conda environment | `/scratch/jdoe/envs/ssa_env` |
| `<CONDA_SH>` | Path to `conda.sh` activation script | `/opt/miniconda3/etc/profile.d/conda.sh` |
| `<CONDA_BIN>` | Path to conda `bin/` directory | `/opt/miniconda3/bin` |
| `<NV_SITE_PACKAGES>` | nvidia packages in site-packages | `<CONDA_ENV>/lib/python3.12/site-packages/nvidia` |
| `<USER_SITE_PACKAGES>` | User-installed nvidia packages | `~/.local/lib/python3.12/site-packages/nvidia` |
| `<TORCH_LIB>` | PyTorch shared libraries | `<CONDA_ENV>/lib/python3.12/site-packages/torch/lib` |
| `<HF_CACHE>` | Hugging Face model cache | `~/.cache/huggingface` |

## Repository Structure

```
src/                    Core library
  metrics/              SSA, DA, autocorrelation
  simulation/           Langevin integrator
  kde/                  KDE score estimator
  noise_floor/          Free-convolution density and noise floor
  plotting/             Shared figure styling
cli/                    Shared pipeline (Algorithm 1 → SSA / DA)
experiments/
  synthetic/            Section 7.1: GMM sweeps, planted spikes, ellipses, noise floor
  sdxl/                 Section 7.2: SDXL guidance sweep, elderly DA, crane
  alanine/              Section 7.3: alanine dipeptide KDE-DA, baselines, MC null
```

---

## Running the Experiments

Each experiment follows the same pattern: a **producer** script runs the
Langevin simulation and saves results, then a **figure** script reads those
results and generates plots. Producer scripts are submitted as SLURM jobs;
figure scripts run locally in seconds.

### Synthetic GMMs (Section 7.1)

**GMM parameter sweeps** (Fig 1a-d) — SSA vs separation, variance, weight, mode count:
```bash
# Run (GPU, ~12h):
sbatch experiments/synthetic/cluster/run_gmm_sweeps.sh
# Output: experiments/synthetic/results/gmm_10d_n200_all_sweeps_T200.json
```

**Random GMM scatter** (Fig A2) — SSA rank vs MI rank over 50 random GMMs:
```bash
sbatch experiments/synthetic/cluster/run_random_sweep.sh
# Output: experiments/synthetic/results/random_sweep_10d_n200.json
```

**dt-robustness** (Fig A3) — SSA stability across dt in {0.005, 0.01, 0.02}:
```bash
sbatch experiments/synthetic/cluster/run_dt_robustness.sh
# Output: experiments/synthetic/results/gmm_10d_n200_dt_robustness.json
```

**On-the-fly experiments** (no pre-computation needed):
```bash
cd experiments/synthetic
python fig_dispersion.py           # Fig 1e: dispersion vs fragmentation
python fig_parallel_ellipses.py    # Fig 1f + A9: DA1 recovers slow axis
python fig_a0_identity.py          # A0=sigma^2 I scale invariance
python fig_noise_floor.py          # Fig A10-A13: free-conv validation
```

**Planted spikes** (Fig A15):
```bash
cd experiments/synthetic
python simulate_planted_spikes.py  # produces results/fig_planted_spikes.npz
```

**Ellipses MC null** (Table 4):
```bash
sbatch experiments/synthetic/cluster/run_ellipses_null.sh
# Output: experiments/synthetic/results/ellipses_null_results.npz
```

### SDXL Guidance Sweep (Section 7.2)

**SSA across 7 prompts x 4 guidance scales** (Fig 3, Table 1):
```bash
# Step 1: Generate x0 latents + compute entropy (GPU, array 0-27, ~7h each):
sbatch experiments/sdxl/cluster/run_guidance_gen.sh

# Step 2: Run Langevin + compute SSA (GPU, array 0-27, ~19h each):
sbatch experiments/sdxl/cluster/run_guidance_ssa.sh

# Step 3: Trajectory-averaged SSA recompute (GPU, array 0-23):
sbatch experiments/sdxl/cluster/run_traj_averaged.sh

# Step 4: Bootstrap CIs (CPU, array 0-58):
sbatch experiments/sdxl/cluster/run_bootstrap_ssa.sh
```

**Elderly DA extraction** (Fig 4, A7, A8):
```bash
# DA directions (GPU, ~24h):
sbatch experiments/sdxl/cluster/run_da_elderly.sh

# C(tau) for spectrum figure (GPU, ~20h):
sbatch experiments/sdxl/cluster/run_sdxl_C_tau.sh

# C(0) eigenvalues for free-conv overlay:
sbatch experiments/sdxl/cluster/run_save_c0_eigvals_elderly.sh
```

**MC null test** (Table 4 SDXL row):
```bash
sbatch experiments/sdxl/cluster/run_sdxl_null.sh  # array 0-2, ~19h each
```

**Image decode** (crane extremes, guidance grids):
```bash
sbatch experiments/sdxl/cluster/run_decode_crane.sh
sbatch experiments/sdxl/cluster/run_decode_grids.sh
```

**Weight comparability** (Appendix H):
```bash
sbatch experiments/sdxl/cluster/run_weights.sh
```

### Alanine Dipeptide (Section 7.3)

Requires raw MD trajectories in `experiments/alanine/alanine_dipeptide/data/`
(~135 MB, download from [MDshare](http://ftp.imp.fu-berlin.de/pub/cmb-data/)).

**C(tau) + KDE-DA** (Fig 5, Table 2):
```bash
# C(tau) at 38 lags (GPU, ~4h):
sbatch experiments/alanine/cluster/run_alanine_C_tau.sh

# KDE-DA directions (GPU, ~12h):
sbatch experiments/alanine/cluster/run_alanine_kde_da.sh
```

**MC null test** (Table 4 alanine row):
```bash
sbatch experiments/alanine/cluster/run_alanine_null.sh  # array 0-9, ~8 min each
```

**Baselines** (Table 3 — sparse K-means, spectral map, diffusion maps):
```bash
sbatch experiments/alanine/cluster/run_baselines.sh  # CPU-only, ~10 min each
```

**On-the-fly** (no pre-computation):
```bash
cd experiments/alanine
python fig_tica_its.py  # Fig A14: TICA implied timescales
```

---

## Reproducing Figures from Pre-computed Results

All small result files (JSON/CSV, ~1.5 MB total) are tracked in git.
Binary NPZ files are gitignored — sync from cluster or regenerate via SLURM.

Once results are in place, figures regenerate locally in seconds (CPU-only):

### Main Text

| Figure | Command |
|--------|---------|
| Fig 1a-d (GMM sweeps) | `cd experiments/synthetic && python fig_gmm_sweeps.py` |
| Fig 1e (dispersion) | `cd experiments/synthetic && python fig_dispersion.py` |
| Fig 1f (parallel ellipses) | `cd experiments/synthetic && python fig_parallel_ellipses.py` |
| Fig 3 (SDXL guidance) | `cd experiments/sdxl && python fig_sdxl_guidance.py` |
| Fig 4 (elderly DA grid) | `cd experiments/sdxl && python fig_elderly_da.py --mode grid` |
| Fig 5 (alanine DA vs TICA) | `cd experiments/alanine && python fig_alanine_da_tica.py` |

### Appendix

| Figure | Command |
|--------|---------|
| Fig A1 (GMM snapshots) | same as Fig 1 |
| Fig A2 (random sweep) | `cd experiments/synthetic && python fig_random_sweep.py` |
| Fig A3 (dt-robustness) | `cd experiments/synthetic && python fig_dt_robustness.py` |
| Fig A4-A5 (alanine bulk) | `cd experiments/alanine && python fig_alanine_bulk.py` |
| Fig A6 (spike count vs tau) | `cd experiments/alanine && python fig_alanine_spike_count.py` |
| Fig A7 (elderly spectrum) | `cd experiments/sdxl && python fig_elderly_da.py --mode spectrum` |
| Fig A8 (elderly histograms) | `cd experiments/sdxl && python fig_elderly_da.py --mode histogram` |
| Fig A9 (ellipses convergence) | same as Fig 2b |
| Fig A10-A13 (noise floor) | `cd experiments/synthetic && python fig_noise_floor.py` |
| Fig A14 (TICA timescales) | `cd experiments/alanine && python fig_tica_its.py` |
| Fig A15 (planted spikes) | `cd experiments/synthetic && python fig_planted_spikes.py --npz results/fig_planted_spikes.npz` |
| $A_0 = \sigma^2 I$ figure | `cd experiments/synthetic && python fig_a0_identity.py` |

### Tables

| Table | Command |
|-------|---------|
| Table 1 (SDXL SSA + entropy) | `cd experiments/sdxl && python fig_sdxl_guidance.py` (prints to stdout) |
| Table 2 (alanine lag sensitivity) | `cd experiments/alanine && python table_alanine_lag_sensitivity.py` |
| Table 3 (alanine baselines) | `cd experiments/alanine && python table_alanine_baselines.py` |
| Table 4 (Bonferroni significance) | `cd experiments/alanine && python table_alanine_bonferroni.py` |
| App H (weight comparability) | `cd experiments/sdxl && python table_weights.py` |

---

## Syncing Results from Cluster

For binary NPZ files:

```bash
# Set up SSH tunnel (replace with your cluster details)
ssh -N -L 2222:<CLUSTER_HOST>:22

# Sync all result files (replace <CLUSTER_USER> and <PROJECT_DIR>)
rsync -avz -e 'ssh -p 2222' \
  <CLUSTER_USER>@localhost:<PROJECT_DIR>/experiments/synthetic/results/ \
  experiments/synthetic/results/

rsync -avz -e 'ssh -p 2222' \
  <CLUSTER_USER>@localhost:<PROJECT_DIR>/experiments/sdxl/results/ \
  experiments/sdxl/results/

rsync -avz -e 'ssh -p 2222' \
  <CLUSTER_USER>@localhost:<PROJECT_DIR>/experiments/alanine/results/ \
  experiments/alanine/results/
```

---

## Compute Requirements

All experiments run on single-GPU SLURM jobs. No multi-GPU or distributed
training is required.

| Experiment | GPU | Array jobs | Est. GPU-hrs |
|---|---|---|---|
| SDXL guidance sweep SSA (Fig 3) | L40S | 28 | 532 |
| SDXL generation + entropy (Fig 3) | L40S | 28 | 196 |
| SDXL traj-averaged recompute (Table 1) | L40S | 24 | 192 |
| SDXL MC null (Table 4) | L40S | 3 | 57 |
| SDXL bootstrap CIs (Table 1) | L40S | 59 | 30 |
| SDXL elderly DA (Fig 4) | L40S | 1 | 24 |
| SDXL C(tau) (Fig A7) | L40S | 1 | 20 |
| SDXL decode grids + crane | L40S | 29 | 15 |
| GMM sweeps + random + dt-robust (Fig 1, A2, A3) | L4 | 3 | 24 |
| Alanine KDE-DA (Fig 5) | L40S | 1 | 6 |
| Alanine C(tau) + null + baselines | L4 | 12 | 5 |
| On-the-fly figures (Fig 1e-f, A0, A9-A14) | CPU | -- | <1 |
| **Total** | | | **~1,100** |

SDXL Langevin dominates (~950 GPU-hours): each cell runs N=200 trajectories x
T=2000 UNet forward passes at 1024x1024, taking ~19h on L40S.

**Minimal reproduction:** To regenerate figures from pre-computed results
(tracked JSON/CSV + synced NPZ): CPU-only, completes in minutes.
