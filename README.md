# Measuring and Decomposing Mode Separation via the Canonical Diffusion

Code repository for the paper. Implements **SSA** (Sum of Squared
Autocorrelations), a scalar multimodality measure, and **DA** (Dominant
Autocorrelation) directions, a directional decomposition ordered by
metastability.

## Algorithm Overview

Given samples from an unknown density f, the method works in three stages:

**Stage 1 — Canonical diffusion** (Algorithm 1, `cli/save_lags.py`).
Run the isotropic Langevin SDE:  *dX = (σ²_f / 2) ∇log f(X) dt + σ_f dW*,
where σ²_f = tr(Cov(X)) / d. Collect N trajectories, each of length T.
Compute the symmetrized autocovariance C_sym(τ) = (C(τ) + C(τ)ᵀ) / 2
at a grid of lags.

**Stage 2a — SSA** (`cli/compute_ssa.py`).
At each lag compute the trace-normalized autocorrelation
ρ̂(τ) = tr Ĉ(τ) / tr Ĉ(0), with Ĉ(τ) the trajectory-averaged lag-τ
autocovariance centered at the pooled sample mean.
Truncate at the first lag where ρ̂(τ)² ≤ 1/N (that lag included). Sum:
**Ŝ = Δt · Σ ρ̂(ℓΔt)²**.
A unimodal isotropic Gaussian gives Ŝ ≈ 1; Ŝ grows with barrier height.
Standard errors come from a trajectory bootstrap.

**Stage 2b — DA directions** (`cli/compute_da.py`).
τ\*(m) is the largest lag at which (i) the top m eigenvalues of C_sym(τ) exceed
the null threshold (the analytic free-convolution edge λ₊(τ), or the matched
Gaussian Monte Carlo null with Bonferroni over the m directions), (ii) the 1D
projection onto each of v₁…v_m rejects unimodality (dip test), and
(iii) ρ̂(τ) < 1/e. The number of metastable directions is the largest m with a
τ\*(m). The eigenvectors of C_sym(τ\*) are the DA directions, ordered by
metastability.

### Code ↔ Paper Mapping

| Paper | Code | Description |
|-------|------|-------------|
| Algorithm 1 (`alg:ssa_da`) | `cli/save_lags.py` | Langevin simulation + C_sym(τ) computation |
| SSA estimator + 1/N stopping rule (`eq:ssa_estimator`, `eq:tmax_stopping`) | `cli/compute_ssa.py` | Riemann sum truncated at the first 1/N crossing; trajectory bootstrap |
| DA lag-selection criterion (Section "DA") | `cli/compute_da.py` | τ\*(m), m̂, eigenvectors of C_sym(τ\*) |
| Algorithm 2 (`alg:operational_floor`) | `cli/compute_da.py` (`da_significance`) | Matched-Gaussian Monte Carlo null |
| Theorem 2 (`thm:limit-spectrum`) | `src/noise_floor/free_conv_utils.py` | Free-convolution upper edge λ₊(τ) |
| Langevin integrator | `src/simulation/langevin.py` | Euler-Maruyama SDE solver |
| KDE score estimator | `src/kde/kde_score.py` | ∇log f̂ for tabular data |
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
  robustness/           Appendix "Additional Robustness and Scope Controls"
    score_error/        Sensitivity to score-estimation error
    numerics/           Numerical parameters (N, Δt, lag window; Gaussian vs ellipses)
    sdxl_lag/           SDXL horizon and lag dependence (elderly DA across lags)
    null_scope/         Scope of the analytic null (overlapping pairs, score error)
    geometry/           Anisotropy, degenerate slow subspaces, and feature visibility
    figures/            Output of the robustness figure scripts
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

**SSA across 14 prompts x 4 guidance scales** (Fig 3, Table 1):
```bash
# Step 1: Generate x0 latents + compute entropy (GPU, array 0-27, ~7h each):
sbatch experiments/sdxl/cluster/run_guidance_gen.sh

# Step 2: Run Langevin + compute SSA (GPU, array 0-27, ~19h each):
sbatch experiments/sdxl/cluster/run_guidance_ssa.sh

# Step 3: Trajectory-averaged SSA recompute (GPU, array 0-23):
sbatch experiments/sdxl/cluster/run_traj_averaged.sh

# Step 4: Bootstrap SEs (CPU, single job, ~10 min; pooled estimator, 1/N rule):
sbatch experiments/sdxl/cluster/run_bootstrap_ssa.sh
# Output: experiments/sdxl/results/bootstrap_pooled/bootstrap_<cell>.json
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

### Robustness and Scope Controls (Appendix "Additional Robustness and Scope Controls")

Code in `experiments/robustness/`, one folder per subsection. SSA values use the
paper's 1/N first-crossing stopping rule; result files tagged `_fc` (first
crossing) are the ones the appendix reports. Small JSON/MD/CSV results are
tracked; `.npz` arrays are not.

**Sensitivity to score-estimation error** (`fig:rebuttal_score_error`):
```bash
sbatch experiments/robustness/score_error/cluster/run_score_robustness.sh        # GPU array 0-49
sbatch experiments/robustness/score_error/cluster/run_score_robustness_merge.sh  # CPU, after the array
cd experiments/robustness/score_error
python analyze_absolute_sensitivity.py   # median / p90 |ΔŜ| -> results/p9_absolute_sensitivity_fc*
python fig_score_error.py
```

**Numerical parameters** (`fig:rebuttal_numerics`):
```bash
sbatch experiments/robustness/numerics/cluster/run_numerical_robustness.sh   # N and Δt ladder
sbatch experiments/robustness/numerics/cluster/run_dispersion_robustness.sh  # Gaussian vs ellipses grid
cd experiments/robustness/numerics && python fig_numerical_robustness.py
```

**SDXL horizon and lag dependence** (elderly DA across lags; CPU, reads the
elderly DA results in `experiments/sdxl/results/`):
```bash
cd experiments/robustness/sdxl_lag
python reconcile_mc_null.py && python compute_lag_sensitivity.py   # -> RESULTS.md, results.json
```

**Scope of the analytic null** (`fig:rebuttal_null_score_error`):
```bash
sbatch experiments/robustness/null_scope/cluster/run_score_error_null.sh   # CPU
cd experiments/robustness/null_scope && python fig_null_fpr.py
```

**Anisotropy, degenerate slow subspaces, and feature visibility**
(`fig:rebuttal_geometry_gmm`, `fig:rebuttal_geometry_rings`):
```bash
sbatch experiments/robustness/geometry/cluster/run_four_mode_and_flat_top.sh   # CPU
cd experiments/robustness/geometry
python p3b_polar_da.py --ratio 8 --T 12000 --lags 250,1000,2000,3000           # two-shell rings, CPU
python fig_da_subspace.py && python fig_rings.py
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
| SDXL figure (`fig:sdxl_combined`) | `cd experiments/sdxl && python fig_sdxl_grids.py` |
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

### Robustness appendix

| Figure | Command |
|--------|---------|
| `fig:rebuttal_score_error` | `cd experiments/robustness/score_error && python fig_score_error.py` |
| `fig:rebuttal_numerics` | `cd experiments/robustness/numerics && python fig_numerical_robustness.py` |
| `fig:rebuttal_null_score_error` | `cd experiments/robustness/null_scope && python fig_null_fpr.py` |
| `fig:rebuttal_geometry_gmm` | `cd experiments/robustness/geometry && python fig_da_subspace.py` |
| `fig:rebuttal_geometry_rings` | `cd experiments/robustness/geometry && python fig_rings.py` |

Outputs go to `experiments/robustness/figures/`.

### Tables

| Table | Command |
|-------|---------|
| Table 1 + SE and full tables (SDXL SSA) | `cd experiments/sdxl && python table_sdxl_ssa.py` |
| Table 2 (alanine lag sensitivity) | `cd experiments/alanine && python table_alanine_lag_sensitivity.py` |
| Table 3 (alanine baselines) | `cd experiments/alanine && python table_alanine_baselines.py` |
| Table 4 (Bonferroni significance) | `cd experiments/alanine && python table_alanine_bonferroni.py` |

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
| SDXL bootstrap SEs (Table 1) | CPU | 1 | <1 |
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

**Faster autocorrelation (added after publication).**
`compute_vector_autocorrelation` (and `compute_vector_ssa_score`) in
`src/metrics/autocorrelation.py` accept `method='fft'`, which computes all lags
at once via Wiener–Khinchin in O(N·d·T log T) instead of the per-lag loop. It
computes the same estimator and matches the default `method='loop'` up to
float32 rounding (|Δρ| ≈ 1e-6). All results in the paper use the default loop.

**Minimal reproduction:** To regenerate figures from pre-computed results
(tracked JSON/CSV + synced NPZ): CPU-only, completes in minutes.
