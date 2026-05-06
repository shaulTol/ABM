# SSA Paper — Reproduction Guide

Code for *"Measuring and Decomposing Multimodality via Canonical Diffusions"*.

**SSA** (Sum of Squared Autocorrelations): scalar multimodality measure.
**DA** (Dominant Autocorrelation) directions: directional decomposition ordered by metastability.

## Setup

```bash
pip install -r requirements.txt
# For SDXL experiments: GPU required (L40S 48GB recommended)
# Additional: pip install latent_clip open_clip_torch==3.3.0 diffusers==0.30.0
```

## Repository Structure

```
src/                    Core library (metrics, simulation, KDE, noise floor, plotting)
cli/                    Shared pipeline: save_lags → compute_ssa / compute_da
experiments/
  synthetic/            GMM sweeps, planted spikes, ellipses, free-conv, dt-robustness
  sdxl/                 Guidance sweep, elderly DA, crane, MC null, bootstrap
  alanine/              KDE-DA, baselines (sparse K-means, spectral map, DM), MC null
older_versions/         Archived files (gitignored)
```

## Shared Pipeline (cli/)

| Script | Function |
|--------|----------|
| `cli/save_lags.py` | Simulate Langevin, compute C_sym(τ), save NPZ |
| `cli/compute_ssa.py` | Compute SSA (Ŝ) + bootstrap CI from saved C(τ) |
| `cli/compute_da.py` | Compute DA directions + MC significance from saved C(τ) |

## Alanine Data

Raw alanine dipeptide MD trajectories needed in `experiments/alanine/alanine_dipeptide/data/`.
Gitignored due to size (~135 MB). Download from [MDshare](http://ftp.imp.fu-berlin.de/pub/cmb-data/).

---

## Paper Figures — Exact Reproduction Commands

All commands below are run from the repository root. Data status:
- **tracked** = JSON/CSV result files committed to git; figures regenerate locally
- **sync** = binary NPZ files; gitignored, sync from cluster or regenerate via SLURM
- **on-the-fly** = computed at figure-generation time, no pre-saved data needed

### Main Text

---

**Fig 1** (GMM sweeps — 4-panel: sep/var/weight/nmodes, 10D N=200)
```
Figure:   cd experiments/synthetic && python fig_gmm_sweeps.py
```
Producer: `sbatch experiments/synthetic/cluster/run_gmm_sweeps.sh`
          (catfish, 1× GPU, 32G mem, 12h time limit)
Data:     `experiments/synthetic/results/gmm_10d_n200_all_sweeps_T200.json` (tracked)

---

**Fig 2a** (dispersion vs fragmentation — blob vs bimodal GMM)
```
Figure:   cd experiments/synthetic && python fig_dispersion.py
```
Producer: on-the-fly (simulates 2000 trajectories, computes DA + SSA + entropy)
Data:     on-the-fly
Runtime:  minutes (GPU optional)

---

**Fig 2b** (parallel ellipses — DA1 recovers slow axis)
```
Figure:   cd experiments/synthetic && python fig_parallel_ellipses.py
```
Producer: on-the-fly (simulates 10k trajectories, convergence profile across 14 lags)
Data:     on-the-fly
Runtime:  minutes (GPU recommended)

---

**Fig 3** (SDXL guidance sweep — SSA vs entropy, 7 prompts × 4 scales)
```
Figure:   cd experiments/sdxl && python fig_sdxl_guidance.py
```
Producer (SSA):     `sbatch experiments/sdxl/cluster/run_guidance_ssa.sh`
                    (salmon, 1× L40S, 48G mem, 3-day time limit, array 0-27)
Producer (entropy): `sbatch experiments/sdxl/cluster/run_guidance_gen.sh`
                    (salmon, 1× L40S, 48G mem, 3-day time limit, array 0-27)
Data:     `experiments/sdxl/results/traj_avg_*.json` (tracked, 56 files)
          `experiments/sdxl/results/gen_*_entropy.json` (tracked, 56 files)

---

**Fig 4** (elderly DA — DA1 vs PC1 image grid)
```
Figure:   cd experiments/sdxl && python fig_elderly_da.py --mode grid
```
Producer: `sbatch experiments/sdxl/cluster/run_da_elderly.sh`
          (salmon, 1× L40S, 48G mem, 2-day time limit)
Data:     `experiments/sdxl/results/clip_da_elderly_gs7p5_fc_dt0p0025.json` (tracked)
          `experiments/sdxl/results/clip_da_elderly_gs7p5_fc_dt0p0025.npz` (sync, 3.1 MB)
          `experiments/sdxl/results/gen_mp_p5_elderly_gs7p5_quick_x0.npz` (sync, latents)
Runtime:  requires GPU (VAE decode + CLIP embed)

---

**Fig 5** (alanine DA vs TICA — Ramachandran scatter)
```
Figure:   cd experiments/alanine && python fig_alanine_da_tica.py
```
Producer (C(τ)):  `sbatch experiments/alanine/cluster/run_alanine_C_tau.sh`
                  (catfish, 1× L4, 16G mem, 4h time limit)
Producer (DA):    `sbatch experiments/alanine/cluster/run_alanine_kde_da.sh`
                  (salmon, 1× L40S, 32G mem, 12h time limit)
Data:     `experiments/alanine/results/alanine_da_results_kde.npz` (sync, 6 KB)
          `experiments/alanine/results/alanine_C_tau_full_h_half_10k.npz` (sync, 5.4 MB)
          Requires raw alanine data in `alanine_dipeptide/data/` (~135 MB, see above)

---

### Appendix

---

**Fig A1** (GMM snapshots)

Same script as Fig 1 (snapshot panels generated within `fig_gmm_sweeps.py`).

---

**Fig A2** (random GMM scatter — SSA rank vs MI rank)
```
Figure:   cd experiments/synthetic && python fig_random_sweep.py
```
Producer: `sbatch experiments/synthetic/cluster/run_random_sweep.sh`
          (catfish, 1× GPU, 32G mem, 6h time limit)
Data:     `experiments/synthetic/results/random_sweep_10d_n200.json` (tracked)

---

**Fig A3** (dt-robustness — 3 dt values overlaid)
```
Figure:   cd experiments/synthetic && python fig_dt_robustness.py
```
Producer: `sbatch experiments/synthetic/cluster/run_dt_robustness.sh`
          (catfish, 1× GPU, 32G mem, 24h time limit)
Data:     `experiments/synthetic/results/gmm_10d_n200_dt_robustness.json` (tracked)
          `experiments/synthetic/results/gmm_10d_n200_all_sweeps_T200.json` (tracked)
          `experiments/synthetic/results/gmm_10d_dt_robust_dt0p02_T200.json` (tracked)

---

**Fig A4–A5** (alanine bulk validation — eigenvalue histograms at 6 lags)
```
Figure:   cd experiments/alanine && python fig_alanine_bulk.py
```
Producer: same as Fig 5 (`run_alanine_C_tau.sh`)
Data:     `experiments/alanine/results/alanine_C_tau_full_h_half_10k.npz` (sync)

---

**Fig A6** (alanine spike count vs lag τ)
```
Figure:   cd experiments/alanine && python fig_alanine_spike_count.py
```
Producer: same as Fig 5 (`run_alanine_C_tau.sh`)
Data:     `experiments/alanine/results/alanine_C_tau_full_h_half_10k.npz` (sync)

---

**Fig A7** (elderly eigenvalue spectrum + MC null + free-conv density)
```
Figure:   cd experiments/sdxl && python fig_elderly_da.py --mode spectrum
```
Producer: `sbatch experiments/sdxl/cluster/run_sdxl_C_tau.sh`
          (salmon, 1× L40S, 48G mem, 1-day time limit)
Data:     `experiments/sdxl/results/clip_da_elderly_gs7p5_fc_dt0p0025.npz` (sync)
          `experiments/sdxl/results/elderly_gs7p5_n200_C_tau_full.npz` (sync, 273 MB)
          `experiments/sdxl/results/c0_eigvals_elderly_gs7p5_fc_dt0p0025.npz` (sync)
Runtime:  CPU-only (no GPU needed)

---

**Fig A8** (elderly DA1/DA2 projection histograms)
```
Figure:   cd experiments/sdxl && python fig_elderly_da.py --mode histogram
```
Producer: same as Fig 4 (`run_da_elderly.sh`)
Data:     `experiments/sdxl/results/clip_da_elderly_gs7p5_fc_dt0p0025.npz` (sync)
          `experiments/sdxl/results/gen_mp_p5_elderly_gs7p5_quick_clip.npz` (sync)
Runtime:  CPU-only (no GPU needed)

---

**Fig A9** (parallel ellipses — convergence panels)

Same script as Fig 2b (appendix panels generated within `fig_parallel_ellipses.py`).

---

**Fig A10–A13** (free-convolution noise floor validation — 6γ × 4τ grid)
```
Figure:   cd experiments/synthetic && python fig_noise_floor.py
```
Producer: on-the-fly (simulates OU processes, eigendecomposes C_sym(τ), overlays MP density)
Data:     on-the-fly
Runtime:  minutes (CPU-only)

---

**Fig A14** (TICA implied timescales)
```
Figure:   cd experiments/alanine && python fig_tica_its.py
```
Producer: on-the-fly (computes TICA from raw alanine data)
Data:     on-the-fly (requires raw alanine data in `alanine_dipeptide/data/`)
Runtime:  fast (CPU-only)

---

**Fig A15** (planted spikes — mode count vs lag)
```
Figure:   cd experiments/synthetic && python fig_planted_spikes.py --npz results/fig_planted_spikes.npz
```
Producer: `cd experiments/synthetic && python simulate_planted_spikes.py`
Data:     `experiments/synthetic/results/fig_planted_spikes.npz` (sync)

---

**A₀=σ²I figure** (scale-invariance demonstration)
```
Figure:   cd experiments/synthetic && python fig_a0_identity.py
```
Producer: on-the-fly (3 scaled GMMs)
Data:     on-the-fly
Runtime:  minutes (GPU optional)

---

### Tables

---

**Table 1** (SDXL SSA + entropy across 14 prompts × 4 guidance scales)

Same script as Fig 3 (`fig_sdxl_guidance.py` prints table to stdout).
Data:     `experiments/sdxl/results/traj_avg_*.json` (tracked)
          `experiments/sdxl/results/bootstrap_traj_*.json` (tracked, 58 files)

---

**Table 2** (alanine lag sensitivity — DA correlations across τ)
```
Table:    cd experiments/alanine && python table_alanine_lag_sensitivity.py
```
Producer: same as Fig 5 (`run_alanine_C_tau.sh`)
Data:     `experiments/alanine/results/alanine_C_tau_full_h_half_10k.npz` (sync)
          Requires raw alanine data in `alanine_dipeptide/data/`

---

**Table 3** (alanine baselines — sparse K-means, spectral map, diffusion maps)
```
Table:    cd experiments/alanine && python table_alanine_baselines.py
```
Producer: `sbatch experiments/alanine/cluster/run_baselines.sh`
          (CPU-only, ~10 min each baseline)
Data:     `experiments/alanine/results/sparse_kmeans.json` (tracked)
          `experiments/alanine/results/spectral_map.json` (tracked)
          `experiments/alanine/results/diffusion_maps.json` (tracked)

---

**Table 4** (Bonferroni — eigenvalue significance at τ*)
```
Table:    cd experiments/alanine && python table_alanine_bonferroni.py
```
Producer: `sbatch experiments/alanine/cluster/run_alanine_null.sh`
          (catfish, 1× L4, 32G mem, 1h time limit, array 0-9; ~8 min/task)
Data:     `experiments/alanine/results/alanine_C_tau_full_h_half_10k.npz` (sync)
          `experiments/alanine/results/alanine_null_lag2k_task*.npz` (sync, 10 files)

---

**Table App H** (weight comparability across guidance scales)
```
Table:    cd experiments/sdxl && python table_weights.py
```
Producer: `sbatch experiments/sdxl/cluster/run_weights.sh`
          (catfish, 1× L4, 32G mem, 2h time limit)
Data:     `experiments/sdxl/results/weight_table.json` (tracked)

---

**Bootstrap SSA CIs** (14 prompts × 4 scales, B=2000)
```
Table:    cd experiments/sdxl && python bootstrap_shat_all14.py
```
Producer: reads per-cell SSA JSONs (already produced by guidance sweep)
Data:     `experiments/sdxl/results/clip_da_*_dt0p005.json` (tracked)
Output:   `experiments/sdxl/results/results_shat14_bootstrap.csv` (tracked)
Runtime:  fast (CPU-only, pure NumPy resampling)

---

## Additional Producers (not tied to a single figure)

**SDXL trajectory-averaged pipeline** (3 modes):
```bash
# Mode 1: Recompute SSA (GPU required)
cd experiments/sdxl && python traj_averaged_pipeline.py \
    --checkpoint results/langevin_ckpt_*.npz --dt 0.005 --name my_cell --output_dir results

# Mode 2: Bootstrap CI (CPU-only)
cd experiments/sdxl && python traj_averaged_pipeline.py --mode bootstrap \
    --clip_traj results/z_clip_traj_*.npz --name my_cell --B 2000

# Mode 3: DA significance (CPU-only)
cd experiments/sdxl && python traj_averaged_pipeline.py --mode da_significance \
    --clip_traj results/z_clip_traj_*.npz --dt 0.0025 --T 2000 --name elderly_gs7p5
```
SLURM: `sbatch experiments/sdxl/cluster/run_traj_averaged.sh`
       (salmon, 1× L40S, 48G mem, 12h time limit, array 0-23)

**SDXL MC null** (anisotropic Gaussian null for DA significance):
```
cd experiments/sdxl && python run_sdxl_mc_null.py --task_id 0
```
SLURM: `sbatch experiments/sdxl/cluster/run_sdxl_null.sh`
       (salmon, 1× L40S, 48G mem, 2-day time limit, array 0-2; ~19h/replicate)

**SDXL C(0) eigenvalues** (for spectrum figure):
```
cd experiments/sdxl && python save_c0_eigvals_elderly.py
```
SLURM: `sbatch experiments/sdxl/cluster/run_save_c0_eigvals_elderly.sh`
       (salmon, 1× L40S, 48G mem)

**Crane PC1/DA1 extremes** (image grids):
```
cd experiments/sdxl && python fig_crane_extremes.py
```
SLURM: `sbatch experiments/sdxl/cluster/run_decode_crane.sh`
       (salmon, 1× L40S, 32G mem, 30 min time limit)
Requires GPU (VAE decode).

---

## Syncing Results from Cluster

For binary NPZ files marked "sync" above:

```bash
# Set up SSH tunnel
ssh -N -L 2222:moriah-gw:22 shaulytolk@bava.cs.huji.ac.il

# Sync all result files
rsync -avz -e 'ssh -p 2222' \
  shaulytolk@localhost:/sci/labs/orzuk/shaulytolk/ABM/experiments/synthetic/results/ \
  experiments/synthetic/results/

rsync -avz -e 'ssh -p 2222' \
  shaulytolk@localhost:/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl/results/ \
  experiments/sdxl/results/

rsync -avz -e 'ssh -p 2222' \
  shaulytolk@localhost:/sci/labs/orzuk/shaulytolk/ABM/experiments/alanine/results/ \
  experiments/alanine/results/
```

All small JSON/CSV result files (~1.5 MB total) are tracked in git.
Binary NPZ files are gitignored; sync from cluster or regenerate via SLURM.
