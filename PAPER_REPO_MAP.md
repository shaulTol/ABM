# Paper → Repo Mapping

Self-contained reference mapping every figure, table, and experiment in the NeurIPS 2026 paper
to actual repo files: generation scripts, data sources, cluster results, and SLURM scripts.

**Last updated**: 2026-05-06 (post-restructure)

---

## Main Text Figures

### Figure 1 — GMM Parameter Sweeps (Section 3.1)
**File**: `figures/fig_gmm_sweeps_curves.png` (+ `.pdf`)
**Generator**: `figures/make_fig_gmm_sweeps_curves_10d.py`
**Data**: `experiments/ssa_truncation_redef/results/gmm_10d_n200_all_sweeps_T200.json`
**Compute**: `experiments/ssa_truncation_redef/code/` (cluster, job 44584375)
**SLURM**: `experiments/ssa_truncation_redef/cluster/`
**Content**: 1×4 dual-axis panels — Separation, Variance, Weight asymmetry, Mode count. SSA (blue, left) vs MI (red, right). d=10, N=200, 10 seeds, 1/N floor (5e-4), T_phys=200s.
**Regen**: `python figures/make_fig_gmm_sweeps_curves_10d.py`
**Status**: Ready
**Modified**: 2026-04-30

### Figure 2 — Dispersion vs Fragmentation + Parallel Ellipses (Section 3.1.2 + 3.1.3)
Combined figure (two subfigures in LaTeX):

#### (a) Dispersion vs Fragmentation
**File**: `figures/fig_dispersion_vs_fragmentation.png` (+ `.pdf`)
**Generator**: `figures/make_fig_dispersion_vs_fragmentation.py` (standalone) or `figures/regenerate_all.py` → `make_fig_dispersion_vs_fragmentation()`
**Data**: Computed on-the-fly (2D GMM Langevin + analytical entropy). Also saves `figures/dispersion_rho_dt01.csv`.
**Content**: Two 2D scatter panels — (left) unimodal Gaussian blob (high H, low SSA), (right) parallel-ellipses GMM with DA1/PC1 arrows (low H, high SSA). SSA/H annotation boxes.
**SLURM**: `figures/cluster/run_dispersion_fig.sh` (GPU for Langevin, or runs locally with CPU)
**Regen**: `python figures/make_fig_dispersion_vs_fragmentation.py`
**Status**: Ready
**Modified**: 2026-04-30

#### (b) Parallel Ellipses
**File**: `figures/fig_da_parallel_ellipses_2panel.png` (+ `.pdf`)
**Generator**: `figures/regenerate_all.py` → `make_fig_da_parallel_ellipses()` or `figures/make_fig_da_parallel_ellipses.py`
**Data**: Computed on-the-fly (3000-point scatter, seed=42)
**Content**: 2D scatter with DA1 (vertical, blue) and PC1 (horizontal, red) arrows.
**Regen**: `python figures/regenerate_all.py --only ellipses`
**Status**: Ready

### Figure 3 — SDXL SSA vs Entropy (Section 4.2)
**File**: `figures/fig_sdxl_ssa_vs_entropy.png`
**Generator**: `experiments/sdxl/guidance_sweep/plot_multiprompt_sweep.py`
**Data**: `experiments/sdxl/guidance_sweep/results_multiprompt_sweep/clip_da_mp_p{0-7}_{label}_gs{1p0,3p0,7p5,15p0}.json` (8 prompts × 4 scales = 32 JSONs, repatched with Ŝ)
**Compute**: `experiments/sdxl/guidance_sweep/run_ssa_da_clip.py`
**SLURM**: `experiments/sdxl/guidance_sweep/cluster/run_multiprompt_sweep_ssa.sh` (array job, salmon L40S)
**Content**: Dual-axis chart — SSA (blue, left) peaks at w=7.5 for 5/8 prompts, kNN entropy (red, right) monotonically decreases. SSA ≠ entropy.
**Cluster results**: `/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl/guidance_sweep/results_multiprompt_sweep/`
**Status**: Ready
**Modified**: 2026-04-30

### Table 1 — Multi-Prompt Guidance Sweep (Section 4.2)
**Data**: Same 32 JSONs as Figure 3 + `experiments/sdxl/guidance_sweep/results_multiprompt_sweep/gen_mp_p{0-7}_{label}_gs{scale}_entropy.json`
**Compute**: `experiments/sdxl/guidance_sweep/run_ssa_da_clip.py` + `experiments/sdxl/guidance_sweep/recompute_entropy.py`
**Content**: 8 prompts × 4 guidance scales. SSA (Ŝ with free-conv floor), entropy (kNN). Person prompt added (8th). dt=0.005, T=2000, N=200.
**Prompts**:
| idx | label     | prompt                                              |
|-----|-----------|-----------------------------------------------------|
| 0   | cat       | a photograph of a cat sitting on a windowsill       |
| 1   | mountains | a landscape photograph of mountains at sunset       |
| 2   | flowers   | a still life painting of flowers in a vase          |
| 3   | street    | a street photograph of a busy city intersection     |
| 4   | food      | a photograph of a plate of food at a restaurant     |
| 5   | elderly   | a portrait of an elderly person                     |
| 6   | car       | a photograph of a car parked on a street            |
| 7   | person    | a photograph of a person                            |
**Status**: Ready

### Figure 4 — Elderly DA1 vs PC1 Image Grid (Section 4.2)
**File**: `figures/fig_elderly_da_pca_compact.png` (+ `.pdf`)
**Generator**: `experiments/sdxl/elderly_da/fig_elderly_da_vs_pca_freeconv.py` (requires GPU for VAE decode + CLIP)
**Data**:
  - DA results: `experiments/sdxl/elderly_da/results_da_pcavda/clip_da_elderly_gs7p5_fc_dt0p0025.{json,npz}`
  - Latents: `experiments/sdxl/guidance_sweep/results_multiprompt_sweep/gen_mp_p5_elderly_gs7p5_quick_x0.npz`
  - Indices: `experiments/sdxl/elderly_da/results_da_pcavda/fig_elderly_da_vs_pca_freeconv_taustar_indices.npz`
**Compute**: `experiments/sdxl/guidance_sweep/run_ssa_da_clip.py` with free-conv stopping
**SLURM**: `experiments/sdxl/elderly_da/cluster/run_da_elderly_freeconv.sh` (salmon, job 44596109)
**Figure gen SLURM**: `experiments/sdxl/elderly_da/cluster/run_fig_elderly_da_pca_freeconv.sh`
**Content**: 4 rows × 5 columns — DA1 high, DA1 low, PC1 high, PC1 low. DA1=gender (ρ=0.911 peak / 0.900 at τ*), |cos(DA1,PC1)|=0.514.
**Bootstrap** (B=2000, DA recomputed per replicate, sign-aligned): |ρ|=0.90 [95% CI: 0.78, 0.94], SE=0.046, P(|ρ|>0.8)=0.936. Median gap λ₁/λ₂=1.80. Results: `experiments/sdxl/elderly_da/bootstrap/results.json`.
**Cluster results**: `/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl/elderly_da/results_da_pcavda/`
**Status**: Ready
**Modified**: 2026-04-29

### Figure 5 — Alanine DA vs TICA Ramachandran (Section 4.3)
**File**: `figures/fig_alanine_da_tica.png` (+ `.pdf`)
**Generator**: `figures/regenerate_all.py` → `make_fig_alanine_da_tica()`
**Data**:
  - C(τ) at τ=10,000ps: `experiments/save_autocovariance/results/alanine_C_tau_full_h_half_10k.npz` (38 lags, h=h_Scott/2=0.0091)
  - TICA/dihedral data: `experiments/alanine/main/results/alanine_da_results_gmm_only.npz`
  - Distance data: downloaded via `mdshare` (alanine-dipeptide-3x250ns-heavy-atom-distances.npz)
**Compute**: `experiments/save_autocovariance/save_alanine_C_tau_h_half_10k.py` (salmon, ~62 min)
**Content**: Three Ramachandran (ψ vs φ) scatter panels colored by DA1, TICA1, PC1 projections. h=h_Scott/2, τ=10,000ps. DA1|r(ψ)|=0.672±0.001, DA2|r(φ)|=0.873±0.000.
**Regen**: `python figures/regenerate_all.py --only alanine`
**Status**: Ready
**Modified**: 2026-04-30

### Table 2 — Alanine DA vs TICA Correlations (Section 4.3)
**Data**: Computed from `experiments/save_autocovariance/results/alanine_C_tau_full_h_half_10k.npz`
**CSV**: `figures/table_alanine_lag_sensitivity_h_half.csv` (all 38 lags, full metrics)
**Also**: `figures/table_alanine_rho_vs_tau_h_half.csv`, `figures/table_alanine_lam1_ratio_vs_tau_h_half.csv`
**Content**: |r(φ)|, |r(ψ)| for DA1/DA2 vs TICA1. DA1 tracks ψ, stable from τ=700ps to 10,000ps (<2° rotation). Bootstrap B=5000.
**Status**: Ready

---

## Appendix Figures

### Figure A1 — GMM Distribution Snapshots (4×5 grid)
**File**: `figures/fig_gmm_sweeps_snapshots.png` (+ `.pdf`)
**Generator**: `figures/regenerate_all.py` → `make_fig_gmm_sweeps_snapshots()`
**Data**: Computed on-the-fly (scipy multivariate_normal PDFs)
**Content**: 4 sweep types × 5 parameter values. Blue density heatmaps with contour overlay.
**Regen**: `python figures/regenerate_all.py --only snapshots`
**Status**: Ready

### Figure A2 — Random GMM Sweep (SSA rank vs MI rank)
**File**: `figures/fig_random_sweep.png` (+ `.pdf`)
**Generator**: `figures/make_fig_random_sweep_10d.py`
**Data**: `experiments/ssa_truncation_redef/results/random_sweep_10d_n200.json`
**Content**: Rank scatter for 50 random 10D bimodal GMMs, N=200, 1/N floor. Spearman ρ=0.890.
**Regen**: `python figures/make_fig_random_sweep_10d.py`
**Status**: Ready
**Modified**: 2026-04-27

### Figure A3 — Δt-Robustness Table/Figure
**File**: `experiments/dt_robustness/fig_dt_robustness.png`
**Generator**: `experiments/dt_robustness/plot_dt_comparison.py`
**Data**: `experiments/ssa_truncation_redef/results/gmm_10d_n200_dt_robustness.json` (dt=0.005), `gmm_10d_n200_all_sweeps.json` (dt=0.01), `gmm_10d_dt_robust_dt0p02_T200.json` (dt=0.02)
**Content**: dt ∈ {0.005, 0.01, 0.02} (4× range). All Spearman ≥ 0.97. Max relative diff 7.8%.
**Status**: Ready

### Figure A4 — Alanine Bulk Validation (eigenvalue spectrum at τ=10,000ps)
**File**: `figures/fig_alanine_bulk_validation.png`
**Generator**: `experiments/save_autocovariance/make_fig_alanine_bulk_validation.py`
**Data**: `experiments/save_autocovariance/results/alanine_C_tau_full_h_half_10k.npz`
**Content**: Symlog eigenvalue spectrum with free-conv floor (λ₊=1.97e-4) and C(0) MP edge (1.22e-3). 2 spikes above floor at τ=10,000ps.
**Status**: Ready

### Figure A5 — Alanine Bulk Validation at τ=0 (C(0) MP baseline)
**File**: `figures/fig_alanine_bulk_validation_tau0.png`
**Generator**: `experiments/save_autocovariance/make_fig_alanine_bulk_validation.py`
**Data**: Same as A4
**Content**: C(0) eigenvalue spectrum with Bai-Yin floor overlay.
**Status**: Ready

### Figure A6 — Alanine Spike Count vs τ
**File**: `figures/fig_alanine_spike_count_vs_tau.png`
**Generator**: `experiments/save_autocovariance/diagnostic_taustar_negspikes.py` (or inline)
**Data**: `experiments/save_autocovariance/results/alanine_C_tau_full_h_half_10k.npz`
**Content**: Number of eigenvalues above free-conv floor vs lag τ. 2 spikes stable from τ≥942ps, transient jump to 3 at τ=7444ps.
**Status**: Ready

### Figure A7 — Elderly Eigenvalue Spectrum (bulk at τ*)
**File**: `figures/fig_elderly_spectrum.png`
**Generator**: `experiments/save_autocovariance/make_fig_sdxl_elderly_bulk_dt0p0025.py`
**Data**: `experiments/save_autocovariance/results/elderly_gs7p5_n200_dt0p005_C_tau_full.npz` (or dt=0.0025 variant)
**Content**: SDXL elderly eigenvalue bulk spectrum with free-conv floor at τ*=2000.
**Status**: Ready

### Figure A8 — Elderly DA1/DA2 Projection Histograms
**File**: `figures/fig_elderly_projection_histograms.png` (TODO: needs standalone script)
**Data**:
  - DA1 eigvec: `experiments/sdxl/elderly_da/results_da_pcavda/clip_da_elderly_gs7p5_fc_dt0p0025.npz`
  - CLIP embeddings (N=2000): `experiments/sdxl/guidance_sweep/results_multiprompt_sweep/gen_mp_p5_elderly_gs7p5_clip.npz`
**Content**: DA1 histogram (bimodal, Silverman p<0.002) + DA2 histogram (unimodal, Silverman p=0.77). N=2000 CLIP projections.
**Status**: Ready (no standalone .py script — created inline)

### Figure A9 — Parallel Ellipses DA Convergence (across lags)
**File**: `figures/fig_da_parallel_ellipses_convergence.png`
**Generator**: `figures/make_fig_da_parallel_ellipses_convergence.py`
**Data**: Computed on-the-fly
**Status**: Ready

### Figure A10 — Free-Convolution Density Validation
**File**: Generated by `experiments/mp_generalization/plot_free_conv_validation.py`
**Generator**: `experiments/mp_generalization/plot_free_conv_validation.py`
**Compute**: `experiments/mp_generalization/run_free_conv_overlay.py`, `run_mp_generalization.py`
**Data**: `experiments/mp_generalization/results/`
**Content**: Empirical eigenvalue histogram vs free-conv analytic density overlay. Validates ρ² noise floor formula.
**Status**: Ready

### Figure A11 — Marchenko-Pastur Baseline (4-panel)
**File**: `figures/fig_mp_baseline.png`
**Generator**: `experiments/noise_floor/mp_figures/generate_mp_figures.py`
**Content**: MP law density for γ = 0.2, 0.5, 1.0, 2.0.
**Status**: Ready

### Figure A12 — Eigenvalue Floor (Bai-Yin overlay)
**File**: `figures/fig_mp_eigenvalue_floor.png`
**Generator**: `experiments/noise_floor/mp_figures/generate_mp_figures.py`
**Content**: 8-panel spectra with analytic Bai-Yin floor (1+√γ)².
**Status**: Ready

### Figure A13 — Eigenvalue Trajectories (λ₁ across lags)
**File**: `figures/fig_mp_eigenvalue_trajectories.png`
**Generator**: `experiments/noise_floor/mp_figures/generate_mp_figures.py`
**Content**: λ₁ trajectories with floor overlay across lags.
**Status**: Ready

### Figure A14 — TICA Implied Timescales
**File**: `figures/fig_tica_its.png`
**Generator**: `experiments/alanine/main/make_tica_its_figure.py`
**Data**: `experiments/alanine/main/alanine_dipeptide/data/alanine-dipeptide-3x250ns-heavy-atom-distances.npz` (via mdshare)
**Status**: Ready

### Figure A15 — Planted Spikes (synthetic validation)
**File**: Generated in `experiments/synthetic_spikes/`
**Generator**: `experiments/synthetic_spikes/simulate_planted_spikes.py`
**Variants**: `simulate_planted_spikes_variants.py`, `simulate_planted_spikes_nonuniform.py`, `simulate_planted_spikes_varying_rate.py`
**Spike count figure**: `experiments/synthetic_spikes/plot_spike_count_per_direction.py`
**SLURM**: `experiments/synthetic_spikes/cluster/`
**Content**: Synthetic ground-truth spikes planted in d-dimensional GMM. Validates DA spike counting against known number of metastable directions.
**Status**: Ready

### Figure A16 — Alanine Baselines (Sparse K-Means + Spectral Map)
**File**: Generated in `experiments/alanine_baselines_sparse_kmeans_spectralmap/`
**Content**: Sparse K-means w* (|r(ψ)|=0.332, |r(φ)|=0.793) and Spectral Map (|r(ψ)|=0.044, |r(φ)|=0.752) both fail to recover ψ (slow CV). DA1 and TICA1 succeed.
**Status**: Ready (gitignored folder, self-contained venv)

### SDXL Guidance Grids (visual diversity)
**Files**: `fig_sdxl_guidance_grid_1.png`, `fig_sdxl_guidance_grid_2.png` (if included)
**Data**: `experiments/sdxl/guidance_sweep/results_multiprompt_sweep/gen_mp_p{0-7}_{label}_gs{scale}_quick_x0.npz`
**Generator**: `experiments/sdxl/guidance_sweep/decode_all_grids.py` + `run_decode_multiprompt_sweep.sh`
**Content**: Image grids showing visual diversity at each guidance level per prompt.
**Status**: Ready (requires cluster for VAE decode)

### SDXL Extended Prompts Sweep (6 ambiguous nouns × 4 guidance)
**Dir**: `experiments/sdxl/extended_prompts/`
**Script**: `sanity_checks.py` (post-processing), `cluster/run_ext_gen.sh`, `run_ext_quick.sh`, `run_ext_ssa.sh`
**Prompts**: bat, crane, mouse, seal, jaguar, palm (all "a photo of a ___" — semantically ambiguous nouns with animal/object duality)
**Config**: 6 prompts × 4 guidance scales (1.0, 3.0, 7.5, 15.0), N=200, dt=0.005, 1/N floor
**Output**: `clip_da_ext_p{0-5}_{label}_{gs}_dt0p005.json`, `extended_sweep_results.json`, LaTeX tables
**Status**: Scripts ready, cluster jobs pending submission (`cluster/submit_all.sh`)

### Elderly DA1 Bootstrap CI
**Dir**: `experiments/sdxl/elderly_da/bootstrap/`
**Script**: `run_bootstrap.py`
**Data**: `clip_da_elderly_gs7p5_fc_z_tau.npz` (CLIP trajectory matrix from canonical free-conv run)
**Results**: `results.json`, `bootstrap_histogram.png`
**Key numbers**: |ρ|=0.90 [95% CI: 0.78, 0.94], SE=0.046, B=2000, P(|ρ|>0.8)=0.936, median gap λ₁/λ₂=1.80
**Status**: Complete

### Alanine DA Bonferroni Significance Audit
**Dir**: `experiments/alanine_da_bonferroni/`
**Content**: 12-phase forensic audit of DA2 significance under permutation null. Uncovered 10× lag mismatch (step vs ps units) in earlier session's conclusions.
**Key finding**: DA1 PASSES at all lags (200, 2000, 10000 ps). **DA2 FAILS at every correctly matched lag** (p≈0.53 at 10,000ps). φ mode's autocovariance indistinguishable from isotropic Gaussian null.
**Files**: `PHASE_0_AUDIT.md` through `PHASE_12_pca_scree.png` (read-only audit reports + diagnostic plots)
**Status**: Complete (audit only, no code changes)

### Alanine Diffusion Maps Baseline
**Dir**: `experiments/alanine/baselines/diffusion_maps/`
**Script**: `run_diffusion_maps.py`
**Results**: `results/results.json`, `PAPER_NUMBERS.md`
**Key numbers**: ψ₂|r(ψ)|=0.615±0.023, ψ₃|r(φ)|=0.659±0.020 (5 seeds, N=2000, α=1, ε*=0.0655). Correct ordering (ψ₂↔ψ slow, ψ₃↔φ fast) but weaker correlations than DA (0.672) and TICA.
**Status**: Complete

---

## CUT from Paper (completed but not included)

### Chignolin DA vs TICA
**File**: `figures/fig_chignolin_da_tica.png`
**Generator**: `experiments/alanine/main/make_paper_figures.py`
**Data**: `experiments/alanine/main/results/chignolin_da_results.npz`
**Result**: DA1: |r(dRMSD)|=0.976, TICA1: 0.847. Good result, cut for space.

### MNIST DDPM (Section 4.4)
**Files**: `figures/fig_mnist_ssa_entropy_bars.png`, `fig_mnist_clusters_compact.png`, `fig_mnist_digit1_da1_pca1_heatmaps.png`, `fig_mnist_digit1_gap_profile.png`
**Generators**: `figures/make_fig_mnist_ssa_entropy_bars.py`, `figures/make_fig_mnist_clusters_compact.py`, `experiments/mnist/scripts/make_paper_figures.py`
**Data**: `experiments/mnist/results/ssa_digit_{0-9}.json`, `experiments/mnist/results_da/da_digit_{0-9}.npz`, `experiments/mnist/data/entropy_results.csv`
**Result**: SSA range 0.667–0.873 (digit 1 outlier), KDE correlation ρ=0.842. Cut from paper.

### SDXL Square DA vs PCA
**Files**: `figures/fig_sdxl_da_pca.png`, `fig_sdxl_da1_sorted.png`, `fig_sdxl_pc1_sorted.png`, `fig_sdxl_scribble.png`
**Data**: `experiments/sdxl/results_autoresearch/clip_da_autoresearch_square_T2500.{json,npz}`
**Scribble**: `experiments/sdxl/guidance_sweep/make_autoresearch_scribbles.py` → `make_face_boxed()`
**Result**: DA1=gender (ρ=0.905), DA1⊥PC1 (|cos|=0.29). SDXL-Turbo + ControlNet. Cut for space; elderly figure replaces it.

---

## Key Data Dependencies

### Pre-computed data (checked into repo via JSON)
- `experiments/ssa_truncation_redef/results/gmm_10d_n200_all_sweeps_T200.json` — 4 sweep configs, d=10, N=200, 10 seeds
- `experiments/ssa_truncation_redef/results/random_sweep_10d_n200.json` — 50 random 10D GMMs
- `experiments/ssa_truncation_redef/results/gmm_10d_n200_dt_robustness.json` — dt=0.005 variant
- `experiments/ssa_truncation_redef/results/gmm_10d_dt_robust_dt0p02_T200.json` — dt=0.02 variant
- `experiments/alanine/main/results/alanine_da_results_gmm_only.npz` — alanine GMM-DA (gitignored .npz)
- `experiments/mnist/results/` — SSA per digit (gitignored)
- `experiments/mnist/data/entropy_results.csv` — kNN entropy

### Cluster-generated data (synced via rsync, gitignored .npz/.json)
- `experiments/sdxl/guidance_sweep/results_multiprompt_sweep/` — 8-prompt guidance sweep (32 JSONs + x0 latents + CLIP embeddings)
- `experiments/sdxl/elderly_da/results_da_pcavda/` — DA vs PCA results (elderly canonical: `clip_da_elderly_gs7p5_fc_dt0p0025.*`)
- `experiments/save_autocovariance/results/` — Full C(τ) matrices (`alanine_C_tau_full_h_half_10k.npz`, elderly bulk)
- `experiments/noise_floor/` — noise floor JSONs per dataset
- `experiments/mp_generalization/results/` — free-conv validation data

**Cluster path**: `/sci/labs/orzuk/shaulytolk/ABM/experiments/`
**Sync command**: `rsync -avz -e 'ssh -p 2222' shaulytolk@localhost:/sci/labs/orzuk/shaulytolk/ABM/experiments/<subdir>/results* experiments/<subdir>/`

### External data (not in repo)
- Alanine dipeptide distances: `alanine-dipeptide-3x250ns-heavy-atom-distances.npz` (mdshare, auto-downloaded)
- MNIST: torchvision (auto-downloaded)
- SDXL Base: `stabilityai/stable-diffusion-xl-base-1.0` (HuggingFace)
- CLIP: `openai/clip-vit-large-patch14` (768D)

---

## Regeneration Commands

### All synthetic figures (no GPU needed)
```bash
python figures/regenerate_all.py
```
Generates: `fig_random_sweep`, `fig_gmm_sweeps_curves`, `fig_dispersion_vs_fragmentation`, `fig_da_parallel_ellipses_2panel`, `fig_gmm_sweeps_snapshots`, `fig_alanine_da_tica`

**Note**: `regenerate_all.py` wraps older function calls. Some figures now have standalone scripts that are more current:
- `make_fig_gmm_sweeps_curves_10d.py` — canonical 10D version (replaces `regenerate_all.py --only sweeps`)
- `make_fig_random_sweep_10d.py` — canonical 10D version (replaces `regenerate_all.py --only random`)
- `make_fig_dispersion_vs_fragmentation.py` — standalone with dt=0.01 (replaces `regenerate_all.py --only dispersion`)

### Individual figures
```bash
python figures/make_fig_gmm_sweeps_curves_10d.py     # Fig 1 (10D sweeps)
python figures/make_fig_random_sweep_10d.py           # Fig A2 (random GMMs)
python figures/make_fig_dispersion_vs_fragmentation.py # Fig 2a (dispersion)
python figures/regenerate_all.py --only ellipses      # Fig 2b (parallel ellipses)
python figures/regenerate_all.py --only snapshots     # Fig A1 (GMM snapshots)
python figures/regenerate_all.py --only alanine       # Fig 5 (alanine DA/TICA)
```

### Bulk validation figures
```bash
python experiments/save_autocovariance/make_fig_alanine_bulk_validation.py  # Fig A4-A5
python experiments/save_autocovariance/make_fig_sdxl_elderly_bulk_dt0p0025.py # Fig A7
```

### Noise floor figures
```bash
python experiments/noise_floor/mp_figures/generate_mp_figures.py  # Fig A11-A13
python experiments/mp_generalization/plot_free_conv_validation.py  # Fig A10
```

### SDXL figures (GPU required, run on cluster)
```bash
python experiments/sdxl/elderly_da/fig_elderly_da_vs_pca_freeconv.py  # Fig 4 (elderly grid)
python experiments/sdxl/guidance_sweep/plot_multiprompt_sweep.py           # Fig 3 (SSA vs entropy)
```

---

## Core Library (`src/`)

| Module | Purpose | Used by |
|--------|---------|---------|
| `src/metrics/autocorrelation.py` | `compute_vector_autocorrelation()`, `aggregate_ssa()` (Ŝ Riemann sum + floor truncation) | All SSA computations |
| `src/simulation/langevin.py` | `LangevinSimulatorD`, `simulate_d_trajectories()` | GMM sweeps, dispersion |
| `src/metrics/da_directions.py` | `extract_da_directions()` — eigendecompose C(τ) for DA | DA figures |
| `src/kde/kde_score.py` | `KDEScore` — KDE-based score estimation | Alanine KDE-DA |
| `experiments/noise_floor/free_conv_utils.py` | `fc_edges()`, free-conv ρ² floor | Floor truncation everywhere |

---

## Style + Formatting
- **Config**: `figures/style_config.py`
- **Colors**: SSA = `#2166ac` (blue), MI = `#b2182b` (red), Scatter = `#404040`, Diagonal = `#999999`
- **Width**: `TEXT_WIDTH = 5.5` inches (NeurIPS single column)
- **Font**: Times New Roman / DejaVu Serif, size 9
- **DPI**: 300 for savefig
- **Format**: PNG + PDF for all main figures

---

## Paper → Figure Quick Reference

| Paper ref | File in `figures/` | Generator | Status |
|-----------|-------------------|-----------|--------|
| Fig 1 | `fig_gmm_sweeps_curves.png` | `make_fig_gmm_sweeps_curves_10d.py` | Ready |
| Fig 2a | `fig_dispersion_vs_fragmentation.png` | `make_fig_dispersion_vs_fragmentation.py` | Ready |
| Fig 2b | `fig_da_parallel_ellipses_2panel.png` | `regenerate_all.py --only ellipses` | Ready |
| Table 1 | (LaTeX, from SDXL JSONs) | `plot_multiprompt_sweep.py` | Ready |
| Fig 3 | `fig_sdxl_ssa_vs_entropy.png` | `plot_multiprompt_sweep.py` | Ready |
| Fig 4 | `fig_elderly_da_pca_compact.png` | `fig_elderly_da_vs_pca_freeconv.py` | Ready |
| Fig 5 | `fig_alanine_da_tica.png` | `regenerate_all.py --only alanine` | Ready |
| Table 2 | `table_alanine_lag_sensitivity_h_half.csv` | `regenerate_all.py` | Ready |
| App A1 | `fig_gmm_sweeps_snapshots.png` | `regenerate_all.py --only snapshots` | Ready |
| App A2 | `fig_random_sweep.png` | `make_fig_random_sweep_10d.py` | Ready |
| App A3 | `fig_dt_robustness.png` | `dt_robustness/plot_dt_comparison.py` | Ready |
| App A4 | `fig_alanine_bulk_validation.png` | `make_fig_alanine_bulk_validation.py` | Ready |
| App A5 | `fig_alanine_bulk_validation_tau0.png` | `make_fig_alanine_bulk_validation.py` | Ready |
| App A6 | `fig_alanine_spike_count_vs_tau.png` | (inline / diagnostic script) | Ready |
| App A7 | `fig_elderly_spectrum.png` | `make_fig_sdxl_elderly_bulk_dt0p0025.py` | Ready |
| App A8 | `fig_elderly_projection_histograms.png` | (no standalone script) | Ready* |
| App A9 | `fig_da_parallel_ellipses_convergence.png` | `make_fig_da_parallel_ellipses_convergence.py` | Ready |
| App A10 | (free-conv validation) | `plot_free_conv_validation.py` | Ready |
| App A11 | `fig_mp_baseline.png` | `generate_mp_figures.py` | Ready |
| App A12 | `fig_mp_eigenvalue_floor.png` | `generate_mp_figures.py` | Ready |
| App A13 | `fig_mp_eigenvalue_trajectories.png` | `generate_mp_figures.py` | Ready |
| App A14 | `fig_tica_its.png` | `make_tica_its_figure.py` | Ready |
| App A15 | (planted spikes) | `simulate_planted_spikes.py` | Ready |
| App A16 | (alanine baselines) | (self-contained in gitignored folder) | Ready |
| App grids | `fig_sdxl_guidance_grid_{1,2}.png` | `decode_all_grids.py` | Ready |

---

## Known Duplicates

1. **`regenerate_all.py` vs standalone `make_fig_*.py` scripts**: `regenerate_all.py` contains older versions of several figure functions. The standalone scripts (`make_fig_gmm_sweeps_curves_10d.py`, `make_fig_random_sweep_10d.py`, `make_fig_dispersion_vs_fragmentation.py`) are more current and should be canonical. `regenerate_all.py` still works for `--only ellipses`, `--only snapshots`, `--only alanine`.

2. **Old 2D GMM data vs new 10D**: `figures/data/{sep_2d,var_2d,weight_2d,nmodes_fixdist_2d}_n5000_data.json` and `experiments/gmm_sweep/results/` contain old 2D N=5000/10000 sweep results. Paper now uses 10D N=200 data from `experiments/ssa_truncation_redef/results/`. Old data is unused.

3. **`fig_elderly_da_vs_pca.py` vs `fig_elderly_da_vs_pca_freeconv.py`**: The `_freeconv` version is canonical (uses free-conv stopping run dt=0.0025). The non-freeconv version uses older N=200 data without free-conv stopping.

4. **`make_fig_gmm_sweeps_split.py`**: Generates `fig_10d_separation.png` and `fig_10d_variance.png`. These are superseded by `make_fig_gmm_sweeps_curves_10d.py` which produces the combined 4-panel figure. The split version may still be useful for appendix.

5. **SDXL results directories**: `results_guidance_sweep/` (single-prompt, old), `results_multiprompt_sweep/` (8-prompt, canonical), `results_da_pcavda/` (DA), `results_autoresearch/` (Square, cut), `results_clip_da/` (old CLIP DA), `results_T5000/` (incomplete backup). Only `results_multiprompt_sweep/` and `results_da_pcavda/` are used in the paper.

---

## Orphans

### Scripts not mapping to any current paper figure
- `experiments/sdxl/guidance_sweep/compute_sdxl_ssa_v2.py` — old SSA computation, replaced by `run_ssa_da_clip.py`
- `experiments/sdxl/guidance_sweep/compare_circle_vs_face.py` — ControlNet comparison (cut)
- `experiments/sdxl/guidance_sweep/debug_gender_ranking.py` — one-off debugging
- `experiments/sdxl/guidance_sweep/debug_latentclip_validation.py` — Latent-CLIP validation (method abandoned)
- `experiments/sdxl/guidance_sweep/interpret_latentclip_pca.py` — Latent-CLIP (abandoned)
- `experiments/sdxl/guidance_sweep/validate_latentclip_probes.py` — Latent-CLIP (abandoned)
- `experiments/sdxl/guidance_sweep/plot_dps_comparison.py`, `plot_dps_scatter.py` — DPS experiments (cut)
- `experiments/sdxl/guidance_sweep/noise_denoise_turbo.py` — noise/denoise diagnostic
- `experiments/sdxl/guidance_sweep/compute_square_direction.py` — Square mean-CLIP direction (cut)
- `experiments/sdxl/guidance_sweep/visualize_da.py`, `visualize_results.py`, `visualize_clip_da_sweep.py` — old visualization
- `figures/make_fig_a0_identity.py` — A₀=σ²I vs A₀=Σ_f comparison figure (related to Cov_f experiment, not in paper)
- `experiments/mc_dip_validation/scan_exact_score_ou.py`, `scan_exact_score_ou_B1000.py`, `scan_iso_null_decay.py` — MC null diagnostics (not figures)
- `experiments/save_autocovariance/save_alanine_C_tau.py` — original h_Scott version (superseded by h_half variants)
- `experiments/save_autocovariance/make_fig_sdxl_elderly_bulk.py` — old elderly bulk (superseded by dt0p0025 version)
- `experiments/save_autocovariance/make_fig_bulk_scott_vs_h_half.py` — bandwidth comparison diagnostic
- `experiments/save_autocovariance/n_scaling_sweep.py` — N-scaling diagnostic
- `experiments/save_autocovariance/save_alanine_C_tau_h_half_50k.py` — 50k grid variant (unused)

### SLURM scripts for cut/abandoned experiments (~60 of 89 in `experiments/sdxl/guidance_sweep/cluster/`)
- `run_controlnet_*.sh` (8 scripts) — ControlNet exploration
- `run_ssa_da_v9*.sh` (8 scripts) — fashionable animal experiment
- `run_square_*.sh` (3 scripts) — Square experiment (cut)
- `run_da_street_nobox*.sh` (2 scripts) — Nobox experiment (cut)
- `run_dps_*.sh` (3 scripts) — DPS experiments (cut)
- `run_T5000_*.sh` (4 scripts) — T=5000 incomplete backup
- `run_debug_*.sh` (3 scripts) — debugging
- Various one-off scripts: `run_base_vs_turbo.sh`, `run_special_dog.sh`, `run_male_female.sh`, etc.

### Experiment directories (gitignored, not in paper)
- `experiments/integrated_da/` — Failed: Int-DA1≈PC1
- `covf_replication/` — Failed: Cov_f diffusion breaks weight/nmodes rankings
- `experiments_whitening/` — Tested: whitening breaks weight/nmodes rankings, not in paper
- `experiments/autoresearch/` — Square/Egg/V9/V11 experiments (cut)
- `experiments/alanine_burnin_ctau/` — Burn-in test (no improvement, canonical is no-burn-in)

---

## TODOs

1. **`fig_elderly_projection_histograms.png`** — No standalone generation script. Created inline in a conversation. Should create `figures/make_fig_elderly_projection_histograms.py` for reproducibility. Uses N=2000 CLIP embeddings from `gen_mp_p5_elderly_gs7p5_clip.npz`.

2. **`regenerate_all.py` stale references** — Still references old 2D data paths (`sep_2d`, `var_2d`, etc.) in some code paths. The `--only sweeps` and `--only random` flags call old functions. Should either update to call standalone 10D scripts or remove those code paths.

3. **`fig_alanine_spike_count_vs_tau.png`** — No clear standalone generator script. May have been generated by `diagnostic_taustar_negspikes.py` or inline.

4. **Planted spikes figure** — Output location unclear (generated in `experiments/synthetic_spikes/` not in `figures/`). Need to verify which PNG is the paper figure and copy to `figures/`.

5. **Alanine baselines figure** — Lives in gitignored `experiments/alanine_baselines_sparse_kmeans_spectralmap/`. Need to decide whether to commit the figure (not the venv) or reference it differently.

6. **Free-conv validation figure** — Output from `experiments/mp_generalization/plot_free_conv_validation.py` goes to `experiments/mp_generalization/results/`, not `figures/`. Need to verify filename and consider copying to `figures/`.

7. **SDXL guidance grid figures** — Generation process unclear (VAE decode on cluster). Document exact SLURM + post-processing steps.

8. **Clean up ~60 orphan SLURM scripts** in `experiments/sdxl/guidance_sweep/cluster/` that are for cut/abandoned experiments.

9. **Clean up orphan Python scripts** (~15 files in `experiments/sdxl/guidance_sweep/` for Latent-CLIP, DPS, debugging, old visualization).

10. **Old 2D GMM data** in `figures/data/` and `experiments/gmm_sweep/results/` — no longer used. Consider removing or archiving.
