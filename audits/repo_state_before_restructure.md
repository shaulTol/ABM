# Repository State — Pre-Restructure Snapshot

**Date**: 2026-05-06
**Branch**: `main`
**Purpose**: Complete inventory before any restructuring. Navigation aid for going back.

---

## Size Summary

| Component | Size | Notes |
|-----------|------|-------|
| `experiments/sdxl/` | **15 GB** | ~12 GB dead (ControlNet, DPS, old results) |
| `experiments/autoresearch/` | **5.9 GB** | All cut from paper, gitignored |
| `experiments/alanine_baselines_*/` | 787 MB | 750 MB is self-contained venv |
| `experiments/save_autocovariance/` | 561 MB | C(τ) matrices — all needed |
| `experiments/da_directions/` | 470 MB | Alanine trajectory data + results |
| `experiments_whitening/` | 329 MB | Failed experiment, gitignored |
| `experiments/alanine_pca_panel/` | 211 MB | Has own venv (200 MB) |
| `experiments/ssa_truncation_redef/` | 182 MB | GMM sweeps, Ŝ redefinition |
| `figures/` | 157 MB | ~80 MB SDXL image grids |
| Everything else | < 100 MB combined | |
| **Total repo** | **~24 GB** | |

---

## experiments/ Directory Inventory

### Paper-Critical Directories

| # | Directory | Size | Description | Paper Content |
|---|-----------|------|-------------|---------------|
| 1 | `sdxl/results_traj_averaged/` | 49 MB | Trajectory-averaged SSA + bootstrap | **Tables 1, 6** (SSA values + SEs), **Fig 20** (crane extremes) |
| 2 | `sdxl/results_multiprompt_sweep/` | 1.7 GB | 8-prompt guidance sweep: x0 latents, CLIP embeddings, entropy JSONs | **Fig 3** (SSA vs entropy), **Table 1/6** (entropy), Fig 4 (x0 latents) |
| 3 | `sdxl/results_da_pcavda/` | 1.5 GB | DA vs PCA analysis (elderly canonical) | **Fig 4** (elderly DA grid), elderly DA eigenvalues |
| 4 | `save_autocovariance/` | 561 MB | Full C(τ) matrices for alanine (h/2 canonical) + SDXL elderly | **Fig 5** (alanine DA/TICA), **Table 2** (correlations), **Table 4** (lag sensitivity), bulk validation figs |
| 5 | `ssa_truncation_redef/` | 182 MB | Ŝ redefinition: GMM sweeps d=10 N=200, random sweep, dt robustness, trajectory bootstrap | **Fig 1** (GMM sweeps), **Fig A2** (random), **Table 3** (mode-count bootstrap), **Table 10** (Δt Spearman) |
| 6 | `da_directions/` | 470 MB | DA vs TICA on alanine + chignolin | **Fig 5** data (TICA comparison), alanine KDE-DA, chignolin (cut) |
| 7 | `sdxl_extended_prompts/` | 24 MB | 6 polysemous nouns × 4 guidance scales | **Table 6** (extended prompts SSA + entropy) |
| 8 | `sdxl_weight_comparability/` | 64 KB | w₂ weight ratios + convergence proxy | **Tables 7, 8** (weight comparability, ρ²·N) |
| 9 | `mc_dip_validation/` | 11 MB | MC null + permutation + dip tests | **Table 5** (alanine null), **Table 9** (elderly null) |
| 10 | `noise_floor/` | 192 KB | Free-convolution floor formula + per-dataset validation | `free_conv_utils.py` used everywhere for floor truncation |
| 11 | `mp_generalization/` | 48 KB | Marchenko-Pastur density generalization | Free-conv density validation figures (appendix) |
| 12 | `synthetic_spikes/` | 1.8 MB | Planted-spike validation: uniform + varying rate | Planted spikes figure + spike-count-per-direction (appendix) |
| 13 | `robustness/` | 820 KB | Bootstrap SEs (task1), lag sensitivity (task2), repeatability (task3) | Bootstrap CIs for SDXL, alanine lag sensitivity |
| 14 | `dt_robustness/` | 392 KB | Δt sensitivity study: dt ∈ {0.005, 0.01, 0.02} | Δt robustness appendix figure |
| 15 | `elderly_da_bootstrap/` | 1.3 MB | DA1 bootstrap B=2000 | DA1 |ρ|=0.90 [0.78, 0.94], SE=0.046 |
| 16 | `alanine_baselines_sparse_kmeans_spectralmap/` | 787 MB | Sparse K-means + Spectral Map baselines (self-contained venv) | **Table 2** (SM, sparse K-means entries) |
| 17 | `alanine/diffusion_maps/` | 1.1 MB | Diffusion Maps baseline | **Table 2** (DM entries) |
| 18 | `alanine_da_bonferroni/` | 672 KB | 12-phase DA2 significance audit (markdown + diagnostic plots) | DA2 fails permutation null (p≈0.53) — informs paper's m̂=2→m̂=1 discussion |
| 19 | `sdxl_3prompt/` | 40 KB | Original 3-prompt revision (cat/elderly/person) | Reference for expanded 14-prompt analysis |
| 20 | `gmm_sweep/` | 148 KB | Original GMM sweep code + configs | Sweep config builders (used by ssa_truncation_redef) |

### Dead / Cut / Gitignored Directories

| # | Directory | Size | Description | Why Dead |
|---|-----------|------|-------------|----------|
| 21 | `autoresearch/` | **5.9 GB** | Square, Egg, V9, V11 ControlNet experiments | All cut from paper. Good results (Square DA1=gender) but not included. |
| 22 | `mnist/` | 23 MB | DDPM on MNIST digits | Cut from paper. SSA 0.667–0.873, KDE ρ=0.842. |
| 23 | `beans/` | 5.1 MB | Dry Bean dataset DA | Cut. DA1 finds BOMBAY (gap=19.08). |
| 24 | `california_housing/` | 324 KB | Housing dataset DA vs PCA | Cut. Exploratory. |
| 25 | `wine_quality/` | 1.9 MB | Wine quality DA | Cut. Showed slow mixing ≠ bimodality. |
| 26 | `integrated_da/` | 232 KB | Integrated DA (mean C(τ) across lags) | **Failed**: Int-DA1 ≈ PC1 (88.7° off). |
| 27 | `marchenko_pastur/` | 168 KB | MP visualization only | Superseded by mp_generalization + free_conv_utils. |
| 28 | `peeled_rho/` | 3.0 MB | Eigenvalue peeling across 5 case studies | Support/diagnostic. Not directly in paper. |
| 29 | `alanine_pca_panel/` | 211 MB | PCA panel figure (has own 200 MB venv) | Unclear if still used. |

### Top-Level Dead Directories

| Directory | Size | Description | Why Dead |
|-----------|------|-------------|----------|
| `experiments_whitening/` | 329 MB | Whitening-variant experiments | **Failed**: Rankings destroyed for weight/nmodes, DA1 breaks on alanine. Gitignored. |
| `covf_replication/` | 560 KB | Full-covariance A₀=Σ_f diffusion | **Failed**: Dispersion≡unimodal, flattened sensitivity, numerical instability. Gitignored. |
| `results/` | 18 MB | Old top-level results (unclear provenance) | Likely stale. |
| `results_multiprompt_sweep/` | 72 KB | Old top-level multiprompt results | Stale copy — canonical is under experiments/sdxl/. |

---

## Dead Weight Inside experiments/sdxl/ (12 GB)

| Subdirectory | Size | Description | Why Dead |
|--------------|------|-------------|----------|
| `results_clip_da/` | **4.7 GB** | Old CLIP-DA runs | Superseded by `results_da_pcavda/` |
| `results/` | **2.8 GB** | Old guidance_sweep + base_vs_turbo | Superseded by `results_multiprompt_sweep/` |
| `optimize_bimodal/` | **1.4 GB** | Abandoned bimodality optimization | Never completed |
| `results_autoresearch/` | 448 MB | Square experiment results | Cut from paper |
| `dps_zeta5_run2/` | 172 MB | DPS experiment | Cut from paper |
| `output_circle_200/` | 137 MB | Circle ControlNet decoded images | Cut |
| `output_dps_guided_200/` | 123 MB | DPS guided decoded images | Cut |
| `results_guidance_sweep/` | 100 MB | Old single-prompt guidance sweep | Superseded by multiprompt |
| `results_street_nocn/` | 80 MB | Street without ControlNet | Cut |
| `output_controlnet_black_generic/` | 76 MB | ControlNet variant | Cut |
| `output_controlnet_rubin_face_cs05/` | 74 MB | ControlNet Rubin face | Cut |
| `output_controlnet_black_man_from/` | 73 MB | ControlNet variant | Cut |
| `output_controlnet_circle_turbo_mammal/` | 72 MB | ControlNet circle turbo | Cut |
| `output_controlnet_black_man/` | 71 MB | ControlNet variant | Cut |
| `output_controlnet_rubin_noprompt/` | 70 MB | ControlNet Rubin | Cut |
| `output_controlnet_black_person/` | 70 MB | ControlNet variant | Cut |
| `output_controlnet_black_immigrant/` | 69 MB | ControlNet variant | Cut |
| `output_controlnet_black_man_ancestors/` | 67 MB | ControlNet variant | Cut |
| `output_controlnet_rubin_kissing/` | 66 MB | ControlNet Rubin | Cut |
| `output_controlnet_black_woman_ancestors/` | 66 MB | ControlNet variant | Cut |
| `output_controlnet_black_person_ancestors/` | 66 MB | ControlNet variant | Cut |
| `results_street_nobox/` | 65 MB | Street without box | Cut |
| `output_controlnet_rubin_turbo_cs05/` | 65 MB | ControlNet Rubin turbo | Cut |
| `output_controlnet_circle_turbo_100/` | 65 MB | ControlNet circle | Cut |
| `output_controlnet_black_woman/` | 65 MB | ControlNet variant | Cut |
| `output_controlnet_rubin_couple_cs05/` | 63 MB | ControlNet Rubin | Cut |
| `output_controlnet_face_age_cs05/` | 63 MB | ControlNet face age | Cut |
| `output_controlnet_rubin_turbo/` | 62 MB | ControlNet Rubin turbo | Cut |
| `output_controlnet_circle_turbo_human/` | 62 MB | ControlNet circle | Cut |
| `results_controlnet/` | 61 MB | ControlNet results | Cut |
| `output_controlnet_rubin_couple_cs08/` | 61 MB | ControlNet Rubin | Cut |
| `output_controlnet_face_person/` | 60 MB | ControlNet face | Cut |
| `output_controlnet_circle_turbo_living/` | 57 MB | ControlNet circle | Cut |
| `output_controlnet_face_androgynous/` | 56 MB | ControlNet androgynous | Cut |
| `results_male_female/` | 22 MB | Male/female one-off | Cut |
| `output_noise_denoise_*` (4 dirs) | 49 MB | Noise/denoise diagnostics | Cut |
| `debug_decode/` | 6.9 MB | VAE decode debugging | One-off diagnostic |
| **TOTAL DEAD** | **~12 GB** | | |

---

## Orphan Scripts: experiments/sdxl/scripts/ (42 Python files)

### Paper-Critical Scripts (5 files)

| Script | Purpose | Paper Content |
|--------|---------|---------------|
| `run_ssa_da_clip.py` | Full SSA + DA pipeline in CLIP space | All SDXL SSA/DA results |
| `sdxl_components.py` | Reusable: RealCLIPEmbedder, compute_ssa_da, TextProbeAnalyzer | Imported by run_ssa_da_clip.py |
| `fig_elderly_da_vs_pca_freeconv.py` | Elderly DA1 vs PC1 image grid (free-conv stopping) | **Fig 4** |
| `generate_and_entropy.py` | Standalone generation + CLIP + entropy | Entropy computation |
| `plot_multiprompt_sweep.py` | Multi-prompt guidance sweep chart | **Fig 3** |

### Reference Scripts — Do Not Modify (1 file)

| Script | Purpose | Status |
|--------|---------|--------|
| `compute_sdxl_ssa_v2.py` | Original LC-mode SSA pipeline | Frozen. Old Latent-CLIP code. Never modify. |

### Utility Scripts — May Be Useful (6 files)

| Script | Purpose |
|--------|---------|
| `decode_all_grids.py` | Decode latents to image grids |
| `recompute_entropy.py` | Recompute kNN entropy from saved latents |
| `extract_clip_projections.py` | Extract CLIP embeddings, project onto DA1/PC1 |
| `gen_quick_samples.py` | Quick sample generation from SDXL |
| `save_c0_eigvals_elderly.py` | Compute C(0) eigenvalues for elderly |
| `decode_person_hires.py` | Decode person images at full resolution |

### Dead Scripts — Cut/Abandoned (30 files)

| Script | Purpose | Why Dead |
|--------|---------|----------|
| `visualize_results.py` | Plot old SSA results | Superseded by plot_multiprompt_sweep.py |
| `visualize_da.py` | Decode DA extremes | Superseded by fig_elderly_da_vs_pca_freeconv.py |
| `preprocess_control.py` | ControlNet scribble extraction | ControlNet experiments cut |
| `noise_denoise_turbo.py` | Noise/denoise diagnostic | Cut |
| `plot_latent_clip_pca.py` | Latent-CLIP PCA thumbnails | Latent-CLIP abandoned |
| `compare_clip_latentclip_pca.py` | CLIP vs Latent-CLIP PCA | Latent-CLIP abandoned |
| `validate_latentclip_probes.py` | Latent-CLIP text probe validation | Latent-CLIP abandoned |
| `debug_latentclip_validation.py` | Latent-CLIP debugging | Latent-CLIP abandoned |
| `visualize_clip_da_sweep.py` | Old CLIP/LC comparison | Superseded |
| `compare_circle_vs_face.py` | Circle vs DPS face comparison | Cut |
| `interpret_da_clip.py` | LC-DA text probe interpretation | Latent-CLIP abandoned |
| `make_autoresearch_scribbles.py` | Generate scribble images | Autoresearch cut |
| `make_pca_grids.py` | PCA grid generation | Autoresearch cut |
| `full_correlation_table.py` | Full correlation matrix | Diagnostic only |
| `compute_entropy_comparison.py` | Old entropy comparison | Superseded by recompute_entropy.py |
| `plot_dps_comparison.py` | DPS scribble comparison | DPS cut |
| `plot_dps_scatter.py` | DPS scatter plots | DPS cut |
| `plot_correlation_tables.py` | Correlation table PDF | Presentation aid |
| `plot_da_vs_pca_grids.py` | Old DA vs PCA grids | Superseded by fig_elderly_da_vs_pca_freeconv.py |
| `plot_da_pca_all_prompts.py` | DA1 vs PC1 for all prompts | Exploratory |
| `fig_elderly_da_vs_pca.py` | Old elderly figure (no free-conv) | Superseded by _freeconv version |
| `compute_square_direction.py` | Square CLIP direction | Square cut |
| `debug_gender_ranking.py` | Male/female classification debug | One-off |
| `generate_controlnet_samples.py` | ControlNet sample generation | ControlNet cut |
| `interpret_latentclip_pca.py` | Latent-CLIP PCA interpretation | Latent-CLIP abandoned |
| `make_pc_scatter.py` | PC thumbnail scatter | Exploratory |
| `plot_clip_gender.py` | CLIP male/female axis | One-off |
| `plot_clip_pca.py` | CLIP PCA thumbnails | Exploratory |
| `plot_clip_pca_compare.py` | Two-set CLIP PCA comparison | Exploratory |
| `plot_ssa_comparison.py` | Old SSA comparison PDF | Superseded |

---

## Orphan SLURM Scripts: experiments/sdxl/cluster/ (92 .sh files)

### Paper-Critical (14 files)

| Script | Purpose |
|--------|---------|
| `run_guidance_sweep_gen.sh` | Generate SDXL samples for 8 prompts |
| `run_guidance_sweep_quick.sh` | Quick guidance sweep |
| `run_guidance_sweep_ssa.sh` | SSA computation for guidance sweep |
| `run_multiprompt_sweep_quick.sh` | Quick multi-prompt sweep |
| `run_multiprompt_sweep_gen.sh` | Multi-prompt generation |
| `run_multiprompt_sweep_ssa.sh` | Multi-prompt SSA |
| `run_decode_multiprompt_sweep.sh` | Decode multi-prompt images |
| `run_da_elderly_freeconv.sh` | DA elderly with free-conv stopping (job 44596109) |
| `run_da_elderly_freeconv_catfish.sh` | Same, catfish partition |
| `run_fig_elderly_da_pca_freeconv.sh` | Generate Fig 4 on cluster |
| `run_fig_elderly_da_pca_freeconv_taustar.sh` | Fig 4 with τ* selection |
| `run_save_c0_eigvals_elderly.sh` | Save C(0) eigenvalues |
| `run_recompute_entropy.sh` | Recompute entropy |
| `run_decode_person_hires.sh` | Decode person hi-res |

### Dead — Cut/Abandoned (78 files)

**ControlNet** (8): `run_controlnet_ssa.sh`, `_ssa_black.sh`, `_gen_black.sh`, `_ssa_dps_long.sh`, `_gen_batch.sh`, `_ssa_compare.sh`, `_gen.sh`, `_gen_turbo.sh`

**V9/Fashionable Animal** (10): `run_ssa_da_v9.sh`, `_v9_T500_dt001.sh`, `_v9_T500_dt001_v2.sh`, `_v9_T500_dt005.sh`, `_v9_T500_dt005_v2.sh`, `_v9_T500_dt01.sh`, `_v9_T500_dt01_v2.sh`, `_v9_dt001.sh`, `_v9_dt005.sh`, `_v9_resume.sh`

**Square** (3): `run_square_N1000_T1500.sh`, `_N1000_T2500.sh`, `_N2000_T1500.sh`

**Street/Nobox** (4): `run_da_street_nobox.sh`, `_nobox_T2500.sh`, `_square_T2500.sh`, `run_gen_street_no_cn.sh`

**DPS** (3): `run_dps_comparison.sh`, `_scatter.sh`, `_converged.sh`

**T5000/Resume** (4): `run_T5000_dt0p0025.sh`, `_dt0p005.sh`, `_resume.sh`, `_resume_catfish.sh`

**Latent-CLIP** (5): `run_latent_clip_pca.sh`, `_compare_clip_pca.sh`, `_validate_lc.sh`, `_debug_lc.sh`, `_interpret_lc_pca.sh`

**Old DA variants** (8): `run_da_elderly_person.sh`, `_n1000.sh`, `_all_prompts_n200.sh`, `_all_prompts_n2000.sh`, `_elderly_n1000.sh`, `_pca_grids.sh`, `_plot_da_pca.sh`, `_extract_projections.sh`

**Miscellaneous** (14): `run_base_vs_turbo.sh`, `run_dt_test.sh`, `_sweep_face.sh`, `_guidance_clip.sh`, `_guidance_clip_v2.sh`, `_clip_da.sh`, `_clip_da_sweep.sh`, `_clip_da_compare.sh`, `_clip_da_dt.sh`, `_corr_tables.sh`, `_corr_orth.sh`, `_autoresearch_da.sh`, `_pca_grids.sh`, `_decode_grids.sh`

**One-off** (9): `_interpret_da.sh`, `_male_female.sh`, `_special_dog.sh`, `_square_direction.sh`, `_debug_gender.sh`, `_pc_scatter.sh`, `_noise_denoise.sh`, `_reembed_n200.sh`, `_fig_elderly.sh`, `_ssa_v27_v27c.sh`, `_mp_dt0p005_T5000.sh`, `_gen_street_face_nobox.sh`, `_dps_report.sh`

---

## src/ Module Status

All modules actively used. No stale code.

| Module | Purpose | Primary Users |
|--------|---------|---------------|
| `metrics/autocorrelation.py` | Autocorrelation + `aggregate_ssa()` (Ŝ Riemann sum + floor truncation) | All SSA computations |
| `metrics/da_directions.py` | `extract_da_directions()` — eigendecompose C(τ) | All DA extractions |
| `simulation/langevin.py` | `LangevinSimulatorD`, `simulate_d_trajectories()` | GMM sweeps, dispersion |
| `simulation/generic_langevin.py` | Arbitrary score function Langevin simulator | KDE-based alanine simulations |
| `core/gaussian_mixture.py` | N-dimensional GMM with GPU acceleration | GMM sweep configs |
| `core/device_utils.py` | Memory management, tensor ops | GPU computations |
| `core/gpu_manager.py` | GPU detection, batch optimization | GPU computations |
| `kde/kde_estimator.py` | KDE → GMM conversion | Alanine KDE-DA |
| `kde/kde_score.py` | `KDEScore` — ∇log f(x) via KDE | Alanine simulations |
| `slicing/projection_strategies.py` | Projection sampling strategies | Sliced SSA |
| `slicing/sliced_autocorrelation.py` | 1D sliced SSA on random projections | Used by gmm_sweep |
| `slicing/autocorsanitycheck.py` | Sanity check from full trajectories | Low-use but not dead |

---

## figures/ Contents

### Paper Figure Generators (16 scripts)

| Script | Output | Paper Figure |
|--------|--------|-------------|
| `make_fig_gmm_sweeps_curves_10d.py` | `fig_gmm_sweeps_curves.png` | **Fig 1** |
| `make_fig_dispersion_vs_fragmentation.py` | `fig_dispersion_vs_fragmentation.png` | **Fig 2a** |
| `make_fig_da_parallel_ellipses.py` | `fig_da_parallel_ellipses_2panel.png` | **Fig 2b** |
| `make_fig_random_sweep_10d.py` | `fig_random_sweep.png` | **Fig A2** |
| `make_fig_gmm_sweeps_split.py` | `fig_10d_separation.png`, `fig_10d_variance.png` | Split sweep panels |
| `make_fig_alanine_bulk_peeled.py` | `fig_alanine_bulk_peeled.png` | Bulk histogram |
| `make_fig_alanine_eigenvalue_rank.py` | `fig_alanine_eigenvalue_rank.png` | Eigenvalue rank |
| `make_fig_alanine_bulk_validation_unified.py` | `fig_alanine_bulk_validation_tau*.png` (3) | Bulk validation at 3 lags |
| `make_fig_elderly_spectrum_unified.py` | `fig_elderly_spectrum.png` | **Fig A7** |
| `make_fig_free_conv_validation_unified.py` | `fig_free_conv_validation.png` (+ 6 per-γ) | **Fig A10** |
| `make_fig_free_conv_validation_body.py` | `fig_free_conv_validation_body.png` | Body-text version |
| `make_fig_sdxl_combined.py` | `fig_sdxl_combined.png` | Combined SDXL |
| `make_fig_gmm_combined.py` | `fig_gmm_combined.png` | Combined GMM |
| `make_fig_a0_identity.py` | `fig_a0_identity.png` | A₀ validation (not in paper) |
| `make_fig_da_parallel_ellipses_convergence.py` | `fig_da_parallel_ellipses_convergence.png` | **Fig A9** |
| `regenerate_all.py` | Multiple (wraps older functions) | Master regeneration |

**Also**: `style_config.py` (Matplotlib styling), `_spike_fig_style.py` (spike figure colors).

### Generator Scripts in experiments/ (for paper figures)

| Script | Output → figures/ | Paper Figure |
|--------|-------------------|-------------|
| `sdxl/scripts/fig_elderly_da_vs_pca_freeconv.py` | `fig_elderly_da_pca_compact.png` | **Fig 4** |
| `sdxl/scripts/plot_multiprompt_sweep.py` | `fig_sdxl_ssa_vs_entropy.png` | **Fig 3** |
| `save_autocovariance/make_fig_alanine_bulk_validation.py` | `fig_alanine_bulk_validation.png` | **Figs A4–A5** |
| `save_autocovariance/make_fig_sdxl_elderly_bulk_dt0p0025.py` | `fig_elderly_spectrum.png` | **Fig A7** |
| `save_autocovariance/diagnostic_taustar_negspikes.py` | Diagnostic plots | N/A |
| `dt_robustness/plot_dt_comparison.py` | `fig_dt_robustness.png` | **Fig A3** |
| `da_directions/make_tica_its_figure.py` | `fig_tica_its.png` | **Fig A14** |
| `noise_floor/mp_figures/generate_mp_figures.py` | `fig_mp_baseline.png`, etc. | **Figs A11–A13** |
| `mp_generalization/plot_free_conv_validation.py` | Free-conv overlay | **Fig A10** (alt) |
| `synthetic_spikes/simulate_planted_spikes.py` | `fig_planted_spikes.png` | **Fig A15** |
| `alanine_baselines_*/make_figure.py` | `fig_alanine_baselines_*.png` | **Fig A16** |
| `sdxl/results_traj_averaged/decode_crane_extremes.py` | `crane_*_extremes.png` | **Fig 20** |
| `da_directions/make_alanine_da_tica_noisefloor.py` | Alanine noise floor fig | Appendix |

### Large Image Files (> 1 MB)

| File | Size | Purpose |
|------|------|---------|
| `fig_sdxl_gs1p0_all.png` | 20 MB | SDXL gs=1.0 full grid |
| `fig_sdxl_gs3p0_all.png` | 17 MB | SDXL gs=3.0 full grid |
| `fig_sdxl_gs7p5_all.png` | 16 MB | SDXL gs=7.5 full grid |
| `fig_sdxl_gs15p0_all.png` | 15 MB | SDXL gs=15.0 full grid |
| `fig_sdxl_elderly_da_pc1.png` | 3.9 MB | Elderly DA1 vs PC1 |
| `fig_sdxl_da1_sorted.png` | 3.4 MB | DA1 sorted images |
| `fig_sdxl_pc1_sorted.png` | 3.4 MB | PC1 sorted images |
| `fig_sdxl_person_grid.png` | 3.3 MB | Person prompt grid |
| `fig_sdxl_guidance_grid.png` | 2.5 MB | Guidance sweep grid |
| `person_grid_contact_sheet.png` | 2.5 MB | Person grid contact sheet |
| `fig_alanine_da_tica.png` | 2.3 MB | Alanine DA vs TICA |
| `fig_elderly_da_pca_compact.png` | 2.3 MB | Elderly DA vs PCA compact |
| `fig_crane_pc1_extremes.png` | 2.1 MB | Crane PC1 extremes |
| `fig_crane_da1_extremes.png` | 1.9 MB | Crane DA1 extremes |
| `fig_elderly_da_vs_pca.png` | 1.5 MB | Elderly DA vs PCA |

### Data Files in figures/

| File | Purpose |
|------|---------|
| `figures/data/*.json` (9 files) | Old 2D GMM sweep data (N=1000, N=5000) — **unused**, superseded by 10D |
| `figures/data/a0_identity_values.json` | A₀ validation data — not in paper |
| `figures/data/dispersion_values.json` | Dispersion figure data |
| `figures/dispersion_rho_dt01.csv` | Dispersion robustness |
| `figures/table_alanine_lag_sensitivity_h_half.csv` | **Table 4** data |
| `figures/table_alanine_rho_vs_tau_h_half.csv` | Table 2 correlations |
| `figures/table_alanine_lam1_ratio_vs_tau_h_half.csv` | Eigenvalue ratios |
| `figures/table_alanine_bonferroni.csv` | Bonferroni test results |

---

## Paper Figure/Table → Data Source Complete Cross-Reference

### Main Text

| Paper Item | Data Source | Generator |
|------------|------------|-----------|
| **Fig 1** (GMM sweeps) | `ssa_truncation_redef/results/gmm_10d_n200_all_sweeps_T200.json` | `figures/make_fig_gmm_sweeps_curves_10d.py` |
| **Fig 2a** (dispersion) | On-the-fly (2D GMM) | `figures/make_fig_dispersion_vs_fragmentation.py` |
| **Fig 2b** (ellipses) | On-the-fly (3000 points) | `figures/make_fig_da_parallel_ellipses.py` |
| **Fig 3** (SSA vs entropy) | `sdxl/results_multiprompt_sweep/*.json` | `sdxl/scripts/plot_multiprompt_sweep.py` |
| **Fig 4** (elderly DA grid) | `sdxl/results_da_pcavda/clip_da_elderly_gs7p5_fc_dt0p0025.{json,npz}` | `sdxl/scripts/fig_elderly_da_vs_pca_freeconv.py` |
| **Fig 5** (alanine DA/TICA) | `save_autocovariance/results/alanine_C_tau_full_h_half_10k.npz` | `figures/regenerate_all.py --only alanine` |
| **Table 1** (SSA, 11 prompts) | `sdxl/results_traj_averaged/traj_avg_*.json` + `bootstrap_*.json` | LaTeX (manual) |
| **Table 2** (alanine corrs) | `save_autocovariance/results/alanine_C_tau_full_h_half_10k.npz` + DM/SM results | LaTeX (manual) |

### Appendix Tables

| Paper Item | Data Source |
|------------|------------|
| **Table 3** (mode-count) | `ssa_truncation_redef/results/nmodes_traj_bootstrap.npz` |
| **Table 4** (lag sensitivity) | `figures/table_alanine_lag_sensitivity_h_half.csv` |
| **Table 5** (alanine null) | `mc_dip_validation/results/exact_score_ou_null.npz` |
| **Table 6** (full SSA+entropy) | `sdxl/results_traj_averaged/traj_avg_*.json` + `bootstrap_*.json` + entropy JSONs |
| **Table 7** (w₂ weights) | `sdxl_weight_comparability/weight_table.json` |
| **Table 8** (convergence proxy) | `sdxl_weight_comparability/weight_table.json` (`rho_T_N` field) |
| **Table 9** (elderly null) | `mc_dip_validation/sdxl_iso_null_via_late_t/results/sdxl_anisotropic_null_pvalues.json` |
| **Table 10** (Δt robustness) | `ssa_truncation_redef/results/gmm_10d_n200_dt_robustness.json` + `gmm_10d_dt_robust_dt0p02_T200.json` |
| **Table 12** (licenses) | Documentation only |

### Appendix Figures

| Paper Item | Data Source | Generator |
|------------|------------|-----------|
| **Fig A1** (GMM snapshots) | On-the-fly | `figures/regenerate_all.py --only snapshots` |
| **Fig A2** (random sweep) | `ssa_truncation_redef/results/random_sweep_10d_n200.json` | `figures/make_fig_random_sweep_10d.py` |
| **Fig A3** (Δt robustness) | `ssa_truncation_redef/results/gmm_10d_n200_dt_robustness.json` | `dt_robustness/plot_dt_comparison.py` |
| **Figs A4–A5** (alanine bulk) | `save_autocovariance/results/alanine_C_tau_full_h_half_10k.npz` | `save_autocovariance/make_fig_alanine_bulk_validation.py` |
| **Fig A6** (alanine spike count) | Same NPZ | **⚠️ NO STANDALONE SCRIPT** |
| **Fig A7** (elderly spectrum) | `save_autocovariance/results/elderly_gs7p5_n200_*_C_tau_full.npz` | `figures/make_fig_elderly_spectrum_unified.py` |
| **Fig A8** (elderly histograms) | `sdxl/results_da_pcavda/clip_da_elderly_gs7p5_fc_dt0p0025.npz` | **⚠️ NO STANDALONE SCRIPT** |
| **Fig A9** (ellipses convergence) | On-the-fly | `figures/make_fig_da_parallel_ellipses_convergence.py` |
| **Fig A10** (free-conv validation) | On-the-fly | `figures/make_fig_free_conv_validation_unified.py` ✅ |
| **Figs A11–A13** (MP) | `mp_generalization/results/` | `noise_floor/mp_figures/generate_mp_figures.py` |
| **Fig A14** (TICA ITS) | Alanine trajectory data | `da_directions/make_tica_its_figure.py` |
| **Fig A15** (planted spikes) | On-the-fly | `synthetic_spikes/simulate_planted_spikes.py` (output in experiments/, not figures/) |
| **Fig A16** (alanine baselines) | `alanine_baselines_*/results/*.npz` | `alanine_baselines_*/make_figure.py` (output in experiments/, not figures/) |
| **Fig 20** (crane extremes) | `sdxl/results_traj_averaged/z_clip_traj_crane_*.npz` | `sdxl/results_traj_averaged/decode_crane_extremes.py` |

---

## Missing Figure Generators — Assessment

| # | Figure | Status | Action Needed |
|---|--------|--------|---------------|
| 1 | `fig_elderly_projection_histograms.png` (Fig A8) | **Needs writing from scratch** | No script generates DA1/DA2 projection histograms. Created inline in a conversation session. |
| 2 | `fig_alanine_spike_count_vs_tau.png` (Fig A6) | **Needs writing from scratch** | No script generates spike count vs τ curve. A related `fig_n_spikes_vs_tau_scott_h2.png` exists in save_autocovariance/results/ but for wrong bandwidth. |
| 3 | Planted spikes (Fig A15) | **Script exists, needs plumbing** | `simulate_planted_spikes.py` generates `fig_planted_spikes.png` in experiments/synthetic_spikes/. Needs copy/symlink to figures/. |
| 4 | Free-conv validation (Fig A10) | **Already complete** | `figures/make_fig_free_conv_validation_unified.py` exists and outputs to figures/. Not actually missing. |
| 5 | Alanine baselines (Fig A16) | **Script exists, needs plumbing** | `alanine_baselines_*/make_figure.py` generates `fig_alanine_baselines_*.png` in experiments/results/. Needs copy/symlink to figures/. |

**Summary**: 2 scripts genuinely need writing (elderly histograms, alanine spike count). 2 exist but output to experiments/ instead of figures/. 1 was a false alarm (already complete in figures/).

---

## .gitignore Coverage

Key exclusions (from `.gitignore`):
- `__pycache__/`, `*.pyc`, `.DS_Store`
- All binary outputs: `*.npz`, `*.png`, `*.jpg`, `*.pdf`, `*.pt`, `*.pth`, `*.ckpt`
- `outputs/`, `data/MNIST/`
- `.claude/`, `CLAUDE.md`
- `integrated_da/`, `covf_replication/`, `experiments_whitening/`
- `ssa_truncation_redef/` (partially), `autoresearch/`
- Various experiment result directories
- `docs/`, `PAPER_REPO_MAP.md`

**Note**: JSON files in experiments/ are NOT gitignored and form the primary data layer for paper tables.
