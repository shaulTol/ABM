# Repository Restructure Mapping

**Date**: 2026-05-06
**Branch**: `repo-cleanup` (branched from `main` at `pre-restructure-snapshot`)

---

## Summary

- Moved ~18 GB of dead/cut/failed experiments to `older_versions/` (gitignored)
- Reorganized `experiments/sdxl/` into logical subgroups
- Consolidated alanine experiments under `experiments/alanine/`
- Kept theory/GMM/cross-cutting dirs flat

---

## 1. Dead Weight → older_versions/

### Top-Level Directories

| Old Path | New Path | Size | Reason |
|----------|----------|------|--------|
| `experiments_whitening/` | `older_versions/experiments_whitening/` | 329 MB | Failed experiment |
| `covf_replication/` | `older_versions/covf_replication/` | 560 KB | Failed experiment |
| `results/` | `older_versions/results/` | 18 MB | Stale top-level results |
| `results_multiprompt_sweep/` | `older_versions/results_multiprompt_sweep/` | 72 KB | Stale copy |

### Dead Experiment Directories

| Old Path | New Path | Size | Reason |
|----------|----------|------|--------|
| `experiments/mnist/` | `older_versions/experiments/mnist/` | 23 MB | Cut from paper |
| `experiments/beans/` | `older_versions/experiments/beans/` | 5.1 MB | Cut from paper |
| `experiments/california_housing/` | `older_versions/experiments/california_housing/` | 324 KB | Cut from paper |
| `experiments/wine_quality/` | `older_versions/experiments/wine_quality/` | 1.9 MB | Cut from paper |
| `experiments/integrated_da/` | `older_versions/experiments/integrated_da/` | 232 KB | Failed experiment |
| `experiments/marchenko_pastur/` | `older_versions/experiments/marchenko_pastur/` | 168 KB | Superseded |
| `experiments/peeled_rho/` | `older_versions/experiments/peeled_rho/` | 3 MB | Diagnostic only |
| `experiments/alanine_pca_panel/` | `older_versions/experiments/alanine_pca_panel/` | 211 MB | Unclear if used |
| `experiments/autoresearch/` | `older_versions/experiments/autoresearch/` | 5.9 GB | Cut from paper |

### Dead SDXL Subdirectories (~12 GB)

| Old Path | New Path | Size | Reason |
|----------|----------|------|--------|
| `experiments/sdxl/results_clip_da/` | `older_versions/experiments/sdxl/results_clip_da/` | 4.7 GB | Superseded by results_da_pcavda |
| `experiments/sdxl/results/` | `older_versions/experiments/sdxl/results/` | 2.8 GB | Superseded by results_multiprompt_sweep |
| `experiments/sdxl/optimize_bimodal/` | `older_versions/experiments/sdxl/optimize_bimodal/` | 1.4 GB | Abandoned |
| `experiments/sdxl/results_autoresearch/` | `older_versions/experiments/sdxl/results_autoresearch/` | 448 MB | Cut from paper |
| `experiments/sdxl/dps_zeta5_run2/` | `older_versions/experiments/sdxl/dps_zeta5_run2/` | 172 MB | Cut |
| `experiments/sdxl/output_circle_200/` | `older_versions/experiments/sdxl/output_circle_200/` | 137 MB | Cut |
| `experiments/sdxl/output_dps_guided_200/` | `older_versions/experiments/sdxl/output_dps_guided_200/` | 123 MB | Cut |
| `experiments/sdxl/results_guidance_sweep/` | `older_versions/experiments/sdxl/results_guidance_sweep/` | 100 MB | Superseded |
| `experiments/sdxl/results_street_nocn/` | `older_versions/experiments/sdxl/results_street_nocn/` | 80 MB | Cut |
| `experiments/sdxl/results_controlnet/` | `older_versions/experiments/sdxl/results_controlnet/` | 61 MB | Cut |
| `experiments/sdxl/results_male_female/` | `older_versions/experiments/sdxl/results_male_female/` | 22 MB | Cut |
| `experiments/sdxl/results_street_nobox/` | `older_versions/experiments/sdxl/results_street_nobox/` | 65 MB | Cut |
| `experiments/sdxl/output_controlnet_*/` (19 dirs) | `older_versions/experiments/sdxl/output_controlnet_*/` | ~1.3 GB | All cut |
| `experiments/sdxl/output_noise_denoise_*/` (4 dirs) | `older_versions/experiments/sdxl/output_noise_denoise_*/` | 49 MB | Cut |
| `experiments/sdxl/debug_decode/` | `older_versions/experiments/sdxl/debug_decode/` | 6.9 MB | One-off |

### Dead SDXL Loose Files

| Old Path | New Path | Description |
|----------|----------|-------------|
| `experiments/sdxl/*.out`, `*.err` | `older_versions/experiments/sdxl/` | Old SLURM logs |
| `experiments/sdxl/*.png` | `older_versions/experiments/sdxl/` | Old diagnostic PNGs |
| `experiments/sdxl/EXPERIMENT_dps_converged.md` | `older_versions/experiments/sdxl/` | DPS notes |
| `experiments/sdxl/PLAN.md` | `older_versions/experiments/sdxl/` | Old plan |
| `experiments/sdxl/README.md` | `older_versions/experiments/sdxl/` | Old README |
| `experiments/sdxl/ssa_entropy_table*.md` | `older_versions/experiments/sdxl/` | Old table |
| `experiments/sdxl/paper_rho_curves.csv` | `older_versions/experiments/sdxl/` | Old CSV |

### Dead SDXL Scripts (30 Python files → older_versions/experiments/sdxl/scripts/)

compare_circle_vs_face.py, compare_clip_latentclip_pca.py, compute_entropy_comparison.py, compute_square_direction.py, debug_gender_ranking.py, debug_latentclip_validation.py, fig_elderly_da_vs_pca.py, full_correlation_table.py, generate_controlnet_samples.py, interpret_da_clip.py, interpret_latentclip_pca.py, make_autoresearch_scribbles.py, make_pc_scatter.py, make_pca_grids.py, noise_denoise_turbo.py, plot_clip_gender.py, plot_clip_pca.py, plot_clip_pca_compare.py, plot_correlation_tables.py, plot_da_pca_all_prompts.py, plot_da_vs_pca_grids.py, plot_dps_comparison.py, plot_dps_scatter.py, plot_latent_clip_pca.py, plot_ssa_comparison.py, preprocess_control.py, validate_latentclip_probes.py, visualize_clip_da_sweep.py, visualize_da.py, visualize_results.py

### Dead SLURM Scripts (76 .sh files → older_versions/experiments/sdxl/cluster/)

All SLURM scripts except the 14 paper-critical ones listed in the "Kept" section below.

---

## 2. SDXL Reorganization

### experiments/sdxl/shared/

| Old Path | New Path |
|----------|----------|
| `experiments/sdxl/scripts/sdxl_components.py` | `experiments/sdxl/shared/sdxl_components.py` |
| `experiments/sdxl/scripts/compute_sdxl_ssa_v2.py` | `experiments/sdxl/shared/compute_sdxl_ssa_v2.py` |
| `experiments/sdxl/scripts/decode_all_grids.py` | `experiments/sdxl/shared/decode_all_grids.py` |
| `experiments/sdxl/scripts/extract_clip_projections.py` | `experiments/sdxl/shared/extract_clip_projections.py` |
| `experiments/sdxl/scripts/gen_quick_samples.py` | `experiments/sdxl/shared/gen_quick_samples.py` |
| `experiments/sdxl/data/` | `experiments/sdxl/shared/data/` |

### experiments/sdxl/guidance_sweep/

| Old Path | New Path |
|----------|----------|
| `experiments/sdxl/scripts/run_ssa_da_clip.py` | `experiments/sdxl/guidance_sweep/run_ssa_da_clip.py` |
| `experiments/sdxl/scripts/plot_multiprompt_sweep.py` | `experiments/sdxl/guidance_sweep/plot_multiprompt_sweep.py` |
| `experiments/sdxl/scripts/generate_and_entropy.py` | `experiments/sdxl/guidance_sweep/generate_and_entropy.py` |
| `experiments/sdxl/scripts/recompute_entropy.py` | `experiments/sdxl/guidance_sweep/recompute_entropy.py` |
| `experiments/sdxl/scripts/decode_person_hires.py` | `experiments/sdxl/guidance_sweep/decode_person_hires.py` |
| `experiments/sdxl/results_multiprompt_sweep/` | `experiments/sdxl/guidance_sweep/results_multiprompt_sweep/` |
| `experiments/sdxl/results_traj_averaged/` | `experiments/sdxl/guidance_sweep/results_traj_averaged/` |
| `experiments/sdxl/cluster/run_guidance_sweep_gen.sh` | `experiments/sdxl/guidance_sweep/cluster/run_guidance_sweep_gen.sh` |
| `experiments/sdxl/cluster/run_guidance_sweep_quick.sh` | `experiments/sdxl/guidance_sweep/cluster/run_guidance_sweep_quick.sh` |
| `experiments/sdxl/cluster/run_guidance_sweep_ssa.sh` | `experiments/sdxl/guidance_sweep/cluster/run_guidance_sweep_ssa.sh` |
| `experiments/sdxl/cluster/run_multiprompt_sweep_quick.sh` | `experiments/sdxl/guidance_sweep/cluster/run_multiprompt_sweep_quick.sh` |
| `experiments/sdxl/cluster/run_multiprompt_sweep_gen.sh` | `experiments/sdxl/guidance_sweep/cluster/run_multiprompt_sweep_gen.sh` |
| `experiments/sdxl/cluster/run_multiprompt_sweep_ssa.sh` | `experiments/sdxl/guidance_sweep/cluster/run_multiprompt_sweep_ssa.sh` |
| `experiments/sdxl/cluster/run_decode_multiprompt_sweep.sh` | `experiments/sdxl/guidance_sweep/cluster/run_decode_multiprompt_sweep.sh` |
| `experiments/sdxl/cluster/run_recompute_entropy.sh` | `experiments/sdxl/guidance_sweep/cluster/run_recompute_entropy.sh` |
| `experiments/sdxl/cluster/run_decode_person_hires.sh` | `experiments/sdxl/guidance_sweep/cluster/run_decode_person_hires.sh` |

### experiments/sdxl/elderly_da/

| Old Path | New Path |
|----------|----------|
| `experiments/sdxl/scripts/fig_elderly_da_vs_pca_freeconv.py` | `experiments/sdxl/elderly_da/fig_elderly_da_vs_pca_freeconv.py` |
| `experiments/sdxl/scripts/save_c0_eigvals_elderly.py` | `experiments/sdxl/elderly_da/save_c0_eigvals_elderly.py` |
| `experiments/sdxl/results_da_pcavda/` | `experiments/sdxl/elderly_da/results_da_pcavda/` |
| `experiments/elderly_da_bootstrap/` | `experiments/sdxl/elderly_da/bootstrap/` |
| `experiments/sdxl/cluster/run_da_elderly_freeconv.sh` | `experiments/sdxl/elderly_da/cluster/run_da_elderly_freeconv.sh` |
| `experiments/sdxl/cluster/run_da_elderly_freeconv_catfish.sh` | `experiments/sdxl/elderly_da/cluster/run_da_elderly_freeconv_catfish.sh` |
| `experiments/sdxl/cluster/run_fig_elderly_da_pca_freeconv.sh` | `experiments/sdxl/elderly_da/cluster/run_fig_elderly_da_pca_freeconv.sh` |
| `experiments/sdxl/cluster/run_fig_elderly_da_pca_freeconv_taustar.sh` | `experiments/sdxl/elderly_da/cluster/run_fig_elderly_da_pca_freeconv_taustar.sh` |
| `experiments/sdxl/cluster/run_save_c0_eigvals_elderly.sh` | `experiments/sdxl/elderly_da/cluster/run_save_c0_eigvals_elderly.sh` |

### experiments/sdxl/crane_da/

| Old Path | New Path |
|----------|----------|
| `experiments/sdxl/results_traj_averaged/decode_crane_extremes.py` | `experiments/sdxl/crane_da/decode_crane_extremes.py` |

### experiments/sdxl/weight_comparability/

| Old Path | New Path |
|----------|----------|
| `experiments/sdxl_weight_comparability/` | `experiments/sdxl/weight_comparability/` |

### experiments/sdxl/extended_prompts/

| Old Path | New Path |
|----------|----------|
| `experiments/sdxl_extended_prompts/` | `experiments/sdxl/extended_prompts/` |

### experiments/sdxl/3prompt/

| Old Path | New Path |
|----------|----------|
| `experiments/sdxl_3prompt/` | `experiments/sdxl/3prompt/` |

---

## 3. Alanine Consolidation

| Old Path | New Path |
|----------|----------|
| `experiments/da_directions/*` | `experiments/alanine/main/*` |
| `experiments/alanine_baselines_sparse_kmeans_spectralmap/*` | `experiments/alanine/baselines/*` |
| `experiments/alanine_da_bonferroni/*` | `experiments/alanine/bonferroni/*` |
| `experiments/alanine/diffusion_maps/` | `experiments/alanine/baselines/diffusion_maps/` |

---

## 4. Directories Kept Flat (unchanged)

- `experiments/gmm_sweep/` — GMM sweep configs
- `experiments/ssa_truncation_redef/` — Ŝ redefinition + GMM d=10 sweeps
- `experiments/noise_floor/` — Free-convolution utilities
- `experiments/mp_generalization/` — MP density validation
- `experiments/robustness/` — Cross-cutting: SDXL bootstrap, alanine lag, GMM repeat
- `experiments/save_autocovariance/` — Cross-cutting: alanine + SDXL C(τ) matrices
- `experiments/mc_dip_validation/` — MC null + permutation + dip tests
- `experiments/dt_robustness/` — Δt sensitivity
- `experiments/synthetic_spikes/` — Planted spike validation

---

## 5. Path Fixes Applied

All hardcoded paths in scripts have been updated. No LaTeX changes needed (paper doesn't hardcode experiment paths).

### Python scripts updated:
| File | Change |
|------|--------|
| `figures/regenerate_all.py` | `da_directions` → `alanine/main` (2 occurrences) |
| `experiments/save_autocovariance/save_alanine_C_tau.py` | `da_directions` → `alanine/main` |
| `experiments/save_autocovariance/save_alanine_C_tau_h_half.py` | same |
| `experiments/save_autocovariance/save_alanine_C_tau_h_half_10k.py` | same |
| `experiments/save_autocovariance/save_alanine_C_tau_h_half_50k.py` | same |
| `experiments/save_autocovariance/diagnostic_taustar_negspikes.py` | same |
| `experiments/save_autocovariance/diagnostic_dt_bandwidth.py` | same |
| `experiments/save_autocovariance/n_scaling_sweep.py` | same |
| `experiments/robustness/task2_alanine_lag/alanine_lag_sensitivity.py` | same |
| `experiments/noise_floor/alanine/run_alanine_noise_floor.py` | same |
| `experiments/alanine/baselines/diffusion_maps/run_diffusion_maps.py` | `../../da_directions` → `../../main` (two levels up after move into baselines/) |
| `experiments/robustness/task1_sdxl_bootstrap/bootstrap_shat_all14.py` | `sdxl_extended_prompts` → `sdxl/extended_prompts`, `results_guidance_sweep` → `guidance_sweep/results_multiprompt_sweep` |
| `experiments/sdxl/elderly_da/bootstrap/diag.py` | 3 absolute paths updated |
| `experiments/sdxl/elderly_da/bootstrap/run_bootstrap.py` | Fixed REPO path derivation |
| `experiments/sdxl/3prompt/run_stats.py` | `sdxl_3prompt` → `sdxl/3prompt`, results paths updated |
| `experiments/sdxl/3prompt/build_elderly_compact.py` | `results_da_pcavda` → `elderly_da/results_da_pcavda` |
| `experiments/sdxl/3prompt/build_elderly_compact_freeconv.py` | same |
| `experiments/sdxl/weight_comparability/compute_weights.py` | `sdxl/results_traj_averaged` → `guidance_sweep/results_traj_averaged` |
| `experiments/sdxl/extended_prompts/decode_sample_grid.py` | `../sdxl/scripts` → `../shared` |
| `experiments/sdxl/guidance_sweep/results_traj_averaged/compare_estimators.py` | `sdxl_extended_prompts` → `extended_prompts` |

### sys.path additions (for `from sdxl_components import`):
| File | Added |
|------|-------|
| `experiments/sdxl/guidance_sweep/run_ssa_da_clip.py` | `sys.path.insert(0, os.path.join(EXP_DIR, "shared"))` |
| `experiments/sdxl/guidance_sweep/generate_and_entropy.py` | same |
| `experiments/sdxl/guidance_sweep/recompute_entropy.py` | `sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "shared"))` |
| `experiments/sdxl/elderly_da/fig_elderly_da_vs_pca_freeconv.py` | same |
| `experiments/sdxl/elderly_da/save_c0_eigvals_elderly.py` | same |

### SLURM scripts updated (cd + python paths):
All 14 critical SLURM scripts updated: `cd` targets changed from `experiments/sdxl` to `experiments/sdxl/{guidance_sweep,elderly_da}`, `python scripts/...` → `python ...` or `python ../guidance_sweep/...`

---

## 6. Post-Restructure Directory Tree

```
experiments/
├── alanine/
│   ├── main/              # DA vs TICA (was da_directions/)
│   ├── baselines/         # Sparse K-means, Spectral Map, Diffusion Maps
│   │   └── diffusion_maps/  # DM baseline (Table 2 columns)
│   └── bonferroni/        # DA2 significance audit
├── sdxl/
│   ├── shared/            # sdxl_components.py, utilities, data/
│   ├── guidance_sweep/    # SSA pipeline, 8-prompt sweep, traj-averaged
│   ├── elderly_da/        # DA1=gender, bootstrap, figure generation
│   ├── crane_da/          # Crane PC1/DA1 extremes
│   ├── extended_prompts/  # 6 polysemous nouns
│   ├── weight_comparability/ # w₂ weight ratios
│   └── 3prompt/           # Original 3-prompt revision
├── gmm_sweep/             # GMM configs
├── ssa_truncation_redef/  # Ŝ redefinition, d=10 sweeps
├── noise_floor/           # Free-conv floor
├── mp_generalization/     # MP density validation
├── save_autocovariance/   # C(τ) matrices (alanine + SDXL)
├── mc_dip_validation/     # MC null + dip tests
├── robustness/            # Bootstrap, lag sensitivity, repeatability
├── dt_robustness/         # Δt robustness
└── synthetic_spikes/      # Planted spike validation

older_versions/            # 18 GB dead weight (gitignored)
├── experiments_whitening/
├── covf_replication/
├── results/
├── results_multiprompt_sweep/
└── experiments/
    ├── mnist/
    ├── beans/
    ├── california_housing/
    ├── wine_quality/
    ├── integrated_da/
    ├── marchenko_pastur/
    ├── peeled_rho/
    ├── alanine_pca_panel/
    ├── autoresearch/
    └── sdxl/              # 12 GB dead SDXL outputs + scripts
```
