# SDXL Elderly DA1 — iso-Gaussian MC null via late-t SDXL score

Iso-Gaussian MC null for SDXL elderly DA1, implemented by running the canonical
diffusion using the SDXL score evaluated at a large noise level `t_null` where
`f_{t_null}` is approximately iso-Gaussian. This empirically estimates the
iso-Gaussian Theorem 3 floor while folding in score-estimation noise.

Initial conditions sampled from the iso-Gaussian `f_{t_null}` marginal; the
score is evaluated at the same `t_null` throughout the trajectory. The
real-data run uses `t=1` (small-noise, recovers the data score); this null run
uses `t_null` (large-noise, recovers the iso-Gaussian score).

## Why this is a valid null

At large t in the SDXL forward process, the marginal `f_t(x)` approaches
`N(0, sigma_t^2 I)` by construction of the noise schedule. The denoiser
`eps_theta(x, t_large)` was trained to predict the noise added to data points
at that level — i.e., for x sampled from `f_t ≈ N(0, sigma_t^2 I)`,
`eps_theta(x, t) ≈ x / sigma_t`, giving score
`s(x) = -eps_theta(x, t)/sigma_t ≈ -x / sigma_t^2`. That's the analytical
score of `N(0, sigma_t^2 I)`. So running the canonical Langevin
`dX = (var_f/2) s(X) dt + sqrt(var_f) dW` with `var_f = sigma_t^2` and `s`
above has stationary distribution `N(0, sigma_t^2 I)` — an iso-Gaussian.

Compared to a hand-coded analytical OU iso-Gaussian, this version folds in
the actual score-estimation noise that the SDXL UNet contributes, plus the
VAE-decode + CLIP-embed pushforward. So the resulting null distribution of
`lambda_1(C_hat_sym(tau*))` reflects all the real pipeline's noise
sources except the data's lag-tau* coupling.

## What this null is NOT

It is **not** the anisotropic null Or recommended (`Sigma_hat_latent`-matched
samples). The SDXL model has no representation of `N(0, Sigma_hat_latent)`
at any t — only of the iso-Gaussians along its training schedule.

If we wanted a strict anisotropic null, we'd have to abandon SDXL and run a
hand-coded OU on `Sigma_hat_latent` (point (c) below).

## Critical caveat — must verify before launching the array

This null is valid **iff** `f_{t_null}` is sufficiently close to iso-Gaussian
under SDXL. If at `t_null=999` the SDXL model still has nontrivial
conditional structure (e.g., prompt conditioning persisting at large t), the
null is contaminated. We run a **single-trajectory sanity check** first
(see `cluster/run_sanity_check.sh`) and verify:

1. Initial latent norm matches `N(0, sigma_{t_null}^2 I)` expected scale.
2. Trajectory does NOT collapse to the elderly-person attractor.
3. Final CLIP embeddings `z_tau` show no obvious clustering.
4. `lambda_1(C_hat_sym(tau*))` on the sanity subset is much smaller than
   `lambda_1^obs = 9.96e-3`. If comparable, the null is broken.

## If the sanity check fails — fallbacks

1. (a) Even larger `t_null` (= 1000 or higher).
2. (b) Drop conditional prompt at large t (use unconditional SDXL: gs=0).
3. (c) Abandon SDXL. Use analytic OU process on iso-Gaussian (no SDXL
       pipeline at all), then push through VAE+CLIP. Cleanest separation of
       "null dynamics" (clean OU) from "pipeline noise" (VAE+CLIP).

## Files

- `run_sdxl_iso_null.py` — main per-task script (one SLURM array task →
  N_per_task=50 trajectories, save z0 and z_tau in CLIP space).
- `combine_null_results.py` — concat per-task .npz files (unchanged from
  pre-rename anisotropic version; null-design-agnostic).
- `compute_bootstrap_pvalue.py` — local; bootstrap subsamples 200 from N_big,
  build C_b_sym = (C_b + C_b^T)/2, top-10 eigenvalues, p-values + Gumbel tail
  extrapolation. Unchanged from anisotropic version.
- `cluster/run_sdxl_iso_null.sh` — SLURM array job script (16 tasks ×
  N_per_task=50 = N_big=800).
- `cluster/run_sanity_check.sh` — single-task sanity job (N=50, full
  T=2000) submitted **before** the array.
- `results/` — per-task outputs (gitignored via parent `mc_dip_validation/`).

## Configuration

| Param | Value | Notes |
|---|---|---|
| `t_null` | 999 | Near upper end of SDXL's 1000-step schedule. |
| `N_per_task` | 50 | One full batch with score_batch_size=16 = ~4 batches/step. |
| `T`, `dt` | 2000, 0.0025 | Same as real elderly free-conv run. |
| `guidance_scale` | 7.5 | Same as real run (default). Sanity-check fallback may set 0. |
| `var_f` | `sigma_{t_null}^2` (≈1) | Matches stationary variance of iso-Gaussian at `t_null`. |
| Tasks | 16 | N_big = 800. |
| GPU | L40S | salmon partition. |

## Paper claim

DA1 exceeds an iso-Gaussian null implemented via SDXL's own large-t score
(p < ...) AND the DA1 projection rejects unimodality by Hartigan's dip
test (p < ...). Together: DA1 captures real lag-tau* structure beyond
iso-Gaussian noise, and the structure is fragmentation rather than
dispersion.
