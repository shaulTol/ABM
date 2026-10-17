# P8: SDXL DA lag-sensitivity (elderly prompt, w = 7.5, dt = 0.0025)

Read-only re-analysis of stored SDXL DA results. No new simulation, no GPU, no new SDXL sampling. The only model loaded is the real CLIP text encoder (`openai/clip-vit-large-patch14`) on CPU, used to rebuild the two fixed text probes of `experiments/sdxl/fig_elderly_da.py`.

- Prompt: `a portrait of an elderly person`, guidance `7.5`, `dt = 0.0025`, `T = 2000` steps, N = 200, d = 768.
- Evaluation set (common to every lag): **centered z0_reembed (re-embedded initial samples)**; `score_L = (z0_reembed - mean) @ v1(L)`.
- Gender axis: `normalize(normalize("a photograph of a man") - normalize("a photograph of a woman"))`; recomputed with real CLIP text encoder (axis not stored in NPZ/JSON; only cosines and correlations were stored).
- Reference direction: `v1(L* = 2000)`, tau* = 5.0 (dimensionless OU time).

## Sanity checks

| # | Check | Result | Value |
|---|-------|--------|-------|
| 1 | angle(v1(2000), v1(L*)) = 0 | PASS | 0.00e+00 deg |
| 2 | common-set gender \|rho\| at L* ~ 0.90 | PASS | 0.9175 |
| 3 | eig(C_tau_star) top vector matches eigvecs_lag2000[:,0] (sign-invariant) | PASS | \|<v,v*>\| = 1.000000 (0.000 deg) |
| 4 | lambda1(2000) ~ 9.96e-3 | PASS | 9.96189e-03 |
| 5 | lag arrays and free-conv profiles compatible | PASS | 111 lags, 111 lambda_plus values, strictly increasing |
| + | (extra) recomputed gender axis reproduces the stored cos(DA1, gender) magnitude | PASS | \|cos\| = 0.5683475 vs stored 0.5683473 |

The extra row confirms the CLIP text axis was rebuilt exactly. Note the pipeline's stored probe is oriented `woman - man` while `fig_elderly_da.py` uses `man - woman`; the two differ by a sign only, which is irrelevant here because all correlations are reported as \|rho\|.

## Main table

| L | tau = L dt | lambda1 | lambda2 | lambda2/lambda1 | angle to v1(L*) [deg] | gender \|rho\| | traj-avg rho^2 | lambda1 > lambda_+ (analytic) | matched MC, Null B (OU surrogate) |
|---:|---:|---:|---:|---:|---:|---:|---:|:--|:--|
| 1 **NOISE-DOMINATED** | 0.003 | 1.6453e-02 | 1.3351e-02 | 0.811 | (71.6) | 0.338 | n/a (not stored) | yes (17x) | no |
| 20 **NOISE-DOMINATED** | 0.050 | 1.4692e-02 | 1.3620e-02 | 0.927 | (60.0) | 0.428 | 0.7769 | yes (15x) | no |
| 60 **NOISE-DOMINATED** | 0.150 | 1.4435e-02 | 1.3054e-02 | 0.904 | (48.4) | 0.635 | 0.6712 | yes (15x) | no |
| 160 **NOISE-DOMINATED** | 0.400 | 1.4031e-02 | 1.1261e-02 | 0.803 | (32.8) | 0.792 | 0.4566 | yes (16x) | no |
| 400 | 1.000 | 1.3374e-02 | 9.9876e-03 | 0.747 | 30.3 | 0.813 | 0.0899 | yes (17x) | yes (1.1x p95) |
| 800 | 2.000 | 1.1719e-02 | 7.4854e-03 | 0.639 | 19.2 | 0.894 | 0.0822 | yes (18x) | yes (1.4x p95) |
| 1400 | 3.500 | 1.1668e-02 | 5.3457e-03 | 0.458 | 11.3 | 0.910 | 0.6095 | yes (21x) | yes (2.1x p95) |
| 2000 | 5.000 | 9.9619e-03 | 5.1234e-03 | 0.514 | 0.0 | 0.917 | 0.4092 | yes (20x) | yes (2.4x p95)<br>**Null A: yes, 8.1x null max, p_emp <= 1e-4** |

Rows marked **NOISE-DOMINATED** fail at least one available noise reference at that lag; their angles are shown in parentheses and are NOT interpreted.

Notes.
- `lambda1`, `lambda2` are the top two eigenvalues of the symmetrised C(tau), sorted by value (not by absolute value).
- Angles are computed after sign alignment against `v1(L*)` (flip if the inner product is negative).
- Trajectory-averaged rho^2 is stored only on the fixed-20 lag sub-grid (L = 20, 40, ..., 2000); it does not exist for L = 1 and is marked `n/a`.
- `lambda_+` is the stored analytic free-convolution edge (`freeconv_lambda_plus_profile`) aligned with `snapshot_lags`. This is an analytic null reference under its theorem's assumptions, not a proven bound for this data.
- The matched Monte Carlo column is **Null B**: an analytic Gaussian OU surrogate with the empirical C(0), 200 replicates, seed 42, using the `fig_elderly_da.py` spectrum-figure recipe. It is NOT the null behind the '8.1x' figure in the project notes (that is **Null A**, the pipeline-matched anisotropic null, available only at L*). See the reconciliation section below. The column reports rejection against the Null B 95th percentile, which is stable in replicate count (4.11e-3 / 4.04e-3 / 4.16e-3 at R = 200 / 1000 / 5000); the Null B *maximum* is not, so margins are quoted against the percentile.

## Summary

Two noise references are available and they disagree, so both are reported. The analytic free-convolution edge places DA1 above the null at every lag; the matched Monte Carlo null (same OU recipe as the paper's spectrum figure) does not, because at short lag C(tau) is still close to C(0) and a pure-OU null with the empirical C(0) already produces a top eigenvalue of comparable size. The conservative reading uses both.

- Lags reported: 8; passing both references: 4; noise-dominated under at least one reference: 4.
- **Conservative (both references), lags [400, 800, 1400, 2000]: maximum meaningful angle to `v1(L*)` = 30.3 deg** (at L = 400); **gender |rho| range 0.813 to 0.917**.
- Analytic edge only (all 8 lags): maximum angle 71.6 deg (at L = 1); gender |rho| range 0.338 to 0.917.
- At L* itself the common evaluation set gives gender |rho| = 0.917, reproducing the reported ~0.90.

**Verdict.** DA1 is not lag-invariant across the whole stored grid: it rotates monotonically as the lag grows and only settles at long lag. Within the lags where DA1 is above both noise references (tau >= 1.0), the direction is stable to within 30 deg of `v1(L*)` and the gender interpretation is stable (0.81-0.92). At the shortest lag (L = 1) DA1 is essentially PC1 (|cos| = 0.91), which is expected because C(tau) -> C(0) as tau -> 0, and it is also the regime the matched MC null cannot separate from noise; the PC1 overlap then collapses and settles near 0.54 at L*, matching the reported |cos(DA1, PC1)| = 0.514. The headline result is therefore a long-lag property, not an artefact of the single choice L* = 2000, but it is also not lag-free: reporting DA1 at a short lag would give a materially different direction and a much weaker gender correlation.

## Reconciliation of the two MC margins at L* (8.1x vs 2.1x)

Both numbers are computed from the SAME observed lambda_1 = 9.9619e-03. The disagreement is entirely in the NULL, and it is caused by the null CONSTRUCTION, not by the number of replicates. Both are internally correct for their own null.

| | Null A (pipeline anisotropic) | Null B (analytic OU surrogate) |
|---|---|---|
| What is simulated | N(mu, Sigma_hat_latent) initial latents pushed through the REAL SDXL Langevin (T=2000, dt=0.0025), VAE-decoded and CLIP-embedded | closed-form OU in CLIP space: x_tau = exp(-tau/2) z0 + sqrt(1-exp(-tau)) zeta, z0, zeta ~ N(0, C_hat(0)); no SDXL, no VAE, no CLIP |
| Replicates | 17 GPU runs pooled to N_big = 850 pairs, then bootstrap B = 10000 draws of n = 200 | 200 independent replicates of n = 200 |
| 'null max' means | max over the 10,000 bootstrap draws of the rank-1 eigenvalue BY MAGNITUDE (per rank, not across ranks) | max over the replicates of lambda_1 |
| null_max_abs | 1.2327e-03 | 4.6495e-03 |
| null_mean_signed | 3.3268e-04 | 2.4640e-03 (matched-procedure variant, see 3) |
| obs / null_max | **8.08x** | 2.14x |

**1. Which null is which.** The '8.1x' in the project notes comes from `experiments/sdxl/results/sdxl_anisotropic_null_pvalues.json` (`pvalues.lambda_1.null_max_abs`), produced by `older_versions/experiments/mc_dip_validation/sdxl_iso_null_via_late_t/compute_bootstrap_pvalue.py` from the pool `sdxl_null_anisotropic_combined.npz`. It is the **anisotropic, pipeline-matched** null, not an iso-Gaussian one: the notes' phrase 'ISO-GAUSSIAN floor' is a mislabel. (A genuinely iso-Gaussian late-t null exists as a separate, differently-constructed experiment in the same folder; it is not the source of 8.1x.) 'null max' is the maximum over the 10,000 bootstrap draws of the rank-1 magnitude, i.e. over draws, per rank.

**2. Both are internally correct.** Re-running `compute_bootstrap_pvalue.py`'s procedure on the stored pool with its own seed (20260502) reproduces null_max_abs = 1.232742e-03 against the stored 1.232742e-03 (exact match), giving 8.08x. Neither number is stale; they answer different questions.

**3. Apples-to-apples.** Holding Null A's bootstrap procedure completely fixed (same N_big = 850, n = 200, B = 10000, seed = 20260502) and swapping ONLY the pool construction to Null B moves the null maximum from 1.2327e-03 to 6.0118e-03, a factor of 4.88, and the margin from 8.08x to 1.66x. Construction alone accounts for the entire discrepancy.

**Why.** Null A's pool has essentially decorrelated by tau* (null_mean_signed = 3.327e-04), so its lambda_1 is pure finite-sample fluctuation. Null B BUILDS IN a lag coupling rho = exp(-tau*/2) = 0.0821 in every direction, including the top direction of C_hat(0). Its systematic term alone is rho * lambda_1(C_hat(0)) = 1.302e-03, already ~1.1x Null A's ENTIRE null maximum before any fluctuation is added. Null B is therefore the far more conservative null: it asks 'is DA1 slower than a uniform-rate OU?', while Null A asks 'is DA1 above pure sampling noise?'.

**4. Replicate count is not the cause, and the direction refutes it.** A running maximum cannot decrease, so more replicates can only raise a null max and lower the margin. Null B, independent replicates, seed 42:

| replicates R | Null B null_max_abs | obs / null_max |
|---:|---:|---:|
| 200 | 4.6495e-03 | 2.14x |
| 1000 | 5.4306e-03 | 1.83x |
| 5000 | 6.1849e-03 | 1.61x |

Monotone non-decreasing in R: True; margin monotone non-increasing: True. Going from 200 to 10,000 replicates would push Null B's margin further BELOW 1.6x, never up to 8.1x. Sample size is refuted as the explanation.

**What to quote.** Against the pipeline-matched anisotropic null (Null A, the paper's null): **8.1x the null maximum** over 10,000 bootstrap draws, p_emp <= 1e-4. Against the conservative uniform-rate OU surrogate (Null B): **2.4x the 95th percentile** at 200 replicates. Always name the null. Null A is available ONLY at L* = 2000, because the stored null pool contains z0 and z_tau at tau* alone; that is why the per-lag column uses Null B.

## Limitations

- The free-convolution stopping rule never actually triggered on this run (`freeconv_first_cross_idx = -1`), so L* = 2000 is the fallback value equal to the full trajectory length T, not a detected crossing. Lag sensitivity therefore cannot be probed beyond tau = 5.0: nothing longer was stored.
- Trajectory-averaged rho^2 is stored only on the fixed-20 sub-grid, so L = 1 has no value. Its stored profile is also strongly non-monotone in L (it dips to ~1e-4 near L = 600 and rises again), so the column is reported as-is and not used to rank lags.
- The analytic free-convolution edge is a null reference under its theorem's assumptions; it is not a proven bound for this data, and here it is roughly an order of magnitude more permissive than the matched Monte Carlo null at the same lag. Do not read the 15x-21x margins as significance.
- The matched Monte Carlo null uses 200 replicates at a single seed (42); its 95th percentile carries Monte Carlo error, and rows near 1.0x (L = 400) should be treated as borderline.
- The two MC margins (8.1x and ~2.1x) are RESOLVED above: different nulls, both internally correct. The residual risk is presentational -- any quoted margin must name its null. Null A exists only at L*, so the per-lag noise marking rests on Null B plus the analytic edge.
- Null A itself rests on only 17 independent GPU replicates (850 trajectories) resampled 10,000 times; the bootstrap draws share those 17 simulations, so its p-value is conservative with respect to between-simulation variance, as its own source file notes. Its 'p_emp <= 1e-4' is a bootstrap resolution limit, not evidence of a 1e-4 tail.
- Single prompt, single guidance scale, single dt, N = 200. No cross-seed replication of the SDXL run itself is available, so the angles carry no sampling uncertainty from the generative side.
- The gender axis had to be recomputed (only cosines/correlations were stored, not the axis). Agreement with the stored cosine to 7 significant figures confirms it is the same axis.

## Supplementary: overlap of v1(L) with PC1 of the same evaluation set

| L | tau | \|cos(v1(L), PC1)\| | angle to v1(L*) [deg] |
|---:|---:|---:|---:|
| 1 | 0.003 | 0.913 | 71.6 |
| 20 | 0.050 | 0.225 | 60.0 |
| 60 | 0.150 | 0.007 | 48.4 |
| 160 | 0.400 | 0.339 | 32.8 |
| 400 | 1.000 | 0.382 | 30.3 |
| 800 | 2.000 | 0.568 | 19.2 |
| 1400 | 3.500 | 0.567 | 11.3 |
| 2000 | 5.000 | 0.542 | 0.0 |

## Supplementary: stationarity sensitivity on z_tau (NOT the headline)

Same directions, but both the projection and the gender score are computed on centered `z_tau` (the samples at the end of the Langevin run) instead of the re-embedded initial samples. Reported only as a sensitivity; the paper's reported correlation is the `z0_reembed` one above.

| L | gender \|rho\| on z_tau |
|---:|---:|
| 1 | 0.498 |
| 20 | 0.479 |
| 60 | 0.618 |
| 160 | 0.757 |
| 400 | 0.775 |
| 800 | 0.894 |
| 1400 | 0.894 |
| 2000 | 0.896 |

