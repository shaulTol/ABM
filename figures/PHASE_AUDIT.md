# Phase Audit: Unified Spike-Distribution Figures

Generated: 2026-05-04

## Shared Style Module

**File**: `figures/_spike_fig_style.py`

Colors: empirical `#AED6F1`, MC null `#F5B7B1`, free-conv `#C0392B`, lambda_+(tau) `#1E8449`, lambda_+(inf) `#E67E22`, MC 95th pctl `#8E44AD`.

---

## Figure 1: `fig_free_conv_validation.png`

**Script**: `figures/make_fig_free_conv_validation_unified.py`

**Layout**: 6 rows x 4 cols. Rows: gamma in {0.25, 0.5, 0.75, 1, 2, 5}. Cols: tau in {0, 0.5, 2, 10}.

**Parameters**: N=500 fixed, d=gamma*N, sigma=1.0, dt=0.01, 15 seeds pooled per cell.

**Data**: Simulated on-the-fly (OU Langevin). No external data files.

**Sanity checks**:
- gamma=0.25 (d=125): No atom. Eigenvalue range positive at tau=0 (MP regime). Density matches histogram.
- gamma=1 (d=500): Sub-critical boundary. Eigenvalue range includes 0 at tau=0 (rank=N=d).
- gamma=2 (d=1000): At tau=0, d-N=500 zero eigenvalues (standard Wishart). At tau>0, rank up to 2N=d=1000, so no zeros. Histogram confirms.
- gamma=5 (d=2500): 37500 eigenvalues pooled (15 seeds x 2500). ~60% mass at origin (predicted: 1-2/gamma=0.6). Zero eigenvalues appear naturally in histogram bins near 0.

**Draft caption**:
> Eigenvalue histogram of C_sym(tau) at fixed N=500 with d=gamma*N (light blue) vs. the analytic free-convolution density (red) and support edge lambda_+(tau) (green dotted). Rows: gamma in {0.25, 0.5, 0.75, 1, 2, 5}. Columns: tau in {0, 0.5, 2, 10}. Each histogram pools eigenvalues from 15 independent simulations. At gamma=5 (d=2500, N=500), d-2N=1500 eigenvalues are exactly zero per replicate, producing the visible atom at the origin with empirical mass 1-2/gamma=0.6.

---

## Figure 3: `fig_alanine_bulk_validation.png`

**Script**: `figures/make_fig_alanine_bulk_validation_unified.py`

**Layout**: 3 rows x 2 cols = 6 panels.

**Data**: `experiments/save_autocovariance/results/alanine_C_tau_full_h_half_10k.npz`

**Parameters**: d=45, N=2000, gamma=0.0225, sigma_f_sq=9.214e-4, h=h_Scott/2=0.0091.

**Lags** (indices [28, 29, 32, 34, 36, 37]):

| Index | tau (ps) | lambda_1 | lambda_2 | lambda_+(tau) | lambda_+(inf) |
|-------|----------|----------|----------|---------------|---------------|
| 28 | 701.7 | 1.94e-2 | 1.25e-3 | 1.97e-4 | 1.97e-4 |
| 29 | 942.7 | 1.91e-2 | 6.55e-4 | 1.97e-4 | 1.97e-4 |
| 32 | 2285.5 | 1.78e-2 | 6.74e-4 | 1.97e-4 | 1.97e-4 |
| 34 | 4124.6 | 1.62e-2 | 5.64e-4 | 1.97e-4 | 1.97e-4 |
| 36 | 7443.8 | 1.42e-2 | 8.85e-4 | 1.97e-4 | 1.97e-4 |
| 37 | 10000.0 | 1.21e-2 | 6.10e-4 | 1.97e-4 | 1.97e-4 |

**Tau convention**: lambda_+(tau) and lambda_+(inf) are both 1.9655e-4 at this gamma (saturated by tau~10ps). The dark green and orange dotted lines overlap in all 6 panels.

**Deviations from plan**: None. No peeling, no bulk refit. Uses naive sigma_f_sq directly.

**Draft caption**:
> Eigenvalue histogram of C_sym(tau) for alanine dipeptide (d=45, N=2000, gamma=0.0225) at six lags (light blue), with the analytic free-convolution density (red) and floor lambda_+(tau) (green dotted) and lambda_+(inf) (orange dotted). DA1 and DA2 spikes are labeled above the bulk. Symlog x-axis.

---

## Bonferroni Table: `table_alanine_bonferroni.{md,csv}`

**Script**: `figures/make_table_alanine_bonferroni.py`

**Data**: Same as Figure 3, index 37 (tau*=10000ps).

**Results**:

| i | lambda_i^obs | lambda_+(tau*) | Ratio | Pass |
|---|-------------|----------------|-------|------|
| 1 | 1.2099e-2 | 1.9655e-4 | 61.6 | yes |
| 2 | 6.0956e-4 | 1.9655e-4 | 3.1 | yes |
| 3 | 6.0241e-5 | 1.9655e-4 | 0.3 | no |
| 4 | 4.8199e-5 | 1.9655e-4 | 0.2 | no |
| 5 | 2.1401e-5 | 1.9655e-4 | 0.1 | no |

**hat_m = 2**. Both DA1 (psi) and DA2 (phi) exceed the free-conv floor. DA3 onward are in the bulk.

---

## Figure 6: `fig_elderly_spectrum.png`

**Script**: `figures/make_fig_elderly_spectrum_unified.py`

**Layout**: Single panel.

**Data**:
- Real eigenvalues: `experiments/sdxl/results_da_pcavda/clip_da_elderly_gs7p5_fc_dt0p0025.npz` -> `eigenvalues_full_at_tau_star` (768 values at tau*=5.0s)
- C(0): `experiments/save_autocovariance/results/elderly_gs7p5_n200_C_tau_full.npz` -> `C0` (768x768)

**Parameters**: d=768, N=200, gamma=3.84, tau*=5.0, sigma_f_sq=1.122e-4.

**Results**:
- DA1 = 9.962e-3
- lambda_+(tau*=5.0) = 4.989e-4
- lambda_+(inf) = 4.577e-4
- MC null 95th pctl = 4.109e-3 (B=200, seed=42)
- DA1 / MC_95 = 2.4x
- Near-zero eigenvalues: 518/768 (predicted atom mass 1-2/gamma = 0.479)

**Deviations from plan**: lambda_+(tau*) is 4.99e-4 (not 1.51e-3 as estimated in plan). The plan estimates used a different sigma value. The actual sigma_f_sq from the npz data (1.122e-4) is authoritative. The MC null 95th percentile (4.11e-3) matches the plan exactly.

**Draft caption**:
> Eigenvalue spectrum of C_sym(tau*) for SDXL elderly (w=7.5, d=768, N=200, gamma=3.84, tau*=5.0s). Light blue: empirical eigenvalues. Pink: MC null eigenvalues (200 reps of iid OU draws from C(0)). Red curve: analytic free-convolution density. Green dotted: lambda_+(tau*). Orange dotted: lambda_+(inf). Purple dotted: MC null 95th percentile. DA1 (gender direction) is 2.4x above the MC null threshold.

---

## Verification Checklist

- [x] Tau convention: edges(10000.0, sqrt(9.214e-4), 0.0225) = 1.9655e-4
- [x] Fig 1: gamma=5 histogram shows ~60% mass at zero naturally
- [x] Fig 1: gamma=2 has no zero eigenvalues at tau>0 (sub-critical boundary)
- [x] Fig 1: gamma=0.25 no atom, density matches histogram
- [x] Fig 3: DA1 at tau=10000ps = 1.21e-2, DA2 = 6.10e-4
- [x] Fig 3: lambda_+ = 1.97e-4 at all 6 lags (saturated)
- [x] Fig 6: DA1 = 9.96e-3, MC null 95th pctl = 4.11e-3
- [x] Fig 6: lambda_+(5.0) = 4.99e-4, lambda_+(inf) = 4.58e-4 (distinct)
- [x] Table: all 5 eigenvalues computed from data, hat_m=2 determined empirically
