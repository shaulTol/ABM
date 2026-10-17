# P10: score-error sensitivity of the null spectrum

**Scope.** Theorem 2 is exact only for its isotropic-Gaussian, independent-pair asymptotic model. Everything below is an EMPIRICAL robustness probe on a finite grid of *injected* score errors. It is not an extension of the theorem and not a guarantee for arbitrary unknown score error. Matched Monte Carlo does **not** automatically absorb real score error: it only absorbs error mechanisms that are explicitly simulated, and the mechanisms here are two synthetic random-Fourier-feature families, not a learned score model.

Target N(0, I_d), exact score s(x) = -x. Perturbed score `s_alpha(x) = -x + alpha * E(x)` with E calibrated so its per-coordinate RMS equals the baseline per-coordinate score RMS s0 = 1 (same relative strengths as P1). CONSERVATIVE: E = grad psi, so the perturbed stationary density `p_alpha ~ exp(-|x|^2/2 + alpha psi)` is KNOWN (validated against MALA). NONCONSERVATIVE: the P1 frozen RFF vector field, which is not a gradient and has NO known stationary density; reported separately, after burn-in, with an empirical stationarity check. Spectra are compared after whitening by the EMPIRICAL C(0).


## Regime `alanine-like`: d=40, N_traj=2000, P=2, L=40, dt=0.05, tau=2.0, burn-in=3000 steps, 100 replicates, gamma_traj=0.0200


### Estimator `simple_Ntraj` (M=2000, gamma_pair=0.02000, exact-score analytic edge lambda_+ = 0.5916)

| family | alpha | lam_max median | q95 | q99 | lam_max/lam_+ | FPR vs analytic edge [95% CI] | FPR vs exact-score MC q95 [95% CI] | matched q95 shift | rho_hat (exact-score e^{-tau/2}) | C0 trace/d | M_eff/M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| conservative | 0.0 | 0.5664 | 0.5877 | 0.5963 | 0.957 | 0.030 [0.010, 0.085] | 0.050 [0.022, 0.112] | 1.000 | 0.3633 (0.3679) | 1.012 | 1.256 |
| conservative | 0.1 | 0.5746 | 0.5981 | 0.6060 | 0.971 | 0.100 [0.055, 0.174] | 0.180 [0.117, 0.267] | 1.018 | 0.3650 (0.3679) | 1.014 | 1.163 |
| conservative | 0.25 | 0.5989 | 0.6455 | 0.6678 | 1.012 | 0.590 [0.492, 0.681] | 0.670 [0.573, 0.754] | 1.098 | 0.3708 (0.3679) | 1.017 | 0.941 |
| conservative | 0.5 | 0.7018 | 0.7840 | 0.8566 | 1.186 | 0.960 [0.902, 0.984] | 0.970 [0.915, 0.990] | 1.334 | 0.3866 (0.3679) | 1.022 | 0.470 |
| conservative | 1.0 | 0.8523 | 0.9615 | 0.9735 | 1.441 | 1.000 [0.963, 1.000] | 1.000 [0.963, 1.000] | 1.636 | 0.4351 (0.3679) | 1.037 | 0.237 |
| conservative | 2.0 | 1.0031 | 1.0518 | 1.0628 | 1.695 | 1.000 [0.963, 1.000] | 1.000 [0.963, 1.000] | 1.790 | 0.5794 (0.3679) | 1.280 | 0.146 |
| nonconservative | 0.1 | 0.5717 | 0.5914 | 0.6030 | 0.966 | 0.050 [0.022, 0.112] | 0.080 [0.041, 0.150] | 1.006 | 0.3644 (0.3679) | 1.016 | 1.195 |
| nonconservative | 0.25 | 0.5772 | 0.6046 | 0.6160 | 0.976 | 0.130 [0.078, 0.210] | 0.200 [0.133, 0.289] | 1.029 | 0.3681 (0.3679) | 1.030 | 1.136 |
| nonconservative | 0.5 | 0.6197 | 0.6530 | 0.6557 | 1.047 | 0.980 [0.930, 0.994] | 0.990 [0.946, 0.998] | 1.111 | 0.3830 (0.3679) | 1.081 | 0.799 |
| nonconservative | 1.0 | 0.7454 | 0.7982 | 0.8131 | 1.260 | 1.000 [0.963, 1.000] | 1.000 [0.963, 1.000] | 1.358 | 0.4395 (0.3679) | 1.323 | 0.374 |
| nonconservative | 2.0 | 0.8376 | 0.8741 | 0.8849 | 1.416 | 1.000 [0.963, 1.000] | 1.000 [0.963, 1.000] | 1.487 | 0.5504 (0.3679) | 2.534 | 0.251 |

### Estimator `simple_M` (M=4000, gamma_pair=0.01000, exact-score analytic edge lambda_+ = 0.5238)

| family | alpha | lam_max median | q95 | q99 | lam_max/lam_+ | FPR vs analytic edge [95% CI] | FPR vs exact-score MC q95 [95% CI] | matched q95 shift | rho_hat (exact-score e^{-tau/2}) | C0 trace/d | M_eff/M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| conservative | 0.0 | 0.5020 | 0.5158 | 0.5184 | 0.958 | 0.010 [0.002, 0.054] | 0.050 [0.022, 0.112] | 1.000 | 0.3633 (0.3679) | 1.012 | 1.339 |
| conservative | 0.1 | 0.5127 | 0.5381 | 0.5533 | 0.979 | 0.190 [0.125, 0.278] | 0.400 [0.309, 0.498] | 1.043 | 0.3651 (0.3679) | 1.014 | 1.153 |
| conservative | 0.25 | 0.5602 | 0.6105 | 0.6265 | 1.069 | 0.890 [0.814, 0.937] | 0.960 [0.902, 0.984] | 1.184 | 0.3702 (0.3679) | 1.017 | 0.668 |
| conservative | 0.5 | 0.6791 | 0.7681 | 0.8175 | 1.296 | 1.000 [0.963, 1.000] | 1.000 [0.963, 1.000] | 1.489 | 0.3868 (0.3679) | 1.022 | 0.268 |
| conservative | 1.0 | 0.8316 | 0.9436 | 0.9565 | 1.587 | 1.000 [0.963, 1.000] | 1.000 [0.963, 1.000] | 1.829 | 0.4351 (0.3679) | 1.037 | 0.128 |
| conservative | 2.0 | 0.9779 | 0.9910 | 0.9923 | 1.867 | 1.000 [0.963, 1.000] | 1.000 [0.963, 1.000] | 1.921 | 0.5792 (0.3679) | 1.280 | 0.078 |
| nonconservative | 0.1 | 0.5051 | 0.5197 | 0.5235 | 0.964 | 0.010 [0.002, 0.054] | 0.170 [0.109, 0.255] | 1.008 | 0.3643 (0.3679) | 1.016 | 1.281 |
| nonconservative | 0.25 | 0.5203 | 0.5450 | 0.5543 | 0.993 | 0.390 [0.300, 0.488] | 0.720 [0.625, 0.799] | 1.057 | 0.3682 (0.3679) | 1.030 | 1.045 |
| nonconservative | 0.5 | 0.5823 | 0.6100 | 0.6167 | 1.112 | 1.000 [0.963, 1.000] | 1.000 [0.963, 1.000] | 1.183 | 0.3832 (0.3679) | 1.081 | 0.542 |
| nonconservative | 1.0 | 0.7165 | 0.7603 | 0.7714 | 1.368 | 1.000 [0.963, 1.000] | 1.000 [0.963, 1.000] | 1.474 | 0.4397 (0.3679) | 1.323 | 0.217 |
| nonconservative | 2.0 | 0.8024 | 0.8441 | 0.8513 | 1.532 | 1.000 [0.963, 1.000] | 1.000 [0.963, 1.000] | 1.636 | 0.5506 (0.3679) | 2.534 | 0.144 |

### Estimator `traj` (M=4000, gamma_pair=0.01000, exact-score analytic edge lambda_+ = 0.5238)

| family | alpha | lam_max median | q95 | q99 | lam_max/lam_+ | FPR vs analytic edge [95% CI] | FPR vs exact-score MC q95 [95% CI] | matched q95 shift | rho_hat (exact-score e^{-tau/2}) | C0 trace/d | M_eff/M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| conservative | 0.0 | 0.5221 | 0.5424 | 0.5474 | 0.997 | 0.410 [0.319, 0.508] | 0.050 [0.022, 0.112] | 1.000 | 0.3629 (0.3679) | 1.012 | 1.022 |
| conservative | 0.1 | 0.5296 | 0.5512 | 0.5617 | 1.011 | 0.720 [0.625, 0.799] | 0.160 [0.101, 0.244] | 1.016 | 0.3645 (0.3679) | 1.014 | 0.932 |
| conservative | 0.25 | 0.5744 | 0.6247 | 0.6579 | 1.096 | 0.950 [0.888, 0.978] | 0.830 [0.745, 0.891] | 1.152 | 0.3704 (0.3679) | 1.017 | 0.583 |
| conservative | 0.5 | 0.6913 | 0.7731 | 0.8532 | 1.320 | 1.000 [0.963, 1.000] | 0.990 [0.946, 0.998] | 1.425 | 0.3864 (0.3679) | 1.022 | 0.249 |
| conservative | 1.0 | 0.8420 | 0.9561 | 0.9700 | 1.607 | 1.000 [0.963, 1.000] | 1.000 [0.963, 1.000] | 1.763 | 0.4351 (0.3679) | 1.037 | 0.123 |
| conservative | 2.0 | 0.9988 | 1.0461 | 1.0558 | 1.907 | 1.000 [0.963, 1.000] | 1.000 [0.963, 1.000] | 1.929 | 0.5789 (0.3679) | 1.280 | 0.074 |
| nonconservative | 0.1 | 0.5264 | 0.5405 | 0.5427 | 1.005 | 0.610 [0.512, 0.700] | 0.020 [0.006, 0.070] | 0.996 | 0.3643 (0.3679) | 1.016 | 0.969 |
| nonconservative | 0.25 | 0.5395 | 0.5661 | 0.5723 | 1.030 | 0.950 [0.888, 0.978] | 0.440 [0.347, 0.538] | 1.044 | 0.3681 (0.3679) | 1.030 | 0.831 |
| nonconservative | 0.5 | 0.5958 | 0.6269 | 0.6473 | 1.137 | 1.000 [0.963, 1.000] | 1.000 [0.963, 1.000] | 1.156 | 0.3832 (0.3679) | 1.081 | 0.483 |
| nonconservative | 1.0 | 0.7247 | 0.7834 | 0.7974 | 1.383 | 1.000 [0.963, 1.000] | 1.000 [0.963, 1.000] | 1.444 | 0.4392 (0.3679) | 1.323 | 0.208 |
| nonconservative | 2.0 | 0.8214 | 0.8577 | 0.8776 | 1.568 | 1.000 [0.963, 1.000] | 1.000 [0.963, 1.000] | 1.581 | 0.5501 (0.3679) | 2.534 | 0.134 |

## Regime `sdxl-like`: d=768, N_traj=200, P=2, L=40, dt=0.05, tau=2.0, burn-in=3000 steps, 100 replicates, gamma_traj=3.8400


### Estimator `simple_Ntraj` (M=200, gamma_pair=3.84000, exact-score analytic edge lambda_+ = 5.7487)

| family | alpha | lam_max median | q95 | q99 | lam_max/lam_+ | FPR vs analytic edge [95% CI] | FPR vs exact-score MC q95 [95% CI] | matched q95 shift | rho_hat (exact-score e^{-tau/2}) | C0 trace/d | M_eff/M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| conservative | 0.0 | 4.4571 | 4.5505 | 4.5749 | 0.775 | 0.000 [0.000, 0.037] | 0.050 [0.022, 0.112] | 1.000 | 0.3627 (0.3679) | 1.011 | 1.459 |
| conservative | 0.1 | 4.4326 | 4.5576 | 4.5788 | 0.771 | 0.000 [0.000, 0.037] | 0.070 [0.034, 0.137] | 1.002 | 0.3629 (0.3679) | 1.011 | 1.471 |
| conservative | 0.25 | 4.4204 | 4.5561 | 4.5928 | 0.769 | 0.000 [0.000, 0.037] | 0.060 [0.028, 0.125] | 1.001 | 0.3672 (0.3679) | 1.012 | 1.478 |
| conservative | 0.5 | 4.3273 | 4.5274 | 4.6784 | 0.753 | 0.000 [0.000, 0.037] | 0.040 [0.016, 0.098] | 0.995 | 0.3784 (0.3679) | 1.016 | 1.527 |
| conservative | 1.0 | 3.6982 | 4.0487 | 4.2034 | 0.643 | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.037] | 0.890 | 0.4092 (0.3679) | 1.052 | 1.953 |
| conservative | 2.0 | 3.3653 | 3.5860 | 3.7070 | 0.585 | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.037] | 0.788 | 0.4457 (0.3679) | 1.106 | 2.274 |
| nonconservative | 0.1 | 4.4597 | 4.5398 | 4.5758 | 0.776 | 0.000 [0.000, 0.037] | 0.040 [0.016, 0.098] | 0.998 | 0.3633 (0.3679) | 1.014 | 1.458 |
| nonconservative | 0.25 | 4.3843 | 4.4688 | 4.4811 | 0.763 | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.037] | 0.982 | 0.3676 (0.3679) | 1.028 | 1.496 |
| nonconservative | 0.5 | 4.0790 | 4.1682 | 4.2393 | 0.710 | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.037] | 0.916 | 0.3839 (0.3679) | 1.081 | 1.673 |
| nonconservative | 1.0 | 2.9394 | 3.0287 | 3.0643 | 0.511 | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.037] | 0.666 | 0.4442 (0.3679) | 1.337 | 2.851 |
| nonconservative | 2.0 | 2.6107 | 2.6394 | 2.6475 | 0.454 | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.037] | 0.580 | 0.5524 (0.3679) | 2.607 | 3.501 |

### Estimator `simple_M` (M=400, gamma_pair=1.92000, exact-score analytic edge lambda_+ = 3.6432)

| family | alpha | lam_max median | q95 | q99 | lam_max/lam_+ | FPR vs analytic edge [95% CI] | FPR vs exact-score MC q95 [95% CI] | matched q95 shift | rho_hat (exact-score e^{-tau/2}) | C0 trace/d | M_eff/M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| conservative | 0.0 | 2.7165 | 2.7508 | 2.7722 | 0.746 | 0.000 [0.000, 0.037] | 0.050 [0.022, 0.112] | 1.000 | 0.3629 (0.3679) | 1.011 | 1.633 |
| conservative | 0.1 | 2.7045 | 2.7459 | 2.7533 | 0.742 | 0.000 [0.000, 0.037] | 0.040 [0.016, 0.098] | 0.998 | 0.3637 (0.3679) | 1.011 | 1.645 |
| conservative | 0.25 | 2.7877 | 3.0941 | 3.1515 | 0.765 | 0.000 [0.000, 0.037] | 0.630 [0.532, 0.718] | 1.125 | 0.3679 (0.3679) | 1.012 | 1.561 |
| conservative | 0.5 | 2.9223 | 3.1160 | 3.1616 | 0.802 | 0.000 [0.000, 0.037] | 0.930 [0.863, 0.966] | 1.133 | 0.3786 (0.3679) | 1.016 | 1.440 |
| conservative | 1.0 | 2.3324 | 2.7326 | 2.8224 | 0.640 | 0.000 [0.000, 0.037] | 0.020 [0.006, 0.070] | 0.993 | 0.4095 (0.3679) | 1.052 | 2.144 |
| conservative | 2.0 | 2.0107 | 2.2276 | 2.2952 | 0.552 | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.037] | 0.810 | 0.4462 (0.3679) | 1.106 | 2.836 |
| nonconservative | 0.1 | 2.7037 | 2.7450 | 2.7649 | 0.742 | 0.000 [0.000, 0.037] | 0.040 [0.016, 0.098] | 0.998 | 0.3638 (0.3679) | 1.014 | 1.646 |
| nonconservative | 0.25 | 2.6783 | 2.7165 | 2.7429 | 0.735 | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.037] | 0.988 | 0.3680 (0.3679) | 1.028 | 1.674 |
| nonconservative | 0.5 | 2.5087 | 2.5798 | 2.6069 | 0.689 | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.037] | 0.938 | 0.3841 (0.3679) | 1.081 | 1.879 |
| nonconservative | 1.0 | 1.6883 | 1.7511 | 1.7788 | 0.463 | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.037] | 0.637 | 0.4449 (0.3679) | 1.337 | 4.027 |
| nonconservative | 2.0 | 1.4362 | 1.4451 | 1.4513 | 0.394 | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.037] | 0.525 | 0.5529 (0.3679) | 2.607 | 5.713 |

### Estimator `traj` (M=400, gamma_pair=1.92000, exact-score analytic edge lambda_+ = 3.6432)

| family | alpha | lam_max median | q95 | q99 | lam_max/lam_+ | FPR vs analytic edge [95% CI] | FPR vs exact-score MC q95 [95% CI] | matched q95 shift | rho_hat (exact-score e^{-tau/2}) | C0 trace/d | M_eff/M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| conservative | 0.0 | 3.5772 | 3.6623 | 3.6937 | 0.982 | 0.090 [0.048, 0.162] | 0.050 [0.022, 0.112] | 1.000 | 0.3613 (0.3679) | 1.011 | 1.030 |
| conservative | 0.1 | 3.5748 | 3.6456 | 3.6871 | 0.981 | 0.060 [0.028, 0.125] | 0.030 [0.010, 0.085] | 0.995 | 0.3620 (0.3679) | 1.011 | 1.031 |
| conservative | 0.25 | 3.6181 | 3.8449 | 4.0096 | 0.993 | 0.440 [0.347, 0.538] | 0.390 [0.300, 0.488] | 1.050 | 0.3659 (0.3679) | 1.012 | 1.011 |
| conservative | 0.5 | 3.7373 | 3.9322 | 3.9807 | 1.026 | 0.750 [0.657, 0.825] | 0.720 [0.625, 0.799] | 1.074 | 0.3770 (0.3679) | 1.016 | 0.960 |
| conservative | 1.0 | 3.1604 | 3.5263 | 3.6871 | 0.867 | 0.030 [0.010, 0.085] | 0.030 [0.010, 0.085] | 0.963 | 0.4080 (0.3679) | 1.052 | 1.262 |
| conservative | 2.0 | 2.8328 | 3.0571 | 3.1596 | 0.778 | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.037] | 0.835 | 0.4443 (0.3679) | 1.106 | 1.518 |
| nonconservative | 0.1 | 3.5832 | 3.6362 | 3.6724 | 0.984 | 0.040 [0.016, 0.098] | 0.030 [0.010, 0.085] | 0.993 | 0.3619 (0.3679) | 1.014 | 1.027 |
| nonconservative | 0.25 | 3.5477 | 3.6054 | 3.6253 | 0.974 | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.037] | 0.984 | 0.3663 (0.3679) | 1.028 | 1.044 |
| nonconservative | 0.5 | 3.3108 | 3.4083 | 3.4228 | 0.909 | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.037] | 0.931 | 0.3824 (0.3679) | 1.081 | 1.168 |
| nonconservative | 1.0 | 2.2858 | 2.3693 | 2.4150 | 0.627 | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.037] | 0.647 | 0.4430 (0.3679) | 1.337 | 2.225 |
| nonconservative | 2.0 | 1.9572 | 1.9746 | 1.9808 | 0.537 | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.037] | 0.539 | 0.5513 (0.3679) | 2.607 | 2.989 |

Column notes: `lam_max` is the top eigenvalue of the WHITENED symmetrized lagged autocovariance. `FPR vs analytic edge` counts replicates exceeding the EXACT-SCORE analytic edge lambda_+(tau; gamma_pair); `FPR vs exact-score MC q95` uses the alpha=0 Monte-Carlo 95th percentile (so it is 0.05 by construction at alpha=0). `matched q95 shift` is that alpha's own MC q95 divided by the alpha=0 MC q95: it says how far a practitioner's matched threshold would move if the error mechanism were simulated. `rho_hat` is the empirical whitened lag-tau correlation; it rising above e^{-tau/2} means the perturbation SLOWED the dynamics, which is an error-induced false SIGNAL rather than null noise. `M_eff` inverts lambda_+(tau; d/M_eff) = lam_max, as in P7.


---

# Joint interpretation of P7 and P10 (null-spectrum concern)

Theorem 2 describes one idealization: an isotropic Gaussian, an exact score, and independent pairs. The applied estimator departs from it in two ways at once, and P7 and P10 measure them one at a time, each with everything else held fixed.

**P7 (dependence).** With the pair count M held EXACTLY equal, overlapping trajectory pairs never lowered the null maximum and raised it by up to 4.18x (36 configurations; the no-overlap control returns 0.991-1.002, i.e. 1 within Monte-Carlo error, and is bitwise identical under matched seeds). The effective number of independent pair products falls to as little as 5.5% of M. So the naive edge at gamma_pair = d/M is ANTI-conservative for overlapping data, and the inflation grows with the within-pair correlation and with how finely the trajectory is sampled.

**P10 (score error).** See the tables above for the measured effect of injected score error at the P1 strengths.

**Together.** The two departures are qualitatively different. Dependence is a variance-of-the-estimator effect: it changes how much the null top eigenvalue fluctuates for a given nominal M, and it is fully reproducible in a matched Monte-Carlo null because the trajectory geometry is known and can be simulated exactly. Score error is a model effect: at the strengths tested it acts mostly through the sampled density itself, and a matched Monte-Carlo null only absorbs it if the error mechanism is written into the simulation - which for a learned score model it generally is not. The practical consequence for the paper's pipeline is that the matched Monte-Carlo / permutation null already used for the reported DA claims handles the P7 departure by construction, whereas the P10 departure has to be argued empirically, which is what these grids do. Neither result licenses a universal one-sided bound: both are finite grids, and the analytic edge remains a theorem only under its own assumptions.
