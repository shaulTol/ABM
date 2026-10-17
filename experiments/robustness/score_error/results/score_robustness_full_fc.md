**SSA->MI Spearman rank correlation under score perturbation** (50 random 10-D GMMs, N=200, dt=0.01, T=100.0s finite horizon, 1/N noise floor, 3 seed(s); mean +/- SD across seeds).

| Perturbation | Level | Spearman(SSA, MI) |
|---|---|---|
| _none (exact score)_ | 0 | +0.877 +/- 0.006 |
| Additive score noise | alpha=0.1 x s0 | +0.872 +/- 0.008 |
| Additive score noise | alpha=0.25 x s0 | +0.874 +/- 0.009 |
| Additive score noise | alpha=0.5 x s0 | +0.865 +/- 0.007 |
| Additive score noise | alpha=1.0 x s0 | +0.832 +/- 0.008 |
| Additive score noise | alpha=2.0 x s0 | +0.741 +/- 0.010 |
| Gaussian smoothing | +0.1 x var per comp | +0.876 +/- 0.006 |
| Gaussian smoothing | +0.25 x var per comp | +0.855 +/- 0.013 |
| Gaussian smoothing | +0.5 x var per comp | +0.597 +/- 0.018 |
| Gaussian smoothing | +1.0 x var per comp | +0.008 +/- 0.008 |

**DA1 angular deviation from exact-score baseline** (representative GMMs = top-5 by true MI; fixed baseline lag per GMM; angle uses sign-invariant |cos|).

| GMM (sep) | baseline DA1 vs e1 | additive alpha=1.0 | smoothing +1.0xvar |
|---|---|---|---|
| #15 (sep=6.9) | 0.6 deg | 1.5 deg | 0.9 deg |
| #29 (sep=7.0) | 0.7 deg | 1.5 deg | 1.8 deg |
| #37 (sep=7.6) | 1.0 deg | 2.6 deg | 1.2 deg |
| #45 (sep=7.9) | 0.9 deg | 2.1 deg | 2.1 deg |
| #28 (sep=6.1) | 1.4 deg | 3.5 deg | 3.9 deg |

