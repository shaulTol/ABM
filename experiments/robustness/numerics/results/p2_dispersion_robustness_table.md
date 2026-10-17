### P2 numerical robustness: dispersion vs fragmentation (main-figure pair)

Distributions: isotropic Gaussian N(0,25I) vs equal-weight parallel-ellipses GMM (means (0,±3), component covariance diag(25,1)) — the exact pair from `fig_dispersion.py`. Baseline: dt=0.01, N=2000, physical simulation horizon 50.0, physical lag window 40.0, 5 seeds (mean ± sd).

**Baseline reproduction (first-crossing rule, the paper's rule, as implemented by the figure code):** Gaussian 0.990 (submitted 1.022), ellipses 1.781 (submitted 1.822) — MISMATCH.

Ŝ is a **finite-horizon** quantity throughout; † marks a window-limited entry (stopping lag reached the edge of the lag window), which is a lower bound, not a resolved population value. Two stopping rules are computed from the *same* stored ρ² curve: **first-crossing** = the paper's rule (and the figure code); **largest-lag** = an alternative rule, as a sensitivity check.


**Stopping rule: first-crossing 1/N (paper rule)**

| condition | Ŝ Gaussian | Ŝ ellipses | ratio | ordering preserved | τ* G / E |
|---|---|---|---|---|---|
| dt=0.005 | 0.994±0.006 | 1.791±0.013 | 1.80× | **YES** | 7.7 / 19.9 |
| dt=0.01 | 0.990±0.004 | 1.781±0.015 | 1.80× | **YES** | 7.5 / 19.4 |
| dt=0.02 | 0.985±0.007 | 1.769±0.018 | 1.80× | **YES** | 7.3 / 20.9 |
| N=200 | 0.997±0.020 | 1.853±0.059 | 1.86× | **YES** | 5.4 / 13.8 |
| N=500 | 0.987±0.017 | 1.783±0.034 | 1.81× | **YES** | 6.3 / 15.8 |
| N=1000 | 0.996±0.008 | 1.785±0.022 | 1.79× | **YES** | 6.8 / 18.0 |
| N=2000 | 0.990±0.004 | 1.781±0.015 | 1.80× | **YES** | 7.5 / 19.4 |
| lag window=10 | 0.990±0.004 | 1.754±0.011† | 1.77× | **YES** | 7.5 / 10.0 |
| lag window=20 | 0.990±0.004 | 1.781±0.014† | 1.80× | **YES** | 7.5 / 19.2 |
| lag window=30 | 0.990±0.004 | 1.781±0.015 | 1.80× | **YES** | 7.5 / 19.4 |
| lag window=40 | 0.990±0.004 | 1.781±0.015 | 1.80× | **YES** | 7.5 / 19.4 |

**Stopping rule: largest-qualifying-lag 1/N (sensitivity)**

| condition | Ŝ Gaussian | Ŝ ellipses | ratio | ordering preserved | τ* G / E |
|---|---|---|---|---|---|
| dt=0.005 | 0.994±0.006 | 1.791±0.014 | 1.80× | **YES** | 7.7 / 21.5 |
| dt=0.01 | 0.990±0.004 | 1.782±0.017 | 1.80× | **YES** | 7.5 / 22.6 |
| dt=0.02 | 0.985±0.007 | 1.769±0.018 | 1.80× | **YES** | 7.3 / 20.9 |
| N=200 | 0.997±0.020 | 1.853±0.059 | 1.86× | **YES** | 5.4 / 13.8 |
| N=500 | 0.987±0.017 | 1.783±0.034 | 1.81× | **YES** | 6.3 / 15.8 |
| N=1000 | 0.996±0.008 | 1.785±0.022 | 1.79× | **YES** | 6.8 / 18.0 |
| N=2000 | 0.990±0.004 | 1.782±0.017 | 1.80× | **YES** | 7.5 / 22.6 |
| lag window=10 | 0.990±0.004 | 1.754±0.011† | 1.77× | **YES** | 7.5 / 10.0 |
| lag window=20 | 0.990±0.004 | 1.781±0.014† | 1.80× | **YES** | 7.5 / 19.2 |
| lag window=30 | 0.990±0.004 | 1.781±0.015 | 1.80× | **YES** | 7.5 / 19.4 |
| lag window=40 | 0.990±0.004 | 1.782±0.017 | 1.80× | **YES** | 7.5 / 22.6 |

**Computational cost (score evaluations and wall time reported separately)**

| condition | N | dt | steps T | max_lag | score evals | wall (s) |
|---|---|---|---|---|---|---|
| dt=0.005 | 2000 | 0.005 | 10000 | 8000 | 2.000e+08 | 366.3 |
| baseline_reproduction | 2000 | 0.01 | 5000 | 4000 | 1.000e+08 | 92.6 |
| dt=0.02 | 2000 | 0.02 | 2500 | 2000 | 5.000e+07 | 24.9 |
| N=200 | 200 | 0.01 | 5000 | 4000 | 1.000e+07 | 13.6 |
| N=500 | 500 | 0.01 | 5000 | 4000 | 2.500e+07 | 25.0 |
| N=1000 | 1000 | 0.01 | 5000 | 4000 | 5.000e+07 | 49.2 |

(The horizon study re-uses the stored baseline ρ² curve and costs no extra score evaluations.)

**Ordering S_ellipse > S_Gaussian preserved in every cell, both rules, all seeds: True**

The existing four-sweep Δt-robustness result (Appendix Fig. A3) is cited, not re-presented here.
