### P2 numerical robustness (common 10D bimodal ladder, T_phys=20.0, 5 seeds)

PRIMARY SSA estimator: threshold=1/N, stopping lag = first lag tau with rho^2(tau) <= 1/N, included (paper eq:tmax_stopping), integrate through tau* (lag 0 excluded), Shat = dt * sum rho^2. 1/(Nd) is a labeled secondary sensitivity only (see JSON summary_secondary_1Nd).
Baseline cell: N=200, dt=0.01. Point stability at representative rung separation=2.5. **† marks a WINDOW-LIMITED (finite-horizon) entry**: τ* reached the edge of the 20.0 lag window, so Ŝ is a finite-horizon quantity and is NOT a resolved population SSA value. `wl rungs` = how many of the 7 ladder rungs are window-limited.


**Step-size study (N=200 fixed, physical horizon AND physical lag window fixed at 20.0; smaller dt = more steps/score evals)**

| dt | steps T | max_lag | Ŝ(sep=2.5) mean±sd | CV | τ*(sep=2.5) | wl rungs | Spearman(Ŝ,MI) | Spearman(Ŝ,baseline) | score evals | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.005 | 4000 | 4000 | 1.0676±0.0186 | 0.017 | 6.82 | 0/7 | 0.957 | 1.000 | 2.800e+07 | 73.8 |
| 0.01 | 2000 | 2000 | 1.0598±0.0242 | 0.023 | 6.70 | 0/7 | 0.907 | 1.000 | 1.400e+07 | 20.3 |
| 0.02 | 1000 | 1000 | 1.0502±0.0299 | 0.028 | 6.90 | 0/7 | 0.914 | 1.000 | 7.000e+06 | 7.7 |

**Trajectory-count study (dt=0.01 fixed, threshold 1/N recomputed per N)**

| N | threshold 1/N | Ŝ(sep=2.5) mean±sd | CV | τ*(sep=2.5) | wl rungs | Spearman(Ŝ,MI) | Spearman(Ŝ,baseline) | score evals | wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 2.00e-02 | 1.0583±0.0430 | 0.041 | 4.64 | 0/7 | 0.757 | 0.929 | 3.500e+06 | 14.9 |
| 100 | 1.00e-02 | 1.0798±0.0171 | 0.016 | 5.92 | 0/7 | 0.864 | 0.857 | 7.000e+06 | 15.3 |
| 200 | 5.00e-03 | 1.0598±0.0242 | 0.023 | 6.70 | 0/7 | 0.907 | 1.000 | 1.400e+07 | 20.3 |
| 400 | 2.50e-03 | 1.0621±0.0202 | 0.019 | 8.28 | 1/7 | 0.943 | 0.964 | 2.800e+07 | 37.5 |

**Observed stopping lag τ* per rung (baseline cell N=200, dt=0.01; † = window-limited)**

| separation | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 4 |
|---|---|---|---|---|---|---|---|
| MI | 0.031 | 0.112 | 0.216 | 0.340 | 0.446 | 0.536 | 0.633 |
| τ* (phys) | 5.42 | 5.24 | 5.46 | 5.68 | 6.70 | 9.85 | 18.38 |
| Ŝ | 0.998 | 0.985 | 1.005 | 1.009 | 1.060 | 1.141 | 1.510 |

**max_lag cap validation (baseline cell, post-hoc; NO cap adopted)**

| max_lag | phys window | Spearman(Ŝ,MI) | max abs rel diff vs full | ordering changed |
|---|---|---|---|---|
| 2000 (full) | 20.00 | 0.964 | — | — |
| 500 (0.25×) | 5.00 | 0.964 | 0.186 | False |
| 1000 (0.50×) | 10.00 | 0.964 | 0.066 | False |
| 1500 (0.75×) | 15.00 | 0.964 | 0.017 | False |
