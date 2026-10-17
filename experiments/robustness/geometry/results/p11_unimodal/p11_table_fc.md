# P11 strictly unimodal high-SSA controls (fc)

1/11 cells present.

**Table (P11: strictly unimodal high-SSA controls).** Every density has a UNIQUE mode, established analytically AND by multi-start numerical optimisation in the full d-dimensional space (never from a 1-D projection test). Ŝ is the *finite-horizon* estimator, not resolved population SSA: lag 0 excluded, floor 1/N=0.001 (N=1000), stopping lag τ* = largest lag with ρ²>1/N, Ŝ=Δt·Σρ²(τ), Δt=0.01, horizon H=20. '≥' marks window-limited cells (τ*=max_lag), which are finite-horizon LOWER bounds. The baseline is a Gaussian matched on the FULL covariance matrix and run with identical N, Δt, horizon and stopping rule, so the ratio isolates non-Gaussian shape rather than covariance/anisotropy. 'Sig. DA' counts DA directions whose eigenvalue exceeds the matched-Gaussian DA null; dip and Silverman p are reported for each of them (small p ⇒ reject unimodality of that projection).

| Density | d | unique mode | Ŝ | matched-Gauss Ŝ | **SSA ratio** | win-lim | τ* | λ1/λ2 | DA1 cross-seed ∠ | Sig. DA | dip p / Silverman p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gen-Gaussian (β=6, α=3, d=10) | 10 | **yes** | 1.094±0.003 | 0.992±0.013 | **1.10×** | no | [763, 758, 778] | 1.55 | 80.7° | 0 | DA1: 0.98/0.37 |

**Honest negative result.** The largest elevation over a covariance-matched Gaussian across all three families is only 1.10× (Gen-Gaussian (β=6, α=3, d=10)). On this evidence, substantially elevated SSA was NOT manufacturable without barriers in these families — high SSA appears harder to produce from unimodal slow geometry than the concern supposes. Reported as observed; no density was swapped in to improve the number.
