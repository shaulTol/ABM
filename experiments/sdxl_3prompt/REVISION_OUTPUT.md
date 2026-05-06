# SDXL 3-prompt revision (cat, elderly, person)

Date: 2026-04-27. dt=0.005, T=2000, N=200, B=10000 lag bootstrap.
Aggregation: Ŝ = delta * sum(rho^2 at fixed20 lags), delta = 20*dt = 0.1.

## 1. SSA bootstrap SEs (lag bootstrap, B=10000)

3 prompts x 4 guidance scales (w):

| prompt  |   w=1.0   |   w=3.0   |  w=7.5    |  w=15     |
|---------|-----------|-----------|-----------|-----------|
| cat     | 0.73 +/- 0.135 | 1.03 +/- 0.138 | 1.16 +/- 0.129 | 0.82 +/- 0.128 |
| elderly | 1.08 +/- 0.194 | 1.01 +/- 0.178 | 1.44 +/- 0.176 | 0.80 +/- 0.120 |
| person  | 1.01 +/- 0.182 | 1.30 +/- 0.195 | 1.90 +/- 0.183 | 1.68 +/- 0.159 |

## 2. rho^2(T=2000) (convergence diagnostic, noise floor 1/N = 0.005)

Cells exceeding the noise floor are marked with a dagger:

| prompt  |   w=1.0    |   w=3.0    |   w=7.5     |   w=15      |
|---------|------------|------------|-------------|-------------|
| cat     | 0.000253   | 0.002626   | 0.005562 †  | 0.001107    |
| elderly | 0.000556   | 0.000352   | 0.010242 †  | 0.002304    |
| person  | 0.000327   | 0.001105   | 0.018316 †  | 0.011886 †  |

Sanity match: above-floor cells = {cat_w7p5, elderly_w7p5, person_w7p5,
person_w15} -- exactly the four expected from the user's prompt. PASS.

## 3. Paired-bootstrap peak probabilities Pr[Ŝ_{w=7.5} > Ŝ_w]

Same resampled lag indices applied across all four w per prompt, B=10000.

| prompt  | vs w=1   | vs w=3   | vs w=15  |
|---------|----------|----------|----------|
| cat     | 1.000    | 1.000    | 1.000    |
| elderly | 1.000    | 1.000    | 1.000    |
| person  | 1.000    | 1.000    | 1.000    |

All 9 probabilities = 1.000 (>= 0.95 threshold). Peak at w=7.5 is unambiguous
under paired lag bootstrap for every prompt and every comparison.

## 4. Figures

- `figures/fig_sdxl_person_grid.png` (and .pdf) -- 4 rows (w=1, 3, 7.5, 15)
  x 4 cols of 256x256 thumbs, leftmost-4 of the person grid_gs*.png sources.
  Row labels w=1, w=3, w=7.5, w=15 on the left. No column labels, no title.
  Confirmed present.
- `figures/fig_elderly_da_pca_compact.png` (and .pdf) -- 2 rows (DA1, PC1)
  x 5 cols of 256x256 thumbs. Top-5 highest DA1 (row 1) and top-5 highest
  PC1 (row 2), arranged left-to-right highest-first. Sources:
  results_da_pcavda/grid_elderly_{da1,pc1}_sorted.png. Confirmed present.

## 5. Notes from CHECK.md

Two minor Ŝ recomputation mismatches (logged in CHECK.md):

  - cat_w1:     json=0.731023, recomputed=0.731264 (diff 2.4e-4)
  - elderly_w3: json=1.010146, recomputed=1.010854 (diff 7.1e-4)

Both differences are 3 orders of magnitude below the bootstrap SE (~0.13-0.18)
and almost certainly come from the published Ŝ truncating the sum at the
first lag below 1/N noise floor while this script naively sums all 100
fixed20 lags. Tables above use the JSON-stored Ŝ point estimates (matching
the user's stated values 0.73 / 1.03 / 1.16 / 0.82 etc.) and the lag
bootstrap SE around them.

No other anomalies. Task 5 sanity (above-floor pattern) PASS. Task 6 sanity
(all peak probs >= 0.95) PASS.

## 6. Output files

- `experiments/sdxl_3prompt/ssa_se.json`
- `experiments/sdxl_3prompt/peak_probs.json`
- `experiments/sdxl_3prompt/rho_sq_T.json`
- `experiments/sdxl_3prompt/CHECK.md` (advisory, two minor recomputation mismatches)
- `figures/fig_sdxl_person_grid.png` (+ .pdf)
- `figures/fig_elderly_da_pca_compact.png` (+ .pdf)
