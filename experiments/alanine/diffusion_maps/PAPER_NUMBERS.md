# Diffusion Maps — Paper Numbers

## Setup

- N = 2000 subsampled frames, 45D heavy-atom distances, 5 seeds
- α = 1 normalization (Coifman-Lafon)
- Bandwidth: ε* = 0.0655, selected via log-log criterion (standard textbook choice)

## Numbers for paper

- ψ₂ |r(ψ)|: **0.615 ± 0.023** (mean ± SD across 5 seeds)
- ψ₃ |r(φ)|: **0.659 ± 0.020** (mean ± SD across 5 seeds)
- Ordering ψ₂ ↔ ψ (slow), ψ₃ ↔ φ (fast): **yes**, at all 5 seeds

## Source

`results/results.json`, key `aggregate.eps_star`
