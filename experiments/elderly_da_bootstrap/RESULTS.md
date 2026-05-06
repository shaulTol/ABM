# Elderly DA1↔gender bootstrap CI

Point |ρ| = 0.8998 (paper: 0.9115). Bootstrap mean = 0.8810, SE = 0.0456, 95% CI = [0.7768, 0.9408].
B = 2000 replicates, all finite; sign-flip rate 50.2%, median λ₁/λ₂ = 1.80; P(|ρ|>0.8) = 0.936.
DA recomputed per resample, sign-aligned to point-estimate DA1; projection on the same fixed N=200 final-snapshot embeddings; gender labels = projection onto CLIP-ViT-L/14 text direction normalize("a photograph of a woman" − "a photograph of a man").

## LaTeX line (ready to paste)

> Bootstrap over the $N=200$ trajectories ($B=2000$, DA direction recomputed per replicate, sign-aligned to the point-estimate DA1) gives $|\rho| = 0.90$ [95\% CI: 0.78, 0.94].

## Files
- `results.json` — full numerical summary
- `bootstrap_histogram.png` — histogram of bootstrap |ρ|
- `run_bootstrap.py` — this script
