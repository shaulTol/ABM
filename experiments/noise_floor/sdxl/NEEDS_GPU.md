# SDXL Noise Floor — GPU Requirements

## What was done locally
- Computed analytic floor from cached CLIP embeddings (sigma_f_sq)
- Computed conservative permutation floor (x0 vs shuffled x0)
- Applied floor criterion to cached eigenvalue profiles (20 lags, T=2500)

## What needs GPU re-run
For the **exact permutation floor at each lag**, we need CLIP embeddings
of x_tau at each snapshot lag. Currently only raw latent-space pairs are saved
(16384D), not CLIP embeddings. The pipeline would be:

1. Load Langevin checkpoint (x_tau latents at each snapshot)
2. VAE-decode each x_tau to pixel space (requires SDXL VAE, ~4GB VRAM)
3. CLIP-encode to get 768D embeddings
4. Compute permutation floor at each lag

### Script changes needed
In `run_ssa_da_clip.py`, add `--save_clip_snapshots` flag to save CLIP embeddings
(not just latents) at each snapshot. This adds ~200×768×4 bytes × 20 lags ≈ 46 MB.

### Cluster command
```bash
# After adding --save_clip_snapshots flag:
sbatch cluster/run_ssa_da_v9_T500_dt001_v2.sh  # or similar
```
