# SDXL SSA Experiment

Measure SSA and DA directions in SDXL's latent-CLIP embedding space (512D).

## What This Measures

We run Langevin dynamics using the SDXL denoising score (Tweedie formula + classifier-free guidance) and compute autocorrelation structure in a Latent-CLIP embedding space. The key question: does SSA capture multimodality in text-to-image diffusion models?

## Pipeline

1. **SDXLScore**: Compute score via Tweedie formula from UNet noise prediction, with CFG
2. **Langevin dynamics**: Run trajectories in SDXL's 4×64×64 latent space
3. **Latent-CLIP embedding**: Project latent snapshots to 512D (no pixel decoding needed)
4. **SSA + DA**: Compute autocorrelation metrics in the 512D space

## Scripts

- `scripts/compute_sdxl_ssa_v2.py` — Main computation (fp16 UNet, wandb logging)
- `scripts/visualize_results.py` — SSA plots from result JSONs
- `scripts/visualize_da.py` — DA direction analysis

## Running

```bash
# On cluster (salmon partition — needs L40S 48GB)
sbatch cluster/run_guidance_sweep.sh    # 7 guidance scales
sbatch cluster/run_base_vs_turbo.sh     # SDXL Base vs Turbo
sbatch cluster/run_dt_test.sh           # dt sensitivity check
```

## Current Results

### Guidance Sweep (prompt: "a photo of a dog")
| Guidance (w) | SSA | Entropy |
|---|---|---|
| 1.0 | 0.011 | -1219 |
| 2.0 | 0.012 | -1355 |
| 3.0 | 0.014 | -1408 |
| 5.0 | **0.015** | -1421 |
| 7.5 | 0.013 | -1413 |
| 10.0 | 0.012 | -1390 |
| 15.0 | 0.011 | -1336 |

SSA peaks at w=5 while entropy is monotonic — SSA != entropy confirmed.

### Base vs Turbo
SSA identical (0.014) despite huge entropy difference (-1157 vs -1435).

## Known Issues

- **dt sensitivity**: dt=0.02 may cause artificially fast mixing. dt=0.0001 shows rho²=0.387 at phys_time=0.25 vs ~0.25 with dt=0.02.
- **Micro-conditioning**: SDXL Base at 512×512 requires `original_size=(1024,1024)` and matching `time_ids`. Test pending (job 44367544).
- **DA1 not semantic**: DA1 direction does not correlate with gender/age/expression/lighting.
- **Latent-CLIP**: Fixed 8×8 positions — only works with 64×64 VAE latents, not 128×128.
- **All SSA values very low** (~0.01-0.015): May need different prompt/model or finer dt.

## Environment

Requires `latent_clip` + `open_clip_torch==3.3.0` in addition to base requirements. After installing latent_clip (which pins diffusers==0.28.0), reinstall `diffusers==0.30.0`.

Set `HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface` and `HF_TOKEN_PATH=/dev/null`.
