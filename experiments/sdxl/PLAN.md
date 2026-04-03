# Plan: Refactor SDXL SSA/DA into Reusable Components with Two Embedding Modes

## Context

Latent-CLIP's text encoder is broken (Spearman < 0.2 vs standard CLIP). The current pipeline computes SSA/DA in LC space (512D) where DA directions can't be interpreted semantically. We need a modular architecture where SSA/DA can run in either LC space (fast, existing) or real CLIP space (768D, interpretable DA directions).

**Key constraint:** Do NOT modify `compute_sdxl_ssa_v2.py`. Import from it.

## Files Created

All in `experiments/sdxl/`:

| File | Purpose |
|------|---------|
| `scripts/sdxl_components.py` | Reusable module: embedders, SSA/DA math, text probes, viz |
| `scripts/interpret_da_clip.py` | Post-hoc interpretation of existing LC results |
| `scripts/run_ssa_da_clip.py` | Full CLIP-mode SSA/DA pipeline |
| `scripts/visualize_clip_da_sweep.py` | Cross-mode comparison plots |
| `cluster/run_interpret_da.sh` | SLURM: post-hoc interpretation |
| `cluster/run_clip_da.sh` | SLURM: CLIP-mode smoke test |
| `cluster/run_clip_da_sweep.sh` | SLURM: guidance sweep in CLIP mode |

## Implementation Order

1. **`run_interpret_da.sh`** — Quick win: what does LC-DA1 mean in CLIP text space? (~3 min)
2. **`run_clip_da.sh`** — Smoke test: N=200, T=500, dt=0.0001, Turbo + face prompt (~1h)
3. Verify: rho decays, C(0) PD, deterministic, SSA in [0,1]
4. **`run_clip_da_sweep.sh`** — Full guidance sweep, SDXL Base (~4h/job)
5. **`visualize_clip_da_sweep.py`** — Generate comparison figures

## Verification Checklist

- [ ] `RealCLIPEmbedder`: same latent twice → identical embedding
- [ ] `RealCLIPEmbedder`: output L2-normalized, shape (N, 768)
- [ ] `compute_ssa_da`: rho(0) = 1.0 (by construction after centering)
- [ ] rho_CLIP(tau) decays with tau
- [ ] C_CLIP(0) positive definite
- [ ] `TextProbeAnalyzer`: known gendered images → high Spearman on gender probe
- [ ] Grid PNGs show visually distinct top/bottom samples
- [ ] Both modes produce SSA in comparable ranges
