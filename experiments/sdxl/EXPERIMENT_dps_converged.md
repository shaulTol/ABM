# DPS Scribble Converged Comparison

## Setup

3 DPS scribbles × 3 dt settings = 9 conditions, all using SDXL Turbo + ControlNet-Scribble, cs=0.5, cfg=0.0, gen_steps=2, N=200, t_score=1.

| Scribble | dt=0.001, T=500 | dt=0.005, T=500 | dt=0.0001, T=5000 |
|----------|-----------------|-----------------|-------------------|
| Face (zeta5) | `dps_face_dt001` | `dps_face_dt005` | `dps_face_converged` |
| First (seed 1) | `dps_first_dt001` | `dps_first_dt005` | `dps_first_converged` |
| Second (seed 2) | `dps_second_dt001` | `dps_second_dt005` | `dps_second_converged` |

## Results (converged, dt=0.0001)

| Experiment | PC1-gender | DA1-gender | DA1 gap | SSA | SSA_gender |
|------------|-----------|-----------|---------|-----|------------|
| Face dt=1e-4 | 0.904 | **-0.904** | 2.37x | 0.442 | 0.954 |
| First dt=1e-4 | -0.434 | **0.622** | 1.60x | 0.472 | 0.886 |
| Second dt=1e-4 | 0.777 | **-0.920** | 1.44x | 0.451 | 0.948 |

## Full dt comparison

| Experiment | PC1-gender | DA1-gender | DA1 gap | SSA | SSA_gender |
|------------|-----------|-----------|---------|-----|------------|
| Face dt=0.001 | 0.904 | -0.901 | 2.36x | 0.429 | 0.956 |
| First dt=0.001 | -0.434 | -0.592 | 1.58x | 0.454 | 0.879 |
| Second dt=0.001 | 0.777 | -0.914 | 1.40x | 0.432 | 0.928 |
| Face dt=0.005 | 0.904 | 0.911 | 2.83x | 0.115 | 0.471 |
| First dt=0.005 | -0.434 | -0.377 | 2.04x | 0.117 | 0.329 |
| Second dt=0.005 | 0.777 | -0.145 | 1.41x | 0.111 | 0.354 |
| Face dt=1e-4 | 0.904 | -0.904 | 2.37x | 0.442 | 0.954 |
| First dt=1e-4 | -0.434 | 0.622 | 1.60x | 0.472 | 0.886 |
| Second dt=1e-4 | 0.777 | -0.920 | 1.44x | 0.451 | 0.948 |

## Key Observations

1. **DA1 beats PC1 at finding gender** — especially for "First" (|0.62| vs |0.43|) and "Second" (|0.92| vs |0.78|). DA directions recover the slow mode (gender) more cleanly than PCA, which mixes in fast-decaying variance.

2. **dt=0.001 vs dt=0.0001**: Nearly identical SSA and correlations — dt=0.001 was already adequate for these scribbles. The converged run confirms the earlier results.

3. **dt=0.005 over-mixes**: SSA drops to ~0.11 (vs ~0.45), DA1-gender correlation degrades badly for First and Second scribbles.

4. **SSA_gender >> SSA** in all converged conditions: Gender-sliced SSA is 0.88–0.95 while full SSA is 0.44–0.47, confirming gender is the dominant slow mode.

5. **"First" scribble is problematic**: DA1-gender only |0.62| (vs |0.90+| for Face and Second). The DA1-sorted grid shows the bottom row contains corrupted/abstract non-face images — the scribble produces generation failures that dominate DA1 instead of gender.

## Issues to Address

1. **Non-person artifacts in "First" scribble**: Many generated samples are not faces at all — they're abstract blobs/textures. This corrupts the DA analysis. Need to either:
   - Filter out non-face samples before SSA/DA
   - Use a different scribble that produces more consistent faces
   - Add a face-detection / CLIP-based quality filter

2. **Gender is binary, not continuous**: The gender distribution is clearly bimodal (two clusters in CLIP space). For a more compelling demonstration of DA directions, we need distributions where gender (or the main slow variable) varies **continuously** — e.g., age, skin tone, or lighting angle. A continuous slow mode would better showcase DA's ability to find the slowest-decorrelating direction vs PCA finding max-variance.

3. **Need more diverse prompts**: Current setup always produces ~50/50 male/female split. Should explore prompts/scribbles that produce richer multimodal structure (e.g., multiple identity clusters, clothing variation, expression variation) where DA can distinguish multiple slow modes.

## Next Steps

- [ ] Debug "First" scribble: filter non-face samples or replace scribble
- [ ] Find prompts/conditions with **continuous gender-like variation** (not binary)
- [ ] Run SSA+DA on age/expression-dominant conditions (e.g., the "street portrait" egg experiment)
- [ ] Compare entropy across conditions (converged results include full entropy suite)

## Files

- Results: `results_clip_da/clip_da_dps_{face,first,second}_converged.{json,npz}`
- Scatter PDF: `results_clip_da/dps_scatter_all.pdf`
- SLURM: `cluster/run_dps_converged.sh`, `cluster/run_dps_report.sh`
- Jobs: 44400449 (array), 44402816 (report)
