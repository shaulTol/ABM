# Table 4 Reconciliation: Paper Claims vs Run Artifacts

**Date**: 2026-05-03
**Scope**: Read-only audit — no files modified.

---

## 1. Files That Produced Table 1 (Paper's Table 4)

### SSA Results (dt=0.005 variant)

| Prompt | File (SSA) |
|--------|------------|
| cat | `experiments/sdxl/results_multiprompt_sweep/clip_da_mp_p0_cat_gs{1p0,3p0,7p5,15p0}_dt0p005.json` |
| mountains | `experiments/sdxl/results_multiprompt_sweep/clip_da_mp_p1_mountains_gs*_dt0p005.json` |
| flowers | `experiments/sdxl/results_multiprompt_sweep/clip_da_mp_p2_flowers_gs*_dt0p005.json` |
| street | `experiments/sdxl/results_multiprompt_sweep/clip_da_mp_p3_street_gs*_dt0p005.json` |
| food | `experiments/sdxl/results_multiprompt_sweep/clip_da_mp_p4_food_gs*_dt0p005.json` |
| elderly | `experiments/sdxl/results_multiprompt_sweep/clip_da_mp_p5_elderly_gs*_dt0p005.json` |
| car | `experiments/sdxl/results_multiprompt_sweep/clip_da_mp_p6_car_gs*_dt0p005.json` |
| person | `experiments/sdxl/results_guidance_sweep/clip_da_sweep_gs{1p0,3p0,7p5,15p0}_dt0p005.json` |

### Entropy Results (N=2000, independent generation)

| Prompt | File (Entropy) |
|--------|---------------|
| cat–car (7 prompts) | `experiments/sdxl/results_multiprompt_sweep/gen_mp_p{0-6}_{label}_gs*_entropy.json` |
| person | `experiments/sdxl/results_guidance_sweep/gen_sweep_gs*_entropy.json` |

### SLURM Scripts

| Phase | Script | Output dir |
|-------|--------|-----------|
| Quick gen (200 x₀ latents) | `cluster/run_multiprompt_sweep_quick.sh` | `results_multiprompt_sweep/` |
| Full gen (2000 images + entropy) | `cluster/run_multiprompt_sweep_gen.sh` | `results_multiprompt_sweep/` |
| Langevin + SSA (dt=0.0005) | `cluster/run_multiprompt_sweep_ssa.sh` | `results_multiprompt_sweep/` |
| T=5000 dt=0.005 (includes T=2000 dt=0.005 as a by-product) | `cluster/run_T5000_dt0p005.sh` | `results_T5000/` (but dt0p005 JSONs also in multiprompt_sweep) |
| Person prompt (dt=0.005) | `cluster/run_guidance_sweep_ssa.sh` (original dt=0.0005) + T5000 variant | `results_guidance_sweep/` |

### Python Entry Points

| Script | Purpose |
|--------|---------|
| `scripts/run_ssa_da_clip.py` | Langevin dynamics + CLIP SSA/DA computation |
| `scripts/generate_and_entropy.py` | Generate N=2000 images + compute entropy suite |
| `scripts/plot_multiprompt_sweep.py` | Visualization (reads unsuffixed dt=0.0005 files — **NOT** the dt=0.005 files used in Table 1) |

---

## 2. Parameter Comparison: Paper Claims vs Ground Truth

| Parameter | Paper claims | Actual (from JSON metadata) | Match? |
|-----------|-------------|---------------------------|--------|
| **Δt** | 0.0005 | **0.005** (in `_dt0p005.json` files matching Table values) | **MISMATCH** |
| **T (steps)** | 2000 | 2000 | ✓ |
| **T_phys** | 1.0 s (= 2000 × 0.0005) | **10.0 s** (= 2000 × 0.005) | **MISMATCH** |
| **N (trajectories)** | 200 | 200 | ✓ |
| **Snapshot schedule** | "111 snapshots, combined log + fixed-interval, fixed-20 variant" | 111 snapshots (log∪fixed-20), `ssa_fixed20` over 100 lags at every 20 steps | ✓ |
| **Entropy: N images** | 2000 | 2000 (separate generation phase via `generate_and_entropy.py`) | ✓ |
| **Entropy: kNN k** | 5 | 5 (field: `entropy.knn.k5.50`) | ✓ |
| **Entropy: PCA d** | 50 | 50 (field key: `"50"`) | ✓ |
| **Model** | SDXL Base 1.0 | `stabilityai/stable-diffusion-xl-base-1.0` | ✓ |
| **DDIM steps** | 50 | 50 (`gen_steps: 50`) | ✓ |
| **Resolution** | 1024×1024 | `height: 1024, width: 1024` | ✓ |
| **CLIP model** | (not specified) | `openai/clip-vit-large-patch14` (768D) | — |
| **Noise floor** | (not specified) | `free_conv`, value ≈ 1.3e-4 | — |
| **SSA metric** | (not specified) | `ssa_fixed20`: Ŝ = dt × Σ ρ²(fixed-20 lags) with free-conv truncation | — |
| **t_score** | (not specified) | 1 (Tweedie at minimum noise) | — |
| **Guidance for scoring** | (not specified) | Same as generation (gs ∈ {1, 3, 7.5, 15}) | — |

---

## 3. The dt Discrepancy — Detailed Analysis

### What happened

1. **Original run** (SLURM: `run_multiprompt_sweep_ssa.sh`): dt=0.0005, T=2000 → T_phys=1.0s. Produces unsuffixed files: `clip_da_mp_p0_cat_gs7p5.json`.

2. **dt=0.005 variant** (SLURM: `run_T5000_dt0p005.sh`, written 2026-04-26): dt=0.005, T=5000 (but T=2000 checkpoint exists as by-product). Produces `_dt0p005.json` files.

3. **SSA repatching** (2026-04-26): All 90 JSONs repatched from saved `rho_sq_all` to fix the Ŝ definition. Both dt=0.0005 and dt=0.005 files received new `ssa_fixed20` values (same mtime: 2026-04-26 09:14:27).

4. **PAPER_REPO_MAP.md** (line 58) states: "dt=0.005, T=2000, N=200" — this matches the `_dt0p005` files.

5. **MEMORY.md canonical table** matches `_dt0p005` values exactly (e.g., person=1.90, elderly=1.44, cat=1.16).

### Which is correct?

**The `_dt0p005.json` files are what the paper reports.** Ground truth:
- **dt = 0.005** (NOT 0.0005)
- T_phys = 10.0 s (NOT 1.0 s)

### Consequence

The paper appendix (`app:sdxl_guidance_details`) claims **dt=0.0005**, but the actual Table values come from **dt=0.005** files. The paper text is wrong by a factor of 10.

---

## 4. The Plotting Script Ambiguity

`plot_multiprompt_sweep.py` (line 42) reads:
```python
ssa_path = os.path.join(results_dir, f"clip_da_{name}.json")  # unsuffixed!
```

This reads **dt=0.0005** files (SSA ≈ 0.55–0.68 range), NOT the dt=0.005 files (SSA ≈ 0.65–1.90 range) that appear in Table 1.

**Implication**: If `plot_multiprompt_sweep.py` was run as-is to produce `fig_sdxl_ssa_vs_entropy.png` (Figure 3), the figure shows dt=0.0005 values while Table 1 shows dt=0.005 values. **Figure 3 and Table 1 may be inconsistent.**

To verify, one would need to check `fig_sdxl_ssa_vs_entropy.png` SSA axis ranges. If bars top out around 0.6–0.7, it's from dt=0.0005. If they reach 1.0–1.9, it's from dt=0.005.

---

## 5. Person Prompt Location

The "person" prompt (`"a superrealistic professional photograph of a person"`) lives in a **different directory** than the other 7 prompts:
- SSA: `results_guidance_sweep/clip_da_sweep_gs*_dt0p005.json`
- Entropy: `results_guidance_sweep/gen_sweep_gs*_entropy.json`

The plotting script `plot_multiprompt_sweep.py` only searches `results_multiprompt_sweep/` and does NOT include person. There is **no script that generates a unified 8-prompt table from both directories**.

---

## 6. Entropy Pipeline Confirmation

Entropy is computed in a fully **separate phase** from SSA:
- **Generation**: `generate_and_entropy.py` generates 2000 fresh SDXL images, encodes them with real CLIP ViT-L/14, runs full entropy suite (Gaussian, kNN at k∈{3,5,10}, KDE; PCA dims ∈ {2,5,10,15,20,30,50}).
- **SSA**: `run_ssa_da_clip.py` resumes from 200 pregenerated x₀ latents, runs Langevin, embeds CLIP at snapshots.
- **No coupling**: The 2000 entropy images are generated independently from the 200 SSA trajectories (different seeds, different sample counts).

---

## 7. Summary of Mismatches

| # | Paper claim | Ground truth | Impact |
|---|-------------|-------------|--------|
| 1 | **Δt = 0.0005** | **Δt = 0.005** | Critical: factor-of-10 error in paper appendix. All SSA values in Table are from dt=0.005. |
| 2 | Implied: Figure 3 matches Table 1 | `plot_multiprompt_sweep.py` reads dt=0.0005 files | **Potential figure/table inconsistency** — needs visual check. |
| 3 | "8 prompts" in Table 1 | Person prompt in different directory; no unified 8-prompt script | Minor: data exists, just split across two result dirs. |

All other parameters (N=200, T=2000 steps, 50 DDIM, 1024×1024, kNN k=5 d=50, 2000 images for entropy) are confirmed correct.

---

## 8. Recommendations (not executed — read-only audit)

1. Fix paper appendix: Δt = 0.005 (not 0.0005). T_phys = 10.0 s.
2. Verify Figure 3 matches Table 1 by checking axis ranges in the PNG.
3. If Figure 3 uses dt=0.0005 values, either regenerate it from `_dt0p005` files or update Table 1 to use dt=0.0005 values (which would halve most SSA numbers).
4. Document that person prompt results live in `results_guidance_sweep/`, not `results_multiprompt_sweep/`.
