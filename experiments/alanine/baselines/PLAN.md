# PLAN: Alanine Baselines (Sparse K-means + Spectral Map)

Date: 2026-04-25
Goal: produce two new rows for tab:alanine with |r(psi)| and |r(phi)| ± SE for
sparse K-means and Spectral Map, plus a 2-panel Ramachandran figure.

## Data pipeline (reused from existing alanine experiment)

- Source: `experiments/da_directions/alanine_dipeptide/data/alanine-dipeptide-3x250ns-heavy-atom-distances.npz`
  and `alanine-dipeptide-3x250ns-backbone-dihedrals.npz`.
- Featurization: 45 pairwise heavy-atom distances (10 heavy atoms, C(10,2)=45).
- N_total = 750,000 frames. Frame interval 1 ps.
- Ground truth: dihedrals phi (col 0), psi (col 1) in radians.
- Sanity check: PCA1 must reproduce |r(psi)| ~ 0.563, |r(phi)| ~ 0.595.
  If it does not match, STOP and debug before continuing.

NOTE FROM USER (2026-04-25): the alanine baseline rerun is in flight. We set up
the environment and code now; the final numbers will be produced once the
rerun results are available. The data file used here is the canonical MDShare
file already on disk; the rerun is on the DA side, so the 45-d feature matrix
fed to sparse K-means and Spectral Map does not change.

## Method 1: Sparse K-means (Witten and Tibshirani 2010)

- K=3 (alpha_R, beta, alpha_L equivalents in the dihedral-distance featurization).
- Tune s by gap statistic on a 10-point grid in [1.1, sqrt(45)] = [1.1, 6.708].
- Min-Python implementation (no rpy2): block-coordinate alternation.
  - Inner loop: weighted K-means with `sklearn.cluster.KMeans` on X * sqrt(w),
    iterate until cluster labels stable.
  - w-update: closed-form soft-thresholding to satisfy ||w||_1 <= s,
    ||w||_2 <= 1, w >= 0 (Witten and Tibshirani Algorithm 3.1).
- Subsample N_fit = 20,000 frames for sparse K-means fitting (fitting on 750k
  is wasteful; the cluster geometry converges well before that).
- "Top direction" = w* normalized to unit L2; project all 750k samples;
  Pearson r against psi and phi.
- Sanity: w* must be sparse (<=15 nonzeros). If dense, flag in results JSON.

Bootstrap: B=5000 frame-level resamples. Each replicate refits sparse K-means
(K-means inner loop initialized with k-means++, single restart for speed).
Replicates run on the N_fit=20k subsample for tractability; the SE reflects
finite-sample variance of the fit, not of the 750k correlation. Note this in
results.json.

## Method 2: Spectral Map (Rydzewski reference implementation)

- Source: PLUMED-NEST plumID:24.005 (Zenodo 10678142). Reference file
  `spectral_map_reference.py` saved at the experiment root. Cite this version.
- Default architecture: [45, 200, 100, 50, 2] with ELU activations.
- Adam, lr=0.001, LinearLR 1.0->0.1 over 50 iters, n_epochs=100, batch_size=1000.
- alpha=0.5, n_eigen=50, n_gap=2, n_neigh=0.5 (defaults from reference impl).
- Train on N_fit = 50,000 frames (subsample for tractability and to match the
  reference's CLN025 batch counts). 80/20 train/valid split.
- After training, embed all 750k frames with the trained NN; SM1 = first
  embedding coordinate, SM2 = second.
- "Top direction" = SM1 (nonlinear; noted as such in results.json).

Bootstrap: B=5000 frame-level resamples on (SM1_i, psi_i) and (SM1_i, phi_i)
pairs only. Model fixed (refitting 5000 times is intractable on a 4-layer NN).
Note this distinction in results.json.

## Sanity checks

1. PCA1 |r(psi)| ~ 0.563, |r(phi)| ~ 0.595 (matches existing PCA row).
2. Sparse K-means: ARI > 0.9 across 3 random inits at the chosen s.
3. Sparse K-means: nnz(w*) <= 15.
4. Spectral Map: training loss (spectral gap, maximized) increases.
5. Spectral Map: SM1 has non-trivial variance (>1e-4).

## Output structure

```
experiments/alanine_baselines_sparse_kmeans_spectralmap/
  PLAN.md
  spectral_map_reference.py     # extracted from PLUMED-NEST plumID:24.005
  run.py                        # main driver
  sparse_kmeans.py              # Witten and Tibshirani implementation
  venv/                         # local venv (torch, sklearn, numpy, scipy, matplotlib, mdshare)
  results/
    results.json
    fig_alanine_baselines_sparse_kmeans_spectralmap.png
    SUMMARY.md
```

## Open questions / defensible defaults

- Sparse K-means N_fit: chose 20k. With 750k frames the fit would be slow and
  the geometry already converges at 20k. Bootstrap SE captures fit variability.
- Spectral Map N_fit: chose 50k for tractability. Reference impl uses 100-epoch
  defaults that were tuned for the Muller-Brown / CLN025 examples in plumID:24.005.
- Spectral Map architecture: kept default [200, 100, 50, 2] from the reference.
  Task brief says "default architecture and training hyperparameters."
- s grid for sparse K-means: 10 log-spaced values in [1.1, sqrt(45)]. The lower
  bound 1.1 is standard from Witten and Tibshirani (s=1 collapses to a single
  feature; s>1 needed to allow >1 nonzero weight).
- B=5000: matches existing rows per task brief.

## Plumbing notes

- venv lives inside the experiment subfolder, NOT ssa_env.
- No git operations.
- No em-dashes.
- All paths absolute or relative to this experiment folder.
