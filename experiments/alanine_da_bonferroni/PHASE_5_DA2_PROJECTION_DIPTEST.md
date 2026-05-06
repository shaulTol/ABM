# Phase 5: DA2 Projection Dip Test — Is the KDE Bimodal Along φ?

**Date:** 2026-05-02
**Purpose:** Test whether the KDE subsample (2000 centers) lacks bimodality along the DA2/φ direction, which would explain why observed λ₂ < null λ₂.

---

## 1. KDE Centers Reconstruction

The saving script (`save_alanine_C_tau_h_half_10k.py`, lines 70–76) uses a deterministic seed:

```python
np.random.seed(42)
sub_idx = np.random.choice(N_total, 10000, replace=False)
X_sub = all_dists[sub_idx]
center_idx = np.random.choice(10000, 2000, replace=False)
centers = X_sub[center_idx]
```

Reconstructed 2000 KDE centers: shape (2000, 45). phi range: [−3.085, 1.240].

Note: the KDE centers are used only to define the score function ∇log f̂_KDE. The simulator (`GenericLangevinSimulator`) receives `samples=X_sub_t` (the full 10,000-point subsample) for initial conditions and moment estimation. However, the KDE score is fit on the 2000 centers.

---

## 2. DA2 Direction

DA2 = second eigenvector of C_sym(τ) at τ = 10,000 ps, extracted from the stored `C_tau[37]` matrix.

- λ₁ = 1.2099e-02, λ₂ = 6.0956e-04 (matches canonical file)
- |r(DA2, φ)| = 0.8729 on all 750k frames
- |r(DA2, φ)| = 0.8840 on 2000 KDE centers

---

## 3. Dip Test Results

| Subset | Quantity | Dip D | Dip p | Mean | Std | Verdict |
|---|---|---|---|---|---|---|
| 2000 KDE centers | DA2 projection | 0.0287 | < 0.0001 | 0.0014 | 0.0750 | **bimodal** |
| 2000 KDE centers | φ dihedral | 0.0122 | 0.0457 | −1.678 | 0.671 | **borderline** |
| 750k frames (50k sub) | DA2 projection | 0.0180 | < 0.0001 | −0.0000 | 0.0757 | **bimodal** |
| 750k frames (50k sub) | φ dihedral | 0.0127 | < 0.0001 | −1.658 | 0.713 | **bimodal** |

---

## 4. Histograms

Four PNGs saved in this directory:

- `phase5_hist_a_centers_da2.png` — 2000 KDE centers projected onto DA2
- `phase5_hist_b_centers_phi.png` — 2000 KDE centers: φ dihedral
- `phase5_hist_c_all_da2.png` — All 750k frames projected onto DA2
- `phase5_hist_d_all_phi.png` — All 750k frames: φ dihedral
- `phase5_histograms.png` — Combined 2×2 panel

---

## 5. Interpretation

**The hypothesis is falsified.** The 2000 KDE centers ARE bimodal along the DA2 direction (dip p < 0.0001). The full 750k frames are also bimodal along DA2 (dip p < 0.0001). Both subsets are bimodal along raw φ (p = 0.046 for 2000 centers, p < 0.0001 for 750k).

The KDE subsample does not lack bimodality. The KDE density fit on these 2000 bimodal centers, with h = 0.0091, should encode a barrier along the φ direction. The DA2 direction (|r(φ)| = 0.88) is well-aligned with the φ barrier.

**This means the contradiction identified in Phase 4 — that observed λ₂ < null λ₂ despite the data being bimodal along DA2 — is NOT due to the KDE subsample being unimodal.** The φ barrier exists in the data, the DA2 direction finds it, and the KDE should represent it. Yet the canonical Langevin's second eigenvalue is smaller than the Gaussian null's second eigenvalue.

**The contradiction remains unresolved.** Possible remaining explanations:

1. **The KDE bandwidth h = 0.0091 oversmooths the φ barrier.** Even though the sample is bimodal, the KDE with this bandwidth may produce a density that is unimodal or has a very shallow barrier along φ. The dip test confirms the *sample* is bimodal, but the *KDE density* could still be unimodal if h is too large relative to the barrier width.

2. **The barrier height matters, not just bimodality.** Even if the KDE density is bimodal along φ, the barrier may be shallow enough that canonical Langevin crosses it frequently. A shallow barrier produces fast mixing along φ → low λ₂. The Gaussian null, lacking any barrier, has its second eigenvalue set by the OU decorrelation rate along the second principal direction of C₀ — which at short-to-medium lags can exceed the λ₂ of a weakly-barriered bimodal.

3. **The N(0, C₀) null is an overly generous null.** C₀ has λ₁/λ₂ = 1.91× — the null Gaussian has nearly as much variance along PC2 as PC1. This gives the null's OU process two slowly-decorrelating directions. A fairer null might be N(0, σ²_f · I), which would have much smaller null λ₂. Under the isotropic null, DA2 might pass.

Explanation (1) is directly testable by evaluating the KDE log-density along a φ-sweep. Explanation (3) is the most structurally important — it questions whether the N(0, C₀) null is the right null for this test.
