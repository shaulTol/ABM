# Phase 6: KDE Barrier Height Along DA2

**Date:** 2026-05-02
**Purpose:** Measure the KDE log-density barrier along the DA2 axis to determine whether the KDE density has a barrier that the canonical Langevin would need to cross.

---

## 1. Setup

- Same 2000 KDE centers, same h = 0.0091, same KDEScore class as the canonical pipeline.
- DA2 = second eigenvector of symmetrized C(τ = 10,000 ps).
- GMM fit on DA2 projections of 2000 centers:
  - Basin A: μ = −0.044, σ = 0.053, weight = 0.636
  - Basin B: μ = +0.081, σ = 0.027, weight = 0.364
  - Separation: |μ_B − μ_A| = 0.125 = 3.14 × mean_std

---

## 2. Three Barrier Estimates

### Method A: 45D KDE evaluated at actual data points (the correct test)

Evaluated `log f̂(x)` at all 2000 KDE centers and plotted vs their DA2 projection.

**Result: The 45D KDE log-density is essentially constant across all 2000 centers.**

- log_f range: [162.32, 162.36] — a span of **0.04 nats**
- Binned mean log_f shows **zero barrier** (< 0.01 kT) between basins A and B

This means the 45D KDE assigns the same density to points in basin A, basin B, and the region between them. There is **no barrier in the 45D KDE along DA2** as seen by the data.

### Method B: 45D conditional-mean sweep (misleading)

For each DA2 value t, constructed a 45D query point using the conditional mean of the 50 nearest centers in DA2-projection space, then replaced the DA2 component with t.

- Barrier A→B: 25.1 kT
- Barrier B→A: 31.4 kT

**This is an artifact.** The conditional-mean path passes through regions of 45D space where no data exists (between the clusters). The KDE density drops precipitously there — not because of a barrier, but because of the **curse of dimensionality**: in 45D, the density between clusters is negligible simply due to volume, not because of a potential barrier.

### Method C: 1D KDE marginal (baseline)

Projected the 2000 centers onto DA2 and fit a 1D KDE with the same bandwidth h = 0.0091.

- Barrier A→B: 0.56 kT
- Barrier B→A: 0.66 kT

This shows the 1D marginal density has a shallow but real barrier (< 1 kT) between the two basins.

---

## 3. Why Method A Gives Zero Barrier

In 45D with N = 2000 centers and h = 0.0091:

- Each kernel K_h(x − x_i) has effective support in a ball of radius ~few × h ≈ 0.03–0.05 in each coordinate.
- The log-density at any center x_i is dominated by its nearest neighbors — the ~10–50 closest centers within a radius of ~h from x_i.
- Because the data lives on a low-dimensional manifold in 45D, the density at each center is determined primarily by the local density of neighbors on the manifold, not by the global basin structure along DA2.
- The two basins (A and B) have similar local neighbor densities on the manifold, so `log f̂(x)` is approximately the same in both basins.

**The 45D KDE does not create a potential barrier between the two phi-basins.** The basins are separated in DA2 projection space, but in full 45D space, the KDE density surface has no ridge or barrier between them — just a smooth density that is locally similar everywhere the data lives.

---

## 4. Implications for Canonical Langevin

The canonical Langevin diffusion is:

    dX_t = (σ²_f / 2) · ∇log f̂(X_t) dt + σ_f · dW_t

If `log f̂(x)` is essentially flat across the two DA2-basins (barrier ≈ 0), then the drift `∇log f̂` has no net force opposing transitions between basins along DA2. The canonical Langevin diffusion mixes freely between the basins along DA2 — just as it would for a Gaussian. This explains why observed λ₂ < null λ₂: the canonical Langevin on the 45D KDE mixes along DA2 at least as fast as (and perhaps faster than) the OU process on the matching Gaussian.

**The contradiction from Phase 4 is resolved.** The physical reasoning "bimodal with barrier must mix slower than Gaussian without barrier" is correct — but the premise is wrong. The 45D KDE does NOT have a barrier along DA2. The samples are bimodal in DA2 projection, but the 45D KDE density surface has no corresponding barrier. With h = 0.0091 in 45D, the KDE oversmooths the barrier: each kernel covers a ball of radius ~h in every direction, and in 45D the kernels from both basins overlap enough (via the 43 orthogonal directions) that the density between basins never dips.

---

## 5. Summary Table

| Method | Barrier A→B (kT) | Barrier B→A (kT) | Interpretation |
|---|---|---|---|
| 45D KDE at data points | **≈ 0** | **≈ 0** | No barrier in the actual density landscape |
| 45D conditional-mean path | 25.1 | 31.4 | Artifact: path through empty 45D regions |
| 1D KDE marginal | 0.56 | 0.66 | 1D projection shows weak bimodality |

---

## 6. Verdict

**Barrier < 0.01 kT in 45D (effectively zero).** This is below the 2 kT threshold for "shallow barrier explains the result."

**The 45D KDE oversmooths the φ-barrier.** The bandwidth h = 0.0091 (= Scott/2 in 45D) is too large to resolve the barrier between the two φ-basins in full 45D space. The samples are bimodal (dip test p < 0.0001), but the KDE density assigns equal density to both basins and the inter-basin region. The canonical Langevin diffusion therefore has no barrier to slow its mixing along DA2, and its λ₂ is consistent with — or even smaller than — the Gaussian null's λ₂.

**This is not a bug in the framework.** It is a fundamental limitation: in 45D with N = 2000 and isotropic Gaussian KDE, the bandwidth needed to resolve inter-basin barriers along the second-slowest mode is smaller than what Scott's rule (or half of it) provides. DA1 (ψ-mode) succeeds because its barrier is much deeper and wider — it survives the KDE smoothing. DA2 (φ-mode) fails because its barrier is narrower and gets washed out in 45D.

---

## PNG

- `PHASE_6_kde_barrier_along_DA2.png` — 3-panel: (1) scatter of log_f at centers vs DA2 projection, (2) conditional-mean sweep vs 1D KDE, (3) histogram of DA2 projections with GMM basin means.
