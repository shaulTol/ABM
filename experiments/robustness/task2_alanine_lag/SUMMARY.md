# Task 2: Alanine Dipeptide DA Lag Sensitivity

## Method

Pre-computed eigenvectors from the KDE-based convergence profile 
(28 geometric checkpoints, dt=0.1 ps, N=2000 trajectories, 
Scott bandwidth h=0.018). Eigenvectors from GMM-DA profile 
(30 checkpoints, dt=0.0001, N=5000) used for DA1/DA2 at each lag.

Noise floor bounds from Appendix H: conservative 
λ⁺_floor = σ²_f · (1+√(d/N))² and aggressive 
λ⁻_floor = σ²_f · (1-√(d/N))².

Bootstrap SEs at τ*=130.2 ps: 5000 replicates, 
resampling the 750,000 alanine frames with replacement.

## Lag Sensitivity Table

| τ (ps) | λ₁ | λ₂ | floor⁺ | gap | DA1 \|r(ψ)\| | DA1 \|r(φ)\| | DA2 \|r(ψ)\| | DA2 \|r(φ)\| | ∠(DA1,ref) | λ₂>floor? |
|--------|-----|-----|--------|-----|-----------|-----------|-----------|-----------|------------|-----------|
| 65.8 | 0.018683 | 0.004898 | 0.000928 | 3.8× | 0.6484 | 0.4309 | 0.1275 | 0.7948 | 5.3° | Yes |
| 92.5 | 0.017771 | 0.003024 | 0.000928 | 5.9× | 0.6612 | 0.3917 | 0.0675 | 0.8142 | 1.4° | Yes |
| 130.2 | 0.016876 | 0.001857 | 0.000928 | 9.1× | 0.6654 | 0.3778 | 0.0497 | 0.8213 | 0.0° | Yes |
| 183.2 | 0.015608 | 0.000732 | 0.000928 | 21.3× | 0.6729 | 0.3494 | 0.0122 | 0.8306 | 2.7° | **No** |
| 257.7 | 0.014056 | 0.000176 | 0.000928 | 80.0× | 0.6706 | 0.3586 | 0.0292 | 0.8427 | 1.9° | **No** |

## Bootstrap SEs at τ*=130.2 ps

| Metric | Point estimate | SE | 95% CI |
|--------|---------------|-----|--------|
| DA1 |r(ψ)| | 0.6654 | 0.0014 | [0.6626, 0.6682] |
| DA1 |r(φ)| | 0.3778 | 0.0009 | [0.3760, 0.3797] |
| DA2 |r(ψ)| | 0.0497 | 0.0011 | [0.0475, 0.0518] |
| DA2 |r(φ)| | 0.8213 | 0.0008 | [0.8198, 0.8228] |

## Assessment

**(a) ψ-slowest, φ-second ordering stable?** Yes — 
DA1 tracks ψ (|r(ψ)| > |r(φ)|) at all tested lags.
DA2 tracks φ (|r(φ)| > |r(ψ)|) at all tested lags.

**(b) Conservative noise floor failures:** 2 lag(s) have λ₂ below the conservative floor:
  - τ = 183.2 ps: λ₂ = 0.000732 < floor = 0.000928
  - τ = 257.7 ps: λ₂ = 0.000176 < floor = 0.000928

**(c) DA1 directional stability:** angles from reference (τ*=130.2 ps) range from 0.0° to 5.3°.
