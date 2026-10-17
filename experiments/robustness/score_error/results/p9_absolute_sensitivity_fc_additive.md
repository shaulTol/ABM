**Absolute finite-horizon SSA under additive score error.** 50 random 10-D GMMs x 3 seeds, N=200, dt=0.01, T=100s, 1/N floor. The perturbation is a frozen, deterministic, state-dependent score-error field (random Fourier features) calibrated so its per-coordinate RMS equals a fraction alpha of the baseline score RMS s0; the same field is reused at every strength with only its amplitude scaled. Each perturbed run is paired with the exact-score run for the SAME GMM and seed. Statistics are |relative change| over **resolved** pairs (neither cell window-limited); window-limited cells are counted separately because their SSA is a finite-horizon lower bound.

| Additive level | median \|rel err\| | IQR | p90 | signed median | ratio (med) | within 10% | window-limited |
|---|---|---|---|---|---|---|---|
| alpha=0.1 x s0 | 0.2% | 0.1%-0.6% | 2.3% | +0.1% | 1.001 | 100.0% | 24/150 |
| alpha=0.25 x s0 | 0.6% | 0.4%-1.1% | 3.6% | +0.4% | 1.004 | 98.4% | 25/150 |
| alpha=0.5 x s0 | 1.7% | 1.1%-2.8% | 6.9% | +1.4% | 1.014 | 93.0% | 22/150 |
| alpha=1.0 x s0 | 5.2% | 3.8%-7.2% | 13.2% | +4.5% | 1.045 | 84.5% | 21/150 |
| alpha=2.0 x s0 | 12.3% | 9.2%-14.9% | 29.0% | +10.9% | 1.109 | 29.5% | 21/150 |

The error is **not one-signed**: at alpha=1.0 the signed median relative change is +5% (low third) / +5% (mid) / -6% (high), i.e. the high baseline-SSA third flips sign (only 36% of its cells are inflated), so no single bias correction applies.

**Stratified by baseline SSA (median |rel err|, resolved pairs).** Terciles of the 50 GMMs by across-seed mean baseline SSA.

| Additive level | low third | middle third | high third |
|---|---|---|---|
| alpha=0.1 x s0 | 0.1% (n=48) | 0.2% (n=48) | 1.4% (n=30) |
| alpha=0.25 x s0 | 0.4% (n=48) | 0.6% (n=48) | 2.4% (n=29) |
| alpha=0.5 x s0 | 1.5% (n=48) | 1.9% (n=48) | 4.7% (n=32) |
| alpha=1.0 x s0 | 4.9% (n=48) | 4.6% (n=48) | 8.8% (n=33) |
| alpha=2.0 x s0 | 12.3% (n=48) | 11.4% (n=48) | 17.2% (n=33) |

**Value vs ranking, on one scale.** Ranking loss is 1 - Spearman(alpha)/Spearman(exact), against an exact-score Spearman of +0.877.

| Additive level | median value error | p90 value error | Spearman(SSA, MI) | ranking loss |
|---|---|---|---|---|
| alpha=0.1 x s0 | 0.2% | 2.3% | +0.872 | 0.6% |
| alpha=0.25 x s0 | 0.6% | 3.6% | +0.874 | 0.4% |
| alpha=0.5 x s0 | 1.7% | 6.9% | +0.865 | 1.4% |
| alpha=1.0 x s0 | 5.2% | 13.2% | +0.832 | 5.1% |
| alpha=2.0 x s0 | 12.3% | 29.0% | +0.741 | 15.6% |

**Does absolute SSA degrade before the ranking?** The 'absolute SSA degrades EARLIER than the ranking' headline does NOT survive on additive evidence alone. It rested on the covariance-smoothing family (at +0.25 x var: 21.7% median value error while Spearman was still +0.855 of an exact-score +0.878), which is excluded from this presentation. Under the additive frozen field the two degrade in step, and at the strongest level the ranking actually degrades slightly more than the value (12.3% median value error vs 15.6% ranking loss at alpha=2.0). What additive evidence does support is a tail statement: at alpha=2 x s0 the upper decile of cells moves by 29.0% even though the ranking retains 84% of its exact-score Spearman, so individual reported SSA magnitudes can shift materially in a minority of cases where the ordering still looks intact.

**Conservative summary.** On resolved pairs (neither cell window-limited) the absolute finite-horizon SSA stays within a median 10% of its exact-score value for additive score error up to alpha=1.0 x s0 (median 5.2%, p90 13.2%), and degrades beyond that to a median 12.3% (p90 29.0%) at alpha=2.0 x s0, with the high baseline-SSA third worst affected (17.2%); over this range the value error and the ranking loss move together rather than the value degrading first (alpha=1.0: 5.2% value vs 5.1% ranking loss; alpha=2.0: 12.3% value vs 15.6% ranking loss), so on additive evidence alone we claim only that both the reported magnitude and the ordering remain usable up to roughly alpha=1 x s0 and should be treated as approximate beyond it.

_Scope note: a second perturbation family (exact score of the Gaussian-smoothed GMM via covariance inflation) was measured over the same 50 GMMs x 3 seeds and is retained in full in `p9_absolute_sensitivity_fc.json` / `_cells.csv` / `_summary.csv`. It is deliberately excluded from this table and figure._
