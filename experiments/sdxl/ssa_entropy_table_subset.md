# SDXL guidance sweep — SSA & kNN entropy (8-prompt subset)

dt = 0.005, T = 2000, N = 200 (SSA);  N = 2000 (entropy, PCA-50, k=5).
**Bold** = peak SSA per row.  **†** = SSA did **not** reach the ρ² ≤ 1/N
noise floor by T_max (Ŝ is a lower bound).

|                                                Prompt | SSA w=1 |     w=3 |   w=7.5 |    w=15 | kNN-H w=1 | w=3  | w=7.5 | w=15 |
|-------------------------------------------------------|--------:|--------:|--------:|--------:|----------:|-----:|------:|-----:|
| a photograph of a cat sitting on a windowsill         |    0.73 |   1.03† | **1.16†** |   0.81† |       −79 |  −96 |  −106 | −110 |
| a portrait of an elderly person                       |   1.08† |    1.01 | **1.44†** |   0.80† |       −71 |  −83 |  −105 | −114 |
| a superrealistic professional photograph of a person  |   1.01† |   1.30† | **1.90†** |   1.68† |       −69 |  −65 |   −74 |  −82 |
| a photo of a bat                                      |    0.94 |    1.48 | **1.75†** |    1.08 |       −74 |  −95 |  −107 | −112 |
| a photo of a crane                                    |    0.99 |   1.35† | **1.60†** |   1.37† |       −74 |  −84 |   −95 | −101 |
| a photo of a mouse                                    |    0.96 | **1.52** |    1.47 |    0.73 |       −77 | −106 |  −118 | −121 |
| a photo of a seal                                     |    1.03 | **1.64†** |   1.29† |    0.52 |       −78 |  −99 |  −106 | −107 |
| a photo of a jaguar                                   |    1.03 | **1.26** |   1.09† |    0.78 |       −88 | −113 |  −126 | −134 |

## LaTeX source (paste-ready)

```latex
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{4}{c}{SSA} & \multicolumn{4}{c}{$k$NN entropy} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Prompt & $w=1$ & 3 & 7.5 & 15 & $w=1$ & 3 & 7.5 & 15 \\
\midrule
a photograph of a cat sitting on a windowsill & $0.73$ & $1.03^{\dagger}$ & $\mathbf{1.16}^{\dagger}$ & $0.81^{\dagger}$ & $-79$ & $-96$ & $-106$ & $-110$ \\
a portrait of an elderly person & $1.08^{\dagger}$ & $1.01$ & $\mathbf{1.44}^{\dagger}$ & $0.80^{\dagger}$ & $-71$ & $-83$ & $-105$ & $-114$ \\
a superrealistic professional photograph of a person & $1.01^{\dagger}$ & $1.30^{\dagger}$ & $\mathbf{1.90}^{\dagger}$ & $1.68^{\dagger}$ & $-69$ & $-65$ & $-74$ & $-82$ \\
a photo of a bat & $0.94$ & $1.48$ & $\mathbf{1.75}^{\dagger}$ & $1.08$ & $-74$ & $-95$ & $-107$ & $-112$ \\
a photo of a crane & $0.99$ & $1.35^{\dagger}$ & $\mathbf{1.60}^{\dagger}$ & $1.37^{\dagger}$ & $-74$ & $-84$ & $-95$ & $-101$ \\
a photo of a mouse & $0.96$ & $\mathbf{1.52}$ & $1.47$ & $0.73$ & $-77$ & $-106$ & $-118$ & $-121$ \\
a photo of a seal & $1.03$ & $\mathbf{1.64}^{\dagger}$ & $1.29^{\dagger}$ & $0.52$ & $-78$ & $-99$ & $-106$ & $-107$ \\
a photo of a jaguar & $1.03$ & $\mathbf{1.26}$ & $1.09^{\dagger}$ & $0.78$ & $-88$ & $-113$ & $-126$ & $-134$ \\
\bottomrule
\end{tabular}
```
