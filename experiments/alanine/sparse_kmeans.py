"""Witten and Tibshirani (2010) sparse K-means.

Reference: D.M. Witten, R. Tibshirani, "A framework for feature selection in
clustering," JASA 105(490): 713-726, 2010. Algorithm 3.1 (Sparse K-means).

Objective:
    maximize_{C, w}  sum_j w_j * BCSS_j(C)
    subject to ||w||_2 <= 1, ||w||_1 <= s, w_j >= 0

where BCSS_j(C) = TSS_j - WCSS_j(C). Block-coordinate alternation:
    1) fix w, run K-means on sqrt(w) * X to update C
    2) fix C, soft-threshold update for w
"""
import numpy as np
from sklearn.cluster import KMeans


def _bcss_per_feature(X, labels, K):
    n = X.shape[0]
    mean_all = X.mean(axis=0)
    tss = ((X - mean_all) ** 2).sum(axis=0)
    wcss = np.zeros(X.shape[1])
    for k in range(K):
        mask = labels == k
        nk = mask.sum()
        if nk <= 1:
            continue
        mu_k = X[mask].mean(axis=0)
        wcss += ((X[mask] - mu_k) ** 2).sum(axis=0)
    return tss - wcss


def _soft_threshold(a, delta):
    return np.sign(a) * np.maximum(np.abs(a) - delta, 0.0)


def _update_w(a, s):
    a_pos = np.maximum(a, 0.0)
    norm = np.linalg.norm(a_pos)
    if norm < 1e-12:
        w = np.zeros_like(a_pos)
        if a_pos.size > 0:
            w[np.argmax(a_pos)] = 1.0
        return w
    w = a_pos / norm
    if w.sum() <= s:
        return w
    lo, hi = 0.0, np.abs(a_pos).max()
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        st = _soft_threshold(a_pos, mid)
        nrm = np.linalg.norm(st)
        if nrm < 1e-12:
            hi = mid
            continue
        l1 = (st / nrm).sum()
        if l1 > s:
            lo = mid
        else:
            hi = mid
    st = _soft_threshold(a_pos, hi)
    nrm = np.linalg.norm(st)
    if nrm < 1e-12:
        w = np.zeros_like(a_pos)
        if a_pos.size > 0:
            w[np.argmax(a_pos)] = 1.0
        return w
    return st / nrm


def sparse_kmeans(X, K, s, max_iter=20, tol=1e-4, n_init=3, seed=0):
    """Fit sparse K-means; return (w, labels, objective_history)."""
    n, p = X.shape
    w = np.full(p, 1.0 / np.sqrt(p))
    rng = np.random.RandomState(seed)
    obj_hist = []
    labels = None
    for it in range(max_iter):
        Xw = X * np.sqrt(w)[None, :]
        km = KMeans(n_clusters=K, n_init=n_init, max_iter=300,
                    random_state=rng.randint(2**31 - 1))
        labels = km.fit_predict(Xw)
        a = _bcss_per_feature(X, labels, K)
        w_new = _update_w(a, s)
        obj = float((w_new * a).sum())
        obj_hist.append(obj)
        diff = np.abs(w_new - w).sum() / max(np.abs(w).sum(), 1e-12)
        w = w_new
        if diff < tol:
            break
    return w, labels, obj_hist


def gap_statistic_select_s(X, K, s_grid, B=10, seed=0, max_iter=15, n_init=3,
                           verbose=False, rule='1se'):
    """Witten and Tibshirani (2010) Section 3.2 gap statistic for s.

    rule:
      'max' = argmax gap
      '1se' = smallest s with gap(s) >= max_gap - sd(s_at_max)
              (the standard "1 standard error" rule used to break ties when
              gap plateaus near the maximum, as recommended in the original
              gap-statistic literature for K selection and adapted here.)

    Returns dict with s_star, gap profile, sds, raw objectives.
    """
    n, p = X.shape
    rng = np.random.RandomState(seed)
    obj_obs = np.zeros(len(s_grid))
    obj_ref = np.zeros((len(s_grid), B))

    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    for i, s in enumerate(s_grid):
        w, labels, hist = sparse_kmeans(X, K, s, max_iter=max_iter,
                                        n_init=n_init, seed=int(rng.randint(2**31 - 1)))
        a = _bcss_per_feature(X, labels, K)
        obj_obs[i] = float((w * a).sum())
        if verbose:
            print(f'  s={s:.3f}: obj={obj_obs[i]:.3e}, '
                  f'nnz(w)={(w > 1e-8).sum()}')
        for b in range(B):
            X_ref = rng.uniform(low=mins, high=maxs, size=X.shape).astype(X.dtype)
            w_b, labels_b, _ = sparse_kmeans(
                X_ref, K, s, max_iter=max_iter, n_init=n_init,
                seed=int(rng.randint(2**31 - 1)))
            a_b = _bcss_per_feature(X_ref, labels_b, K)
            obj_ref[i, b] = float((w_b * a_b).sum())

    log_obs = np.log(np.clip(obj_obs, 1e-30, None))
    log_ref = np.log(np.clip(obj_ref, 1e-30, None))
    gap = log_obs - log_ref.mean(axis=1)
    sds = log_ref.std(axis=1, ddof=1) * np.sqrt(1.0 + 1.0 / B)

    i_max = int(np.argmax(gap))
    if rule == 'max':
        i_star = i_max
    elif rule == '1se':
        threshold = gap[i_max] - sds[i_max]
        candidates = np.where(gap >= threshold)[0]
        i_star = int(candidates.min()) if len(candidates) else i_max
    else:
        raise ValueError(f'unknown rule {rule}')
    return {
        's_star': float(s_grid[i_star]),
        'i_star': i_star,
        'i_max': i_max,
        'gap': gap,
        'sds': sds,
        'obj_obs': obj_obs,
        'obj_ref_mean': obj_ref.mean(axis=1),
    }
