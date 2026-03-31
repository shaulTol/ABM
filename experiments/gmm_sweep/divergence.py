"""Divergence detection between SSA and MI trends."""

import numpy as np
from scipy.signal import savgol_filter


def detect_divergence(param_values, ssa_values, mi_values, smooth_window=5, threshold=0.1):
    """Detect regions where SSA and MI trends disagree.

    Algorithm:
    1. Smooth both curves (Savitzky-Golay, window=smooth_window, order=2)
       - If len < smooth_window, skip smoothing
    2. Compute local slopes via finite differences
    3. Normalize slopes to [-1, 1] (divide by max abs, handle zero)
    4. Find indices where sign(d_ssa) != sign(d_mi) AND both |d_ssa|, |d_mi| > threshold
    5. Group consecutive indices into regions

    Returns:
        list of dicts with keys: 'start_idx', 'end_idx', 'type'
        type is "SSA_up_MI_down" or "SSA_down_MI_up"
    """
    ssa = np.array(ssa_values, dtype=float)
    mi = np.array(mi_values, dtype=float)
    n = len(ssa)

    if n < 3:
        return []

    # Smooth
    if n >= smooth_window and smooth_window >= 3:
        ssa_smooth = savgol_filter(ssa, smooth_window, min(2, smooth_window - 1))
        mi_smooth = savgol_filter(mi, smooth_window, min(2, smooth_window - 1))
    else:
        ssa_smooth = ssa.copy()
        mi_smooth = mi.copy()

    # Finite difference slopes
    d_ssa = np.diff(ssa_smooth)
    d_mi = np.diff(mi_smooth)

    # Normalize to [-1, 1]
    max_ssa = np.max(np.abs(d_ssa))
    max_mi = np.max(np.abs(d_mi))
    if max_ssa > 0:
        d_ssa = d_ssa / max_ssa
    if max_mi > 0:
        d_mi = d_mi / max_mi

    # Find divergence indices
    diverging = (np.sign(d_ssa) != np.sign(d_mi)) & (np.abs(d_ssa) > threshold) & (np.abs(d_mi) > threshold)
    indices = np.where(diverging)[0]

    if len(indices) == 0:
        return []

    # Group consecutive indices into regions
    regions = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            # Determine type
            avg_d_ssa = np.mean(d_ssa[start:prev + 1])
            rtype = "SSA_up_MI_down" if avg_d_ssa > 0 else "SSA_down_MI_up"
            regions.append({'start_idx': int(start), 'end_idx': int(prev), 'type': rtype})
            start = idx
            prev = idx

    # Last region
    avg_d_ssa = np.mean(d_ssa[start:prev + 1])
    rtype = "SSA_up_MI_down" if avg_d_ssa > 0 else "SSA_down_MI_up"
    regions.append({'start_idx': int(start), 'end_idx': int(prev), 'type': rtype})

    return regions
