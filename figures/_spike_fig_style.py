"""Shared color/style constants for unified spike-distribution figures."""

# --- Colors ---
HIST_EMPIRICAL = '#AED6F1'   # light blue, empirical histogram
HIST_MC_NULL   = '#F5B7B1'   # pink, MC null histogram
FREECONV_CURVE = '#C0392B'   # red, free-conv density curve
MP_CURVE       = '#2980B9'   # blue, naive MP density curve
FLOOR_TAU      = '#1E8449'   # dark green, lambda_+(tau) floor
FLOOR_INF      = '#E67E22'   # orange, lambda_+(inf) floor
MC_NULL_PCTL   = '#8E44AD'   # purple, MC null 95th percentile

# --- Style defaults ---
HIST_ALPHA_EMP  = 0.6
HIST_ALPHA_NULL = 0.5
CURVE_LW        = 1.8
VLINE_LW        = 1.4
VLINE_STYLE     = ':'        # dotted for floor lines
