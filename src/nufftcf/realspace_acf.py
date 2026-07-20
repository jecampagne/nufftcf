"""
ACF estimation directly in real (time-difference) space -- no FFT, no
implicit periodicity assumption. This is the same family of method Pastas
uses internally, but with the O(n) per-lag two-pointer optimization (see
`kernels.py`) instead of the naive O(n^2) scan.

Use this as an artifact-free reference, or as the primary estimator if you'd
rather avoid the NUFFT method's small (~1-3%) residual amplitude bias on
strongly periodic signals (see `nufft_acf.py` docstring). Scaling is O(n) per
lag here (vs O(n log n) for NUFFT), so for very long series the NUFFT
variant will eventually be faster, but for most practical sizes both are
fast; benchmark on your own data if it matters.
"""

import numpy as np

from .kernels import (
    compute_b_gaussian,
    compute_b_rectangle,
    compute_c_gaussian,
    compute_c_rectangle,
)
from .utils import standardize


def compute_acf_gaussian_realspace(lags, t, x, bin_width=0.5):
    """Exact gaussian-kernel ACF estimate, computed directly in real space.

    Same signature and return values (c, b) as `compute_acf_gaussian_nufft`.
    """
    t = np.asarray(t, dtype=float)
    x = standardize(np.asarray(x, dtype=float))
    lags = np.asarray(lags, dtype=float)
    c_raw = compute_c_gaussian(t, lags, x, bin_width)
    b = compute_b_gaussian(t, lags, bin_width)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = c_raw / b
    return c, b


def compute_acf_rectangle_realspace(lags, t, x, bin_width=0.5):
    """Exact rectangular-kernel ACF estimate, computed directly in real space.

    Same signature and return values (c, b) as `compute_acf_rectangle_nufft`.
    """
    t = np.asarray(t, dtype=float)
    x = standardize(np.asarray(x, dtype=float))
    lags = np.asarray(lags, dtype=float)
    c_raw = compute_c_rectangle(t, lags, x, bin_width)
    b = compute_b_rectangle(t, lags, bin_width)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = c_raw / b
    return c, b
