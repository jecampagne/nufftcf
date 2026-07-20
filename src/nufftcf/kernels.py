"""
Numba-jitted, two-pointer-optimized kernels.

All functions here require `t` (time, in days) sorted in ascending order.
Complexity is O(n) per lag (instead of the naive O(n^2)), thanks to the
two-pointer technique: as the lag-shifted window center increases
monotonically with j (since t is sorted), the window bounds [lo, hi) only
ever advance, never retreat.

`b_*` functions compute the kernel-weighted (or counted) number of
contributing pairs per lag -- the normalization denominator.

`c_*` functions compute the kernel-weighted sum of x_i * x_j per lag -- the
correlation numerator, for the "real-space" (exact, non-NUFFT) ACF estimator.
"""

import math
import numpy as np
from numba import njit, prange


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def compute_b_gaussian(t, lags, sigma):
    """Gaussian-kernel weighted pair count per lag."""
    n = len(t)
    nlags = len(lags)
    b = np.zeros(nlags)
    den1 = -2 * sigma**2
    den2 = math.sqrt(2 * math.pi) * sigma
    six_den2 = 6 * den2
    for k in prange(nlags):
        lag = lags[k]
        b_sum = 0.0
        lo = 0
        hi = 0
        for j in range(n):
            center = t[j] + lag
            while lo < n and t[lo] < center - six_den2:
                lo += 1
            while hi < n and t[hi] < center + six_den2:
                hi += 1
            for i in range(lo, hi):
                dtlag = t[i] - center
                b_sum += math.exp(dtlag**2 / den1) / den2
        b[k] = b_sum
        if b[k] <= 0:
            b[k] = 1e-16
    return b


@njit(parallel=True, nogil=True, cache=True)
def compute_b_rectangle(t, lags, bin_width):
    """Rectangular-kernel pair count per lag (direct count, no inner loop)."""
    n = len(t)
    nlags = len(lags)
    b = np.zeros(nlags)
    for k in prange(nlags):
        lag = lags[k]
        b_sum = 0.0
        lo = 0
        hi = 0
        for j in range(n):
            center = t[j] + lag
            while lo < n and t[lo] < center - bin_width:
                lo += 1
            if hi < lo:
                hi = lo
            while hi < n and t[hi] <= center + bin_width:
                hi += 1
            b_sum += hi - lo
        b[k] = b_sum
        if b[k] <= 0:
            b[k] = 1e-16
    return b


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def compute_c_gaussian(t, lags, x, sigma):
    """Gaussian-kernel weighted sum of x_i*x_j per lag (correlation numerator)."""
    n = len(t)
    nlags = len(lags)
    c = np.zeros(nlags)
    den1 = -2 * sigma**2
    den2 = math.sqrt(2 * math.pi) * sigma
    six_den2 = 6 * den2
    for k in prange(nlags):
        lag = lags[k]
        c_sum = 0.0
        lo = 0
        hi = 0
        for j in range(n):
            center = t[j] + lag
            while lo < n and t[lo] < center - six_den2:
                lo += 1
            while hi < n and t[hi] < center + six_den2:
                hi += 1
            xj = x[j]
            for i in range(lo, hi):
                dtlag = t[i] - center
                w = math.exp(dtlag**2 / den1) / den2
                c_sum += w * x[i] * xj
        c[k] = c_sum
    return c


@njit(parallel=True, nogil=True, cache=True)
def compute_c_rectangle(t, lags, x, bin_width):
    """Rectangular-kernel sum of x_i*x_j per lag, via prefix sums (O(1) per j,
    since the rectangle kernel weight is uniform inside the window: the
    window sum of x is looked up directly from a precomputed cumulative sum,
    rather than re-summed element by element)."""
    n = len(t)
    nlags = len(lags)
    c = np.zeros(nlags)
    cumsum_x = np.zeros(n + 1)
    for idx in range(n):
        cumsum_x[idx + 1] = cumsum_x[idx] + x[idx]
    for k in prange(nlags):
        lag = lags[k]
        c_sum = 0.0
        lo = 0
        hi = 0
        for j in range(n):
            center = t[j] + lag
            while lo < n and t[lo] < center - bin_width:
                lo += 1
            if hi < lo:
                hi = lo
            while hi < n and t[hi] <= center + bin_width:
                hi += 1
            window_sum = cumsum_x[hi] - cumsum_x[lo]
            c_sum += x[j] * window_sum
        c[k] = c_sum
    return c


# ── Cross-correlation kernel functions ────────────────────────────────────────
# Same two-pointer logic as the ACF kernels above, adapted for two DIFFERENT
# time arrays t (for signal x) and s (for signal y).  The CCF pair condition
# is  s_j - t_i ≈ lag  (whereas for ACF it was  t_i - t_j ≈ lag).
# As j increases (s[j] increases), center = s[j] - lag increases
# monotonically, so lo/hi over t still only advance → O(n_t + n_s) per lag.


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def compute_b_gaussian_cross(t, s, lags, sigma):
    """Gaussian-kernel weighted pair count for CCF.

    Counts pairs (t_i, s_j) with s_j − t_i ≈ lag, weighted by the
    Gaussian kernel.  `t` and `s` must both be sorted ascending.
    """
    n_t = len(t)
    n_s = len(s)
    nlags = len(lags)
    b = np.zeros(nlags)
    den1 = -2 * sigma**2
    den2 = math.sqrt(2 * math.pi) * sigma
    six_den2 = 6 * den2
    for k in prange(nlags):
        lag = lags[k]
        b_sum = 0.0
        lo = 0
        hi = 0
        for j in range(n_s):
            center = s[j] - lag  # t_i near here → s_j − t_i ≈ lag
            while lo < n_t and t[lo] < center - six_den2:
                lo += 1
            while hi < n_t and t[hi] < center + six_den2:
                hi += 1
            for i in range(lo, hi):
                dtlag = t[i] - center
                b_sum += math.exp(dtlag**2 / den1) / den2
        b[k] = b_sum if b_sum > 0 else 1e-16
    return b


@njit(parallel=True, nogil=True, cache=True)
def compute_b_rectangle_cross(t, s, lags, bin_width):
    """Rectangle-kernel pair count for CCF.

    Counts pairs (t_i, s_j) with |s_j − t_i − lag| ≤ bin_width.
    """
    n_t = len(t)
    n_s = len(s)
    nlags = len(lags)
    b = np.zeros(nlags)
    for k in prange(nlags):
        lag = lags[k]
        b_sum = 0.0
        lo = 0
        hi = 0
        for j in range(n_s):
            center = s[j] - lag
            while lo < n_t and t[lo] < center - bin_width:
                lo += 1
            if hi < lo:
                hi = lo
            while hi < n_t and t[hi] <= center + bin_width:
                hi += 1
            b_sum += hi - lo
        b[k] = b_sum if b_sum > 0 else 1e-16
    return b


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def compute_c_gaussian_cross(t, x, s, y, lags, sigma):
    """Gaussian-kernel weighted sum of x_i * y_j for CCF.

    For each lag, accumulates x(t_i) * y(s_j) for pairs where
    s_j − t_i ≈ lag, weighted by the Gaussian kernel w(s_j − t_i − lag).
    """
    n_t = len(t)
    n_s = len(s)
    nlags = len(lags)
    c = np.zeros(nlags)
    den1 = -2 * sigma**2
    den2 = math.sqrt(2 * math.pi) * sigma
    six_den2 = 6 * den2
    for k in prange(nlags):
        lag = lags[k]
        c_sum = 0.0
        lo = 0
        hi = 0
        for j in range(n_s):
            center = s[j] - lag
            while lo < n_t and t[lo] < center - six_den2:
                lo += 1
            while hi < n_t and t[hi] < center + six_den2:
                hi += 1
            yj = y[j]
            for i in range(lo, hi):
                dtlag = t[i] - center
                w = math.exp(dtlag**2 / den1) / den2
                c_sum += w * x[i] * yj
        c[k] = c_sum
    return c


@njit(parallel=True, nogil=True, cache=True)
def compute_c_rectangle_cross(t, x, s, y, lags, bin_width):
    """Rectangle-kernel sum of x_i * y_j for CCF, via prefix sums."""
    n_t = len(t)
    n_s = len(s)
    nlags = len(lags)
    c = np.zeros(nlags)
    cumsum_x = np.zeros(n_t + 1)
    for idx in range(n_t):
        cumsum_x[idx + 1] = cumsum_x[idx] + x[idx]
    for k in prange(nlags):
        lag = lags[k]
        c_sum = 0.0
        lo = 0
        hi = 0
        for j in range(n_s):
            center = s[j] - lag
            while lo < n_t and t[lo] < center - bin_width:
                lo += 1
            if hi < lo:
                hi = lo
            while hi < n_t and t[hi] <= center + bin_width:
                hi += 1
            window_sum = cumsum_x[hi] - cumsum_x[lo]
            c_sum += y[j] * window_sum
        c[k] = c_sum
    return c
