"""
ACF estimation for REGULARLY-sampled series, via classic FFT correlation
(`scipy.signal.correlate`) instead of NUFFT.

This is a fast-path companion to `nufft_acf.py` / `realspace_acf.py`: when
the data happens to be on a uniform grid, there is no need to pay for NUFFT
(or for the O(n)-per-lag numba two-pointer scan) -- a plain FFT correlation
plus a cheap smoothing pass gives the *exact same* gaussian/rectangle
estimator, faster and with no numba/finufft dependency in the hot path.

Three estimators are provided, all sharing the same (lags, t, x) calling
convention as the rest of the package:

- `compute_acf_regular_fft`   : no smoothing kernel at all -- the windowed
                                 Pearson correlation Pastas uses for its
                                 "regular" bin_method (regular data only).
                                 Scales ~O(n) (it is NOT a quadratic method,
                                 unlike Pastas' gaussian/rectangle bin
                                 methods -- see benchmark/).
- `compute_acf_rectangle_fft` : same rectangular-kernel definition as
                                 `compute_acf_rectangle_nufft`/`_realspace`.
- `compute_acf_gaussian_fft`  : same gaussian-kernel definition as
                                 `compute_acf_gaussian_nufft`/`_realspace`.

All three require `t` to be regularly spaced (checked, raises otherwise);
use the `nufft` or `realspace` estimators for irregular sampling.

Implementation note on the `b` (pair-count) denominator
---------------------------------------------------------
For `gaussian`, `b` is obtained by applying the *exact same* smoothing
filter (`gaussian_filter1d`, same sigma, same boundary mode) to the
triangular "raw pair count" ramp `n - |lag|` as is applied to the raw
correlation numerator. This isn't just convenient: computing numerator and
denominator through the same discrete kernel makes any discretization
artifact of that kernel cancel exactly in the ratio, which is what lets
this estimator match `compute_acf_gaussian_realspace` to ~1e-6 without any
extra renormalization step. `mode="mirror"` is required (not scipy's
default `"reflect"`) for this cancellation to hold all the way to lag=0,
since the true pair-count ramp is symmetric *through* lag=0 (whole-sample
symmetry), not around the edge *between* lag=-1 and lag=0 (half-sample
symmetry, which is what `"reflect"` assumes).

For `rectangle`, the unsmoothed ramp `b = n - lag` is already exact as-is
-- no filtering needed -- because `uniform_filter1d` already normalizes by
its own window size internally.
"""

import numpy as np
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from .utils import standardize


def _check_regular_grid(t):
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return 1.0
    dt = np.diff(t)
    dt0 = dt[0]
    if not np.allclose(dt, dt0, rtol=1e-6):
        raise ValueError(
            "compute_acf_*_fft requires a regularly-sampled `t` "
            "(use the `nufft` or `realspace` estimators for irregular data)."
        )
    return dt0


def compute_acf_regular_fft(lags, t, x):
    """ACF estimate with no smoothing kernel, for regularly-sampled data --
    matches Pastas' `bin_method="regular"` (windowed Pearson correlation)
    to numerical precision, but vectorized instead of one `np.corrcoef`
    call per lag.

    Parameters
    ----------
    lags : array_like
        Lags at which to evaluate the ACF (same units as `t`).
    t : array_like
        Regularly-spaced sample times, sorted ascending.
    x : array_like
        Sample values, same length as `t`.

    Returns
    -------
    c, b : ndarray
        ACF estimate and pair count, both shape (len(lags),).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    lags = np.asarray(lags, dtype=float)
    dt = _check_regular_grid(t)
    n = len(x)

    lag_idx = np.round(lags / dt).astype(int)
    b = np.where(n - lag_idx <= 0, 1e-16, (n - lag_idx).astype(float))

    # Cumulative first/second moments -> exact windowed mean & std in O(1)
    # per lag (this is the *exact* expansion Var = E[x^2] - E[x]^2, not an
    # incremental re-centered sum -- the latter looks similar but silently
    # drifts by ~0.1-0.3% away from a true windowed std).
    s1 = np.concatenate(([0.0], np.cumsum(x)))
    s2 = np.concatenate(([0.0], np.cumsum(x * x)))

    def _windowed_mean_std(lo, hi):
        cnt = np.maximum((hi - lo).astype(float), 1.0)
        mean = (s1[hi] - s1[lo]) / cnt
        var = np.maximum((s2[hi] - s2[lo]) / cnt - mean**2, 0.0)
        return mean, np.sqrt(var)

    n_lags = len(lag_idx)
    hi_y = np.clip(n - lag_idx, 0, n)
    lo_x = np.clip(lag_idx, 0, n)
    y_mean, y_std = _windowed_mean_std(np.zeros(n_lags, dtype=int), hi_y)
    x_mean, x_std = _windowed_mean_std(lo_x, np.full(n_lags, n))

    c_raw = correlate(x, x, mode="full")[n - 1 : 2 * n - 1]
    valid = (lag_idx >= 0) & (lag_idx < n)
    c_raw_at = np.where(valid, c_raw[np.clip(lag_idx, 0, n - 1)], np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        cov = (
            c_raw_at / np.where(n - lag_idx > 0, n - lag_idx, np.nan) - y_mean * x_mean
        )
        denom = y_std * x_std
        c = np.where(denom > 1e-12, cov / denom, np.nan)
    c = np.where(valid, c, np.nan)
    return c, b


def compute_acf_rectangle_fft(lags, t, x, bin_width=0.5):
    """ACF estimate via FFT correlation + rectangular smoothing, for
    regularly-sampled data. Same kernel definition (and -- on the same
    data -- numerically equivalent result) as `compute_acf_rectangle_nufft`
    / `compute_acf_rectangle_realspace`, just computed via plain FFT
    correlation instead of NUFFT / a numba two-pointer scan.

    Parameters
    ----------
    lags : array_like
    t : array_like
        Regularly-spaced sample times, sorted ascending.
    x : array_like
    bin_width : float
        Rectangular half-width (same units as `t`).

    Returns
    -------
    c, b : ndarray
    """
    t = np.asarray(t, dtype=float)
    x = standardize(np.asarray(x, dtype=float))
    lags = np.asarray(lags, dtype=float)
    dt = _check_regular_grid(t)
    n = len(x)

    lag_idx = np.round(lags / dt).astype(int)
    # Forcing an ODD kernel size (2*k+1) is deliberate.
    # scipy's uniform_filter1d centers an EVEN-sized window a half
    # -sample off from the true symmetric [-bin_width, +bin_width] interval
    # kernels.py's two-pointer scan evaluates exactly, which otherwise
    # introduces a small but systematic (~0.3-0.5%, not just at lag=0)
    # bias at every lag -- confirmed empirically, see tests/test_fft_acf.py.
    kernel_size = 2 * int(round(bin_width / dt)) + 1

    c_raw = correlate(x, x, mode="full")[n - 1 : 2 * n - 1]
    c_smoothed = uniform_filter1d(c_raw, size=kernel_size, mode="nearest")

    valid = (lag_idx >= 0) & (lag_idx < n)
    c_at = np.where(valid, c_smoothed[np.clip(lag_idx, 0, n - 1)], np.nan)
    b = np.where(valid, np.maximum(n - lag_idx, 1e-16), 1e-16)

    with np.errstate(divide="ignore", invalid="ignore"):
        c = c_at / b
    return c, b


def compute_acf_gaussian_fft(lags, t, x, bin_width=0.5):
    """ACF estimate via FFT correlation + gaussian smoothing, for
    regularly-sampled data. Same kernel definition as
    `compute_acf_gaussian_nufft` / `compute_acf_gaussian_realspace`.

    Parameters
    ----------
    lags : array_like
    t : array_like
        Regularly-spaced sample times, sorted ascending.
    x : array_like
    bin_width : float
        Gaussian standard deviation (same units as `t`).

    Returns
    -------
    c, b : ndarray
    """
    t = np.asarray(t, dtype=float)
    x = standardize(np.asarray(x, dtype=float))
    lags = np.asarray(lags, dtype=float)
    dt = _check_regular_grid(t)
    n = len(x)
    sigma = bin_width / dt

    lag_idx = np.round(lags / dt).astype(int)
    max_idx = int(np.max(np.abs(lag_idx))) if len(lag_idx) else 0
    n_eval = max(n, max_idx + 1)

    c_raw = correlate(x, x, mode="full")[n - 1 : 2 * n - 1]
    c_raw = np.pad(c_raw, (0, max(0, n_eval - n)), constant_values=0.0)
    c_smoothed = gaussian_filter1d(c_raw, sigma=sigma, mode="mirror")

    b_raw = np.maximum(n - np.arange(n_eval, dtype=float), 0.0)
    b_smoothed = gaussian_filter1d(b_raw, sigma=sigma, mode="mirror")

    valid = (lag_idx >= 0) & (lag_idx < n)
    idx = np.clip(lag_idx, 0, n_eval - 1)
    c_at = np.where(valid, c_smoothed[idx], np.nan)
    b_at = np.where(valid, b_smoothed[idx], 1e-16)

    with np.errstate(divide="ignore", invalid="ignore"):
        c = c_at / b_at
    return c, b_at
