"""
CCF estimation for two REGULARLY-sampled series, via classic FFT
cross-correlation (`scipy.signal.correlate`) instead of NUFFT.

This is the cross-correlation counterpart of `fft_acf.py`: when both input
series happen to live on a uniform time grid with the *same* sampling step
`dt` (a common, but not necessary, situation e.g. when both signals come
from the same regularly-resampled source), there is no need to pay for
NUFFT -- a plain FFT cross-correlation plus a cheap smoothing pass gives
the *exact same* gaussian/rectangle CCF estimator as `nufft_ccf.py` /
`realspace_ccf.py`, faster and with no finufft dependency in the hot path.

Two estimators are provided, mirroring `nufft_ccf.py` / `realspace_ccf.py`:

- `compute_ccf_rectangle_fft` : same rectangular-kernel definition as
                                  `compute_ccf_rectangle_nufft`/`_realspace`.
- `compute_ccf_gaussian_fft`  : same gaussian-kernel definition as
                                  `compute_ccf_gaussian_nufft`/`_realspace`.

Both require `t` and `s` to each be regularly spaced with the SAME step
`dt`, and to lie on a common sampling lattice (`s[0] - t[0]` an integer
multiple of `dt`) -- checked, raises otherwise. Use the `nufft` or
`realspace` estimators when the two series don't share a lattice (e.g.
different sampling steps, or an irregular grid).

Sign convention
----------------
Same as `nufft_ccf.py`/`realspace_ccf.py`: the CCF pair condition is
``s_j - t_i ~= lag``, i.e. a positive lag means signal 2 (`s`, `y`) lags
BEHIND signal 1 (`t`, `x`). Concretely, if `y` is a copy of `x` whose
sample times are all shifted forward by `tau` (`s = t + tau`), the CCF
peaks at `lag = +tau`.

Implementation note: mapping a lag to an index of `correlate(x, y, "full")`
----------------------------------------------------------------------------
Let `n_t = len(x)`, `n_s = len(y)`, and `k0 = round((s[0]-t[0]) / dt)` be
the integer sample offset between the two grids. For an integer lag index
`L` (in units of `dt`), the pair condition `s_j - t_i = L*dt` translates,
in terms of the two series' own 0-based sample indices, to `i = k0 + j - L`.
Summing `x[i] * y[j]` over all valid `(i, j)` is then exactly
``scipy.signal.correlate(x, y, mode="full")[k0 - L + n_s - 1]`` (standard
`numpy`/`scipy` "full" cross-correlation indexing, where index `n_s - 1`
of the output corresponds to zero shift between the two arrays). This is
the same "raw sum via a single FFT correlation, then slice per lag" trick
`fft_acf.py` uses for the ACF, just for two different-length series with a
constant integer offset instead of one series against itself.

Unlike the ACF case, `correlate(x, y, "full")` already spans the *entire*
range of lags for which at least one pair exists -- there is no discarded
symmetric half to reconstruct (cross-correlation isn't symmetric in
general). So, unlike `compute_acf_gaussian_fft`'s use of `mode="mirror"`
(needed there specifically to reconstruct the discarded negative-lag half
from the true whole-sample symmetry of the ACF), the gaussian smoothing
here uses `mode="constant", cval=0.0`: past the two true edges of
`correlate`'s output there really are zero contributing pairs, so padding
with zero is the physically correct continuation, not an approximation.

Implementation note on the `b` (pair-count) denominator
---------------------------------------------------------
Same division of labour as `fft_acf.py`: for `rectangle`, the exact
(unsmoothed) overlap-count formula is used directly as `b` -- no filtering
needed, since `uniform_filter1d` already normalizes by its own window size
internally. For `gaussian`, `b` is obtained by running the *same*
`gaussian_filter1d` (same sigma, same `mode="constant"`) over the raw
overlap-count ramp as is applied to the raw cross-correlation numerator,
so that ratio cancellation of discretization artifacts holds just like in
`compute_acf_gaussian_fft`.

Implementation note on normalisation
--------------------------------------
As in `nufft_ccf.py` / `realspace_ccf.py`, both `x` and `y` are
standardised internally, and the raw smoothed-cross-correlation-over-`b`
ratio is further divided by ``sqrt(ACF_x(0) * ACF_y(0))``, where
`ACF_x(0)` and `ACF_y(0)` are each signal's OWN same-kernel, same-family
FFT ACF estimate at lag 0 (reusing `compute_acf_rectangle_fft` /
`compute_acf_gaussian_fft` from `fft_acf.py` directly -- their smoothed
lag-0 value already *is* the right scale, since it folds in the same
kernel-edge/boundary effects that make a standardised signal's own
smoothed ACF(0) not quite exactly 1). This mirrors, function-for-function,
what `_acf_scale_at_lag0` does for the NUFFT family, just without needing
that function's small-symmetric-lags-array trick: the FFT ACF's own use of
`mode="mirror"` already handles lag=0 correctly as an edge sample of a
whole-sample-symmetric ramp (see `fft_acf.py` docstring), so no special
casing is required here -- we simply call the existing, already-tested
`fft_acf.py` functions at `lags=[0.0]`.
"""

import numpy as np
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from .utils import standardize
from .fft_acf import (
    _check_regular_grid,
    compute_acf_rectangle_fft,
    compute_acf_gaussian_fft,
)


def _check_common_regular_grid(t, s):
    """Check that `t` and `s` are each regularly spaced with the SAME step
    `dt`, and that they lie on a common integer sampling lattice (i.e.
    `s[0] - t[0]` is an integer multiple of `dt`).

    This is what lets the CCF be computed via a single FFT cross-correlation
    plus an integer-index lookup per lag, instead of NUFFT/direct summation.

    Returns
    -------
    dt : float
        Common sampling step.
    k0 : int
        Integer sample offset of `s` relative to `t`
        (``s[0] == t[0] + k0 * dt``).
    """
    dt_t = _check_regular_grid(t)
    dt_s = _check_regular_grid(s)
    if not np.isclose(dt_t, dt_s, rtol=1e-6):
        raise ValueError(
            "compute_ccf_*_fft requires `t` and `s` to share the same "
            f"sampling step (got dt_t={dt_t}, dt_s={dt_s}); use the `nufft` "
            "or `realspace` cross-correlation estimators for series with "
            "different sampling steps."
        )
    offset = (s[0] - t[0]) / dt_t
    k0 = int(round(offset))
    if not np.isclose(offset, k0, atol=1e-6):
        raise ValueError(
            "compute_ccf_*_fft requires `t` and `s` to lie on a common "
            "sampling lattice (`s[0] - t[0]` must be an integer multiple of "
            "`dt`); use the `nufft` or `realspace` cross-correlation "
            "estimators otherwise."
        )
    return dt_t, k0


def compute_ccf_rectangle_fft(lags, t, x, s, y, bin_width=0.5):
    """CCF estimate via FFT cross-correlation + rectangular smoothing, for
    two regularly-sampled series sharing the same sampling step. Same
    kernel definition (and -- on the same data -- numerically equivalent
    result) as `compute_ccf_rectangle_nufft` / `compute_ccf_rectangle_realspace`,
    just computed via a plain FFT cross-correlation instead of NUFFT / a
    numba two-pointer scan.

    Parameters
    ----------
    lags : array_like
        Lags at which to evaluate the CCF (same units as `t`/`s`). A
        positive lag means `y` lags behind `x` (see module docstring).
    t, x : array_like
        Regularly-spaced sample times (sorted ascending) and values of
        signal 1.
    s, y : array_like
        Regularly-spaced sample times (sorted ascending) and values of
        signal 2. Must share the same sampling step as `t`, and lie on the
        same integer lattice (`s[0] - t[0]` a multiple of `dt`).
    bin_width : float
        Rectangular half-width (same units as `t`).

    Returns
    -------
    c, b : ndarray
        CCF estimate (Pearson convention, in [-1, 1]) and effective pair
        count, both shape (len(lags),).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    lags = np.asarray(lags, dtype=float)
    dt, k0 = _check_common_regular_grid(t, s)
    n_t = len(x)
    n_s = len(y)

    x_std = standardize(x)
    y_std = standardize(y)

    corr_full = correlate(x_std, y_std, mode="full")  # length n_t + n_s - 1
    # Same ODD-kernel-size rationale as compute_acf_rectangle_fft: forces
    # the box filter to be centered exactly on each sample, matching the
    # symmetric [-bin_width, +bin_width] continuous kernel.
    kernel_size = 2 * int(round(bin_width / dt)) + 1
    c_smoothed = uniform_filter1d(corr_full, size=kernel_size, mode="nearest")

    lag_idx = np.round(lags / dt).astype(int)
    idx = k0 - lag_idx + n_s - 1
    valid = (idx >= 0) & (idx < n_t + n_s - 1)
    idx_c = np.clip(idx, 0, n_t + n_s - 2)

    c_at = np.where(valid, c_smoothed[idx_c], np.nan)

    # Exact (unsmoothed) overlap-count formula: number of sample pairs
    # (t_i, s_j) with i = k0 + j - lag_idx, i in [0, n_t), j in [0, n_s).
    i_lo = np.maximum(0, k0 - lag_idx)
    i_hi = np.minimum(n_t, n_s + k0 - lag_idx)
    b = np.where(valid, np.maximum((i_hi - i_lo).astype(float), 1e-16), 1e-16)

    with np.errstate(divide="ignore", invalid="ignore"):
        c_norm = c_at / b

    scale_x = compute_acf_rectangle_fft(np.array([0.0]), t, x, bin_width)[0][0]
    scale_y = compute_acf_rectangle_fft(np.array([0.0]), s, y, bin_width)[0][0]
    scale = np.sqrt(scale_x * scale_y)

    c = c_norm / scale
    return c, b


def compute_ccf_gaussian_fft(lags, t, x, s, y, bin_width=0.5):
    """CCF estimate via FFT cross-correlation + gaussian smoothing, for two
    regularly-sampled series sharing the same sampling step. Same kernel
    definition as `compute_ccf_gaussian_nufft` / `compute_ccf_gaussian_realspace`.

    Parameters
    ----------
    lags : array_like
        Lags at which to evaluate the CCF (same units as `t`/`s`). A
        positive lag means `y` lags behind `x` (see module docstring).
    t, x : array_like
        Regularly-spaced sample times (sorted ascending) and values of
        signal 1.
    s, y : array_like
        Regularly-spaced sample times (sorted ascending) and values of
        signal 2. Must share the same sampling step as `t`, and lie on the
        same integer lattice (`s[0] - t[0]` a multiple of `dt`).
    bin_width : float
        Gaussian kernel standard deviation (same units as `t`).

    Returns
    -------
    c, b : ndarray
        CCF estimate (Pearson convention, in [-1, 1]) and effective
        Gaussian-weighted pair count, both shape (len(lags),).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    lags = np.asarray(lags, dtype=float)
    dt, k0 = _check_common_regular_grid(t, s)
    n_t = len(x)
    n_s = len(y)
    sigma = bin_width / dt

    x_std = standardize(x)
    y_std = standardize(y)

    n_full = n_t + n_s - 1
    corr_full = correlate(x_std, y_std, mode="full")  # length n_full
    # mode="constant", cval=0.0: past the true edges of `correlate`'s
    # output there are zero contributing pairs -- see module docstring.
    c_smoothed = gaussian_filter1d(corr_full, sigma=sigma, mode="constant", cval=0.0)

    # Raw overlap-count ramp b(idx), expressed over the SAME idx axis as
    # corr_full (idx = k0 - lag_idx + n_s - 1  <=>  lag_idx = k0+n_s-1-idx).
    idx_all = np.arange(n_full)
    lag_of_idx = k0 + n_s - 1 - idx_all
    i_lo = np.maximum(0, k0 - lag_of_idx)
    i_hi = np.minimum(n_t, n_s + k0 - lag_of_idx)
    b_raw = np.maximum((i_hi - i_lo).astype(float), 0.0)
    b_smoothed = gaussian_filter1d(b_raw, sigma=sigma, mode="constant", cval=0.0)

    lag_idx = np.round(lags / dt).astype(int)
    idx = k0 - lag_idx + n_s - 1
    valid = (idx >= 0) & (idx < n_full)
    idx_c = np.clip(idx, 0, n_full - 1)

    c_at = np.where(valid, c_smoothed[idx_c], np.nan)
    b_at = np.where(valid, b_smoothed[idx_c], 1e-16)
    b_at = np.where(b_at > 0, b_at, 1e-16)

    with np.errstate(divide="ignore", invalid="ignore"):
        c_norm = c_at / b_at

    scale_x = compute_acf_gaussian_fft(np.array([0.0]), t, x, bin_width)[0][0]
    scale_y = compute_acf_gaussian_fft(np.array([0.0]), s, y, bin_width)[0][0]
    scale = np.sqrt(scale_x * scale_y)

    c = c_norm / scale
    return c, b_at
