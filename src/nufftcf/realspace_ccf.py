"""
Real-space (direct) cross-correlation function (CCF) estimators.

These are the CCF counterparts of ``realspace_acf.py``.  They evaluate the
kernel-weighted correlation sum directly in the time domain, with no
implicit periodicity assumption -- so they are bias-free even for strongly
periodic signals on sparse/gappy grids.

Complexity: O((n_t + n_s) · n_lags) per call (two-pointer over the cross-pairs).

Use these as a reference when comparing with the NUFFT estimators, or
whenever the NUFFT bias on periodic signals is unacceptable.
"""

import numpy as np

from .utils import standardize
from .kernels import (
    compute_b_gaussian,
    compute_b_rectangle,
    compute_b_gaussian_cross,
    compute_b_rectangle_cross,
    compute_c_gaussian,
    compute_c_rectangle,
    compute_c_gaussian_cross,
    compute_c_rectangle_cross,
)


def compute_ccf_gaussian_realspace(lags, t, x, s, y, bin_width=0.5):
    """Exact CCF estimate via direct Gaussian-kernel weighted summation.

    Parameters
    ----------
    lags      : array_like — lags at which to evaluate the CCF.
    t, x      : 1D array_like — times and values of signal 1 (``t`` sorted).
    s, y      : 1D array_like — times and values of signal 2 (``s`` sorted).
    bin_width : float — Gaussian kernel standard deviation (same units as ``t``).

    Returns
    -------
    c : ndarray, shape (len(lags),) — Pearson CCF in [-1, 1].
    b : ndarray, shape (len(lags),) — effective pair count per lag.

    Notes
    -----
    Normalised by  sqrt( c_x(0) · c_y(0) )  where ``c_x(0)`` and ``c_y(0)``
    are the Gaussian-kernel ACF estimates at lag=0 for each signal separately.
    For a standardised signal this is close to 1, but might differ slightly
    due to kernel edge effects near the boundaries of the time range.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    lags = np.asarray(lags, dtype=float)

    x_std = standardize(x)
    y_std = standardize(y)

    lags_eval = np.concatenate(([0.0], lags))

    c_cross_raw = compute_c_gaussian_cross(t, x_std, s, y_std, lags_eval, bin_width)
    b_cross = compute_b_gaussian_cross(t, s, lags_eval, bin_width)
    with np.errstate(divide="ignore", invalid="ignore"):
        c_cross_norm = c_cross_raw / b_cross

    # Normalisation: sqrt(ACF_x(0) * ACF_y(0)) from the real-space ACF of each signal
    c0x_raw = compute_c_gaussian(t, np.array([0.0]), x_std, bin_width)[0]
    b0x = compute_b_gaussian(t, np.array([0.0]), bin_width)[0]
    c0y_raw = compute_c_gaussian(s, np.array([0.0]), y_std, bin_width)[0]
    b0y = compute_b_gaussian(s, np.array([0.0]), bin_width)[0]
    scale = np.sqrt((c0x_raw / b0x) * (c0y_raw / b0y))

    c = c_cross_norm[1:] / scale
    b = b_cross[1:]
    return c, b


def compute_ccf_rectangle_realspace(lags, t, x, s, y, bin_width=0.5):
    """Exact CCF estimate via direct rectangular-kernel weighted summation.

    Parameters
    ----------
    lags, t, x, s, y : see :func:`compute_ccf_gaussian_realspace`.
    bin_width : float — rectangle half-width (same units as ``t``).

    Returns
    -------
    c, b : ndarray — CCF estimate and effective pair count.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    lags = np.asarray(lags, dtype=float)

    x_std = standardize(x)
    y_std = standardize(y)

    lags_eval = np.concatenate(([0.0], lags))

    c_cross_raw = compute_c_rectangle_cross(t, x_std, s, y_std, lags_eval, bin_width)
    b_cross = compute_b_rectangle_cross(t, s, lags_eval, bin_width)
    with np.errstate(divide="ignore", invalid="ignore"):
        c_cross_norm = c_cross_raw / b_cross

    c0x_raw = compute_c_rectangle(t, np.array([0.0]), x_std, bin_width)[0]
    b0x = compute_b_rectangle(t, np.array([0.0]), bin_width)[0]
    c0y_raw = compute_c_rectangle(s, np.array([0.0]), y_std, bin_width)[0]
    b0y = compute_b_rectangle(s, np.array([0.0]), bin_width)[0]
    scale = np.sqrt((c0x_raw / b0x) * (c0y_raw / b0y))

    c = c_cross_norm[1:] / scale
    b = b_cross[1:]
    return c, b
