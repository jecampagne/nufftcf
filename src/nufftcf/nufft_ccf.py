"""
Cross-correlation function (CCF) estimation via NUFFT + Wiener-Khinchin,
for two irregularly-sampled 1D signals with DIFFERENT sampling grids.

Mathematical note on the sign convention
-----------------------------------------
finufft nufft1d2 with isign=+1 computes  f_j = Σ_k F_k · e^{-i k x_j}
(negative exponent).  For the ACF the power spectrum |X̂|² is Hermitian-
symmetric so the ±i convention does not matter.  For the CCF the
cross-spectrum X̂*(f)·Ŷ(f) is NOT Hermitian in general:

    mul = conj(f1) * f2   →  CCF at  +τ  (correct)
    mul = f1 * conj(f2)   →  CCF at  -τ  (wrong)

Normalisation
-------------
Both x and y are standardised internally.  The NUFFT internal scale is
removed by dividing by sqrt(ACF_x(0)·ACF_y(0)).
"""

import numpy as np
import finufft
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from .utils import standardize
from .kernels import (
    compute_b_gaussian,
    compute_b_rectangle,
    compute_b_gaussian_cross,
    compute_b_rectangle_cross,
)


def _common_time_norm(t, s):
    """Return (t_min, span) covering the union of t and s ranges."""
    t_min = min(float(t.min()), float(s.min()))
    t_max = max(float(t.max()), float(s.max()))
    span = t_max - t_min
    if span == 0:
        span = 1.0
    return t_min, span


def _nufft_cross_spectrum_at_lags(t, x, s, y, lags, N1, eps):
    """NUFFT cross-spectrum X̂*(f)·Ŷ(f) evaluated at the requested lags."""
    xc = x.astype(np.complex128)
    yc = y.astype(np.complex128)

    t_min, span = _common_time_norm(t, s)
    t_norm = (t - t_min) / span * (2 * np.pi)
    s_norm = (s - t_min) / span * (2 * np.pi)
    lags_norm = lags / span * (2 * np.pi)

    if N1 is None:
        N1 = 32 * max(len(x), len(y))

    f1 = finufft.nufft1d1(t_norm, xc, (N1,), eps=eps)
    f2 = finufft.nufft1d1(s_norm, yc, (N1,), eps=eps)
    mul = np.conj(f1) * f2
    return finufft.nufft1d2(lags_norm, mul, eps=eps).real


def _acf_scale_at_lag0(t, x_std, t_min, span, N1, eps, bin_width, kernel):
    """Smoothed ACF scale at lag=0 using the COMMON time span.

    The critical detail: lag=0 is placed at an INTERIOR position of a small
    symmetric lags array [-n_half … 0 … n_half] so that gaussian_filter1d
    applies a fully symmetric kernel there — exactly as it does at the CCF
    peak (also an interior position in the lags_eval array).

    Without this, the Gaussian-smoothed cross-spectrum peak (interior,
    symmetric smoothing) was divided by an unsmoothed ACF scale, giving a
    systematic ~2-3% deficit in the CCF peak.
    """
    xc = x_std.astype(np.complex128)
    t_norm = (t - t_min) / span * (2 * np.pi)
    f1 = finufft.nufft1d1(t_norm, xc, (N1,), eps=eps)
    acf_pow = (np.abs(f1) ** 2).astype(np.complex128)

    # Symmetric lags array: lag=0 at index n_half (interior, ≥ 4·sigma from edges)
    n_half = max(int(np.ceil(4.0 * bin_width)) + 2, 5)
    lags_sym = np.arange(-n_half, n_half + 1, dtype=float)
    mid = n_half

    lags_sym_norm = lags_sym / span * (2 * np.pi)
    c_raw_sym = finufft.nufft1d2(lags_sym_norm, acf_pow, eps=eps).real

    if kernel == "gaussian":
        c_sm_sym = gaussian_filter1d(c_raw_sym, sigma=bin_width)
        b_sym = compute_b_gaussian(t, lags_sym, bin_width)
    else:
        kernel_size = max(1, round(2 * bin_width))
        c_sm_sym = uniform_filter1d(c_raw_sym, size=kernel_size)
        b_sym = compute_b_rectangle(t, lags_sym, bin_width)

    b0 = b_sym[mid]
    return c_sm_sym[mid] / b0 if b0 > 0 else 1e-16


def compute_ccf_gaussian_nufft(lags, t, x, s, y, bin_width=0.5, N1=None, eps=1e-9):
    """Cross-correlation estimate via NUFFT + Gaussian smoothing.

    Parameters
    ----------
    lags      : array_like — lags at which to evaluate the CCF (same units
                as ``t`` and ``s``, typically days).
    t, x      : 1D array_like — sample times and values of signal 1
                (``t`` must be sorted ascending).
    s, y      : 1D array_like — sample times and values of signal 2
                (``s`` must be sorted ascending; may differ from ``t``).
    bin_width : float — Gaussian kernel standard deviation (same units as ``t``).
    N1        : int, optional — NUFFT frequency-grid size.
                Defaults to ``32 * max(len(x), len(y))``.
    eps       : float — NUFFT requested precision.

    Returns
    -------
    c : ndarray, shape (len(lags),)
        CCF estimate, normalised to [-1, 1] (Pearson convention).
    b : ndarray, shape (len(lags),)
        Effective Gaussian-weighted pair count per lag.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    lags = np.asarray(lags, dtype=float)

    x_std = standardize(x)
    y_std = standardize(y)

    lags_eval = np.concatenate(([0.0], lags))

    c_raw = _nufft_cross_spectrum_at_lags(t, x_std, s, y_std, lags_eval, N1, eps)
    c_sm = gaussian_filter1d(c_raw, sigma=bin_width)

    b_cross = compute_b_gaussian_cross(t, s, lags_eval, bin_width)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        c_norm = c_sm / b_cross

    t_min, span = _common_time_norm(t, s)
    N1_val = 32 * max(len(x), len(y)) if N1 is None else N1
    scale_x = _acf_scale_at_lag0(
        t, x_std, t_min, span, N1_val, eps, bin_width, "gaussian"
    )
    scale_y = _acf_scale_at_lag0(
        s, y_std, t_min, span, N1_val, eps, bin_width, "gaussian"
    )
    scale = np.sqrt(scale_x * scale_y)

    c = c_norm[1:] / scale
    b = b_cross[1:]
    return c, b


def compute_ccf_rectangle_nufft(lags, t, x, s, y, bin_width=0.5, N1=None, eps=1e-9):
    """Cross-correlation estimate via NUFFT + rectangular smoothing.

    Parameters
    ----------
    lags, t, x, s, y, N1, eps : see :func:`compute_ccf_gaussian_nufft`.
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

    c_raw = _nufft_cross_spectrum_at_lags(t, x_std, s, y_std, lags_eval, N1, eps)
    kernel_size = max(1, round(2 * bin_width))
    c_sm = uniform_filter1d(c_raw, size=kernel_size)

    b_cross = compute_b_rectangle_cross(t, s, lags_eval, bin_width)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        c_norm = c_sm / b_cross

    t_min, span = _common_time_norm(t, s)
    N1_val = 32 * max(len(x), len(y)) if N1 is None else N1
    scale_x = _acf_scale_at_lag0(
        t, x_std, t_min, span, N1_val, eps, bin_width, "rectangle"
    )
    scale_y = _acf_scale_at_lag0(
        s, y_std, t_min, span, N1_val, eps, bin_width, "rectangle"
    )
    scale = np.sqrt(scale_x * scale_y)

    c = c_norm[1:] / scale
    b = b_cross[1:]
    return c, b
