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

from .utils import standardize, effective_span, padded_angular_map
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


def _nufft_cross_spectrum_at_lags(t, x, s, y, lags, eff_span, N1, eps):
    """NUFFT cross-spectrum X̂*(f)·Ŷ(f) evaluated at the requested lags.

    `eff_span` (>= the true union span of t and s) sets the periodic-domain
    size used for the NUFFT mapping; see `utils.effective_span` /
    `utils.padded_angular_map` for why this is needed to avoid a spurious
    periodic wrap-around contaminating lags close to the data's span.
    """
    xc = x.astype(np.complex128)
    yc = y.astype(np.complex128)

    t_min, span = _common_time_norm(t, s)
    t_norm = padded_angular_map(t, t_min, span, eff_span)
    s_norm = padded_angular_map(s, t_min, span, eff_span)
    lags_norm = lags / eff_span * (2 * np.pi)

    if N1 is None:
        # Not reached via the public compute_ccf_*_nufft API (both wrappers
        # always pass a concrete, eff_span-scaled N1_val) -- kept here only
        # as a safe fallback for direct/internal use of this private
        # function. Mirrors the public default: see
        # compute_ccf_gaussian_nufft's docstring / CHANGELOG.
        N1 = int(round(32 * max(len(x), len(y)) * eff_span / span))

    f1 = finufft.nufft1d1(t_norm, xc, (N1,), eps=eps)
    f2 = finufft.nufft1d1(s_norm, yc, (N1,), eps=eps)
    mul = np.conj(f1) * f2
    return finufft.nufft1d2(lags_norm, mul, eps=eps).real


def _acf_scale_at_lag0(t, x_std, t_min, span, eff_span, N1, eps, bin_width, kernel):
    """Smoothed ACF scale at lag=0 using the COMMON time span.

    IMPORTANT: `eff_span` must be the *same* value used for the main CCF
    numerator (`_nufft_cross_spectrum_at_lags`). If this scale used the raw
    `span` while the numerator used a padded `eff_span`, the two would be
    computed on different periodic domains and the fix above would not
    actually remove the wrap-around bias -- it would just move it from the
    numerator into an inconsistent numerator/denominator ratio instead.

    The other critical detail: lag=0 is placed at an INTERIOR position of a
    small symmetric lags array [-n_half … 0 … n_half] so that
    gaussian_filter1d applies a fully symmetric kernel there — exactly as
    it does at the CCF peak (also an interior position in the lags_eval
    array).

    Without this, the Gaussian-smoothed cross-spectrum peak (interior,
    symmetric smoothing) was divided by an unsmoothed ACF scale, giving a
    systematic ~2-3% deficit in the CCF peak.
    """
    xc = x_std.astype(np.complex128)
    t_norm = padded_angular_map(t, t_min, span, eff_span)
    f1 = finufft.nufft1d1(t_norm, xc, (N1,), eps=eps)
    acf_pow = (np.abs(f1) ** 2).astype(np.complex128)

    # Symmetric lags array: lag=0 at index n_half (interior, ≥ 4·sigma from edges)
    n_half = max(int(np.ceil(4.0 * bin_width)) + 2, 5)
    lags_sym = np.arange(-n_half, n_half + 1, dtype=float)
    mid = n_half

    lags_sym_norm = lags_sym / eff_span * (2 * np.pi)
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
                Defaults to ``32 * max(len(x), len(y)) * eff_span / span``,
                where ``eff_span = span + 2*max(|lags|)`` (see
                ``utils.effective_span``) and ``span`` is the union range
                of ``t`` and ``s``. The ``eff_span/span`` factor compensates
                for the padded periodic domain used to avoid wrap-around
                (see CHANGELOG): without it, a fixed ``32*n`` would resolve
                the (now wider) periodic domain less finely per unit of
                physical time. Passing an explicit ``N1`` bypasses both the
                base default and this scaling.
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

    # `gaussian_filter1d` smooths along ARRAY INDEX, not physical lag value.
    # `lags_eval` prepends 0.0 in front of `lags` as given by the caller --
    # if `lags[0]` is not itself close to 0 (e.g. `lags` starts at a large
    # negative value), array-adjacent entries can be physically very far
    # apart, and the smoothing kernel incorrectly blends them (observed:
    # the huge lag=0 spectrum value leaking into a neighboring lag with
    # very few real pairs, off by >10x). Sorting by physical lag value
    # before smoothing, then inverting the permutation afterwards, makes
    # array-adjacency match physical-adjacency regardless of the order
    # `lags` was supplied in.
    order = np.argsort(lags_eval, kind="stable")
    inv_order = np.argsort(order)
    lags_sorted = lags_eval[order]

    t_min, span = _common_time_norm(t, s)
    eff_span = effective_span(span, lags_sorted)

    # N1 (NUFFT frequency-grid size) sets the angular resolution of the
    # periodic domain. Padding the domain to eff_span (see utils.py)
    # compresses the real data into a narrower arc of that domain, which
    # -- at a FIXED N1 -- reduces the resolution available per unit
    # PHYSICAL time. Scaling N1 by eff_span/span keeps that resolution
    # roughly constant. NOTE this does not fully eliminate the pre-existing
    # sensitivity of this estimator to N1 (see CHANGELOG): it only offsets
    # the *additional* resolution loss introduced by the padding margin
    # itself, on top of whatever precision the un-padded estimator already
    # had at N1=32*n.
    N1_val = (
        int(round(32 * max(len(x), len(y)) * eff_span / span)) if N1 is None else N1
    )

    c_raw = _nufft_cross_spectrum_at_lags(
        t, x_std, s, y_std, lags_sorted, eff_span, N1_val, eps
    )
    c_sm = gaussian_filter1d(c_raw, sigma=bin_width)

    b_cross = compute_b_gaussian_cross(t, s, lags_sorted, bin_width)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        c_norm = c_sm / b_cross

    # back to the order of lags_eval == [0.0] + lags
    c_norm = c_norm[inv_order]
    b_cross = b_cross[inv_order]

    scale_x = _acf_scale_at_lag0(
        t, x_std, t_min, span, eff_span, N1_val, eps, bin_width, "gaussian"
    )
    scale_y = _acf_scale_at_lag0(
        s, y_std, t_min, span, eff_span, N1_val, eps, bin_width, "gaussian"
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

    # See the identical comment in compute_ccf_gaussian_nufft: sort by
    # physical lag before smoothing (array-index-based) and unsort after,
    # so array-adjacency matches physical-adjacency regardless of the
    # order `lags` was supplied in. With bin_width=0.5 on unit spacing this
    # is a no-op (kernel_size=1), but for larger bin_width the rectangle
    # kernel smooths too and is equally exposed.
    order = np.argsort(lags_eval, kind="stable")
    inv_order = np.argsort(order)
    lags_sorted = lags_eval[order]

    t_min, span = _common_time_norm(t, s)
    eff_span = effective_span(span, lags_sorted)

    # See the identical comment in compute_ccf_gaussian_nufft.
    N1_val = (
        int(round(32 * max(len(x), len(y)) * eff_span / span)) if N1 is None else N1
    )

    c_raw = _nufft_cross_spectrum_at_lags(
        t, x_std, s, y_std, lags_sorted, eff_span, N1_val, eps
    )
    kernel_size = max(1, round(2 * bin_width))
    c_sm = uniform_filter1d(c_raw, size=kernel_size)

    b_cross = compute_b_rectangle_cross(t, s, lags_sorted, bin_width)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        c_norm = c_sm / b_cross

    c_norm = c_norm[inv_order]
    b_cross = b_cross[inv_order]

    scale_x = _acf_scale_at_lag0(
        t, x_std, t_min, span, eff_span, N1_val, eps, bin_width, "rectangle"
    )
    scale_y = _acf_scale_at_lag0(
        s, y_std, t_min, span, eff_span, N1_val, eps, bin_width, "rectangle"
    )
    scale = np.sqrt(scale_x * scale_y)

    c = c_norm[1:] / scale
    b = b_cross[1:]
    return c, b
