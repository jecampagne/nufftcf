"""
ACF estimation via NUFFT (non-uniform FFT) + Wiener-Khinchin theorem.

These estimators compute the power spectrum of the (irregularly-sampled)
signal via a type-1 NUFFT, then evaluate the implied autocorrelation at the
requested lags via a type-2 NUFFT. This scales roughly as O(n log n),
dramatically faster than the O(n^2) real-space approach for long series --
but it carries a small, known limitation (see README): because it relies on
a finite-domain Fourier representation, irregular/gappy sampling acts as a
"spectral window" that slightly distorts narrowband (e.g. periodic) signals
more than broadband ones. Empirically this is a ~1-3% relative bias in the
ACF amplitude once N1 is large enough (see N1 note below); for an
artifact-free reference, use the `realspace` module instead.

`N1 = 32 * n` was empirically validated (against the exact real-space
estimator) to bring the NUFFT result into close agreement for both gaussian
and rectangle kernels; pushing higher  gives a marginal further
improvement for gaussian on strongly periodic signals, at negligible extra
cost.
"""

import numpy as np
import finufft
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from .kernels import compute_b_gaussian, compute_b_rectangle
from .utils import standardize


def _nufft_power_spectrum_at_lags(t, x, lags, N1, eps):
    """Shared first stage: NUFFT type-1 (time -> frequency) then type-2
    (frequency -> [0.0] + lags), implementing Wiener-Khinchin. Always
    evaluates an extra point at lag=0 (regardless of whether 0 is already
    in `lags`), used downstream to normalize the result -- the NUFFT pipeline
    has its own internal scale convention that has nothing to do with the
    `b` denominator's scale, so dividing by `b` alone does *not* yield a
    properly normalized correlation in [-1, 1]. Returns the raw (unsmoothed)
    correlation at [0.0] + lags, i.e. length len(lags) + 1.
    """
    x_normalized = standardize(x)
    xc = np.complex128(x_normalized)
    n = len(xc)
    t_min, t_max = t.min(), t.max()
    t_norm = (t - t_min) / (t_max - t_min) * (2 * np.pi)
    lags_norm = (lags - 0) / (t_max - t_min) * (2 * np.pi)
    if N1 is None:
        N1 = 32 * n
    f1 = finufft.nufft1d1(t_norm, xc, (N1,), eps=eps)
    mul = f1 * np.conj(f1)
    c_positive = finufft.nufft1d2(lags_norm, mul, eps=eps).real
    return c_positive  # index 0 is lag=0, indices [1:] correspond to `lags`


def compute_acf_gaussian_nufft(lags, t, x, bin_width=0.5, N1=None, eps=1e-9):
    """ACF estimate via NUFFT + gaussian smoothing.

    Parameters
    ----------
    lags : array_like
        Lags at which to evaluate the ACF (same units as `t`, typically days).
    t : array_like
        Sample times, sorted ascending (same units as `lags`).
    x : array_like
        Sample values, same length as `t`.
    bin_width : float
        Gaussian kernel standard deviation (same units as `t`).
    N1 : int, optional
        NUFFT frequency-grid size. Defaults to 32*len(x) in  _nufft_power_spectrum_at_lags
    eps : float
        NUFFT requested precision.

    Returns
    -------
    c, b : ndarray
        ACF estimate and effective pair count, both shape (len(lags),).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    lags = np.asarray(lags, dtype=float)
    lags_eval = np.concatenate(([0.0], lags))

    c_positive = _nufft_power_spectrum_at_lags(t, x, lags_eval, N1, eps)
    c_smoothed = gaussian_filter1d(c_positive, sigma=bin_width)
    b_eval = compute_b_gaussian(t, lags_eval, bin_width)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        c_eval = c_smoothed / b_eval
    c = c_eval[1:] / c_eval[0]  # normalize by the (always computed) lag=0 value
    b = b_eval[1:]
    return c, b


def compute_acf_rectangle_nufft(lags, t, x, bin_width=0.5, N1=None, eps=1e-9):
    """ACF estimate via NUFFT + rectangular (box) smoothing.

    Same parameters and return values as `compute_acf_gaussian_nufft`.
    The smoothing window size (in samples) is derived from `bin_width` and
    the average lag spacing; with the common default bin_width=0.5 and a
    1-day lag spacing, this resolves to a 1-sample window (i.e. no-op),
    matching the gaussian kernel's "non-overlapping bins" behavior at the
    same bin_width.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    lags = np.asarray(lags, dtype=float)
    lags_eval = np.concatenate(([0.0], lags))

    c_positive = _nufft_power_spectrum_at_lags(t, x, lags_eval, N1, eps)

    dlag = np.mean(np.diff(lags)) if len(lags) > 1 else 1.0
    window_size = max(1, int(round(2 * bin_width / dlag)))
    c_smoothed = uniform_filter1d(c_positive, size=window_size, mode="nearest")

    b_eval = compute_b_rectangle(t, lags_eval, bin_width)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        c_eval = c_smoothed / b_eval
    c = c_eval[1:] / c_eval[0]  # normalize by the (always computed) lag=0 value
    b = b_eval[1:]
    return c, b
