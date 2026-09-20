"""Small shared helpers."""

import numpy as np
import pandas as pd


def t_numeric_of(series: pd.Series) -> np.ndarray:
    """Convert a pandas Series' DatetimeIndex to a float array of elapsed
    days since the first sample (0.0, dt1, dt2, ...)."""
    t = series.index.to_numpy()
    return (t - t[0]).astype("timedelta64[D]").astype(float)


def standardize(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance standardization (same convention as Pastas'
    `_preprocess`), required so that the ACF estimate at lag~0 is ~1."""
    return (x - np.mean(x)) / np.std(x)


def effective_span(span: float, lags: np.ndarray) -> float:
    """Time-domain margin needed so the NUFFT periodic round-trip never
    wraps real data around onto the requested lags.

    The NUFFT ACF/CCF estimators represent the data on a periodic domain
    of period `span` (union range of the input times). Without margin,
    a requested lag approaching `span` aliases with data from the *other*
    end of the record -- the
    exact analogue of computing an FFT-based correlation without the
    zero-padding to length `n1+n2-1` that `scipy.signal.correlate(...,
    mode="full")` applies internally to get a *linear* (non-circular)
    result.

    This returns a periodic-domain size enlarged by twice the largest
    requested |lag|, which is enough margin: with this eff_span, the alias
    of any requested lag falls entirely outside the physical extent of the
    data, so it can only ever multiply by zero (no real pairs there).
    """
    lag_max = float(np.max(np.abs(lags))) if len(lags) else 0.0
    return span + 2.0 * lag_max


def padded_angular_map(
    vals: np.ndarray, t_min: float, span: float, eff_span: float
) -> np.ndarray:
    """Map physical times onto a centered arc of the [0, 2*pi) NUFFT
    circle, of angular width `2*pi*span/eff_span` (instead of the full
    circle, i.e. `eff_span == span`).

    This is the NUFFT equivalent of zero-padding a time series before an
    FFT: no fictitious sample needs to be materialized in the
    complementary arc -- a NUFFT type-1 transform only ever sums over the
    non-uniform points actually supplied, so "no point placed there" is
    already exactly a zero contribution, just like a genuine zero-valued
    padding sample would be. (Materializing explicit zero-valued samples
    instead is *not* equivalent and should be avoided: they would get
    swept into `standardize()`'s mean/variance and bias the result.)
    """
    theta_data = 2.0 * np.pi * span / eff_span
    return np.pi - theta_data / 2.0 + (vals - t_min) / span * theta_data


def default_N1(n_points: int, span: float, lags: np.ndarray) -> int:
    """The default NUFFT frequency-grid size used by every `compute_*_nufft`
    estimator when `N1` is not explicitly passed.

    ``32 * n_points`` is the empirically-validated base resolution (see
    README); it is then scaled up by ``eff_span / span`` (see
    `effective_span`) to compensate for the padded periodic domain used to
    avoid wrap-around at large lags (CHANGELOG, v0.2.0) -- without this
    scaling, padding the domain would silently reduce the NUFFT resolution
    available per unit of *physical* time, even far from the domain edge.

    Call this yourself, with the same ``lags`` you're about to request,
    to know in advance what ``N1`` a `compute_*_nufft` call will use --
    there is no other way to discover it, since it depends on ``lags``
    (through ``eff_span``) and is not returned by the estimators. Useful
    to log/report alongside results, or as a starting point before passing
    a larger `N1` explicitly for extra precision.

    Parameters
    ----------
    n_points : int
        Number of samples in the (longer, for CCF) series -- i.e. what you
        would pass as ``len(x)`` (ACF) or ``max(len(x), len(y))`` (CCF).
    span : float
        Time span of the data -- ``t.max() - t.min()`` for ACF, or the
        union range of ``t`` and ``s`` for CCF (what
        `effective_span`'s ``span`` argument expects).
    lags : array_like
        The lags you intend to request (same array you'll pass to the
        `compute_*_nufft` call).

    Examples
    --------
    >>> span = t.max() - t.min()
    >>> default_N1(len(x), span, lags)
    """
    eff_span = effective_span(span, lags)
    return int(round(32 * n_points * eff_span / span))
