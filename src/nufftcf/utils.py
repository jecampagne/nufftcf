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
    end of the record -- the exact analogue of computing an FFT-based correlation without the
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


def padded_angular_map(vals: np.ndarray, t_min: float, span: float, eff_span: float) -> np.ndarray:
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
