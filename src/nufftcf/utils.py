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
