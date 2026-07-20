"""
Basic correctness tests. Run with: pytest tests/
These check internal consistency (lag~0 normalization, NUFFT vs realspace
agreement on a simple case) -- they are NOT timing benchmarks.
"""

import numpy as np
import pandas as pd
import pytest

from nufftcf import (
    compute_acf_gaussian_nufft,
    compute_acf_rectangle_nufft,
    compute_acf_gaussian_realspace,
    compute_acf_rectangle_realspace,
    t_numeric_of,
)


def _make_correlated_series(n=2000, alpha=10.0, seed=0):
    """AR(1)-like series with real short-range correlation (via exponential
    convolution of white noise), on a regular daily grid. Used for the
    lag~0 normalization check: a pure white-noise series would legitimately
    show some dilution under the gaussian kernel's soft (~6 sigma) window,
    even with fully correct code, since neighboring uncorrelated points
    still receive non-negligible weight. A short-range-correlated signal
    avoids that confound."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n + 200)
    kernel = np.exp(-np.arange(200) / alpha)
    values = np.convolve(noise, kernel, mode="valid")[:n]
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    x = pd.Series(values, index=idx)
    return t_numeric_of(x), x.to_numpy()


def _make_regular_series(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    x = pd.Series(rng.standard_normal(n), index=idx)
    return t_numeric_of(x), x.to_numpy()


def _make_irregular_series(n_years=10, drop_fraction=0.3, seed=0):
    rng = np.random.default_rng(seed)
    n_days = int(n_years * 365)
    idx_full = pd.to_datetime(np.arange(n_days), unit="D", origin="2000")
    keep = rng.random(n_days) > drop_fraction
    idx = idx_full[keep]
    x = pd.Series(rng.standard_normal(len(idx)), index=idx)
    return t_numeric_of(x), x.to_numpy()


LAGS = np.arange(0.0, 366.0)


@pytest.mark.parametrize(
    "fn",
    [
        compute_acf_gaussian_nufft,
        compute_acf_rectangle_nufft,
        compute_acf_gaussian_realspace,
        compute_acf_rectangle_realspace,
    ],
)
def test_lag0_is_close_to_one(fn):
    """At lag=0, the ACF of a standardized, short-range-correlated series
    against itself should be ~1."""
    t, x = _make_correlated_series()
    c, b = fn(LAGS, t, x)
    assert np.isclose(c[0], 1.0, atol=0.05)


@pytest.mark.parametrize(
    "fn",
    [
        compute_acf_gaussian_nufft,
        compute_acf_rectangle_nufft,
        compute_acf_gaussian_realspace,
        compute_acf_rectangle_realspace,
    ],
)
def test_runs_on_irregular_data(fn):
    """Smoke test: should run without error on irregular/gappy data and
    return finite values for most lags."""
    t, x = _make_irregular_series()
    c, b = fn(LAGS, t, x)
    assert c.shape == LAGS.shape
    assert np.isfinite(c[1:50]).all()  # short lags should always be well-populated


def test_nufft_vs_realspace_agreement_gaussian():
    """NUFFT and realspace should agree closely (within ~5%) on a regular series."""
    t, x = _make_regular_series(n=3000)
    c_nufft, _ = compute_acf_gaussian_nufft(LAGS, t, x, bin_width=0.5)
    c_real, _ = compute_acf_gaussian_realspace(LAGS, t, x, bin_width=0.5)
    diff = np.abs(c_nufft[1:] - c_real[1:])
    assert np.nanmean(diff) < 0.05


def test_nufft_vs_realspace_agreement_rectangle():
    t, x = _make_regular_series(n=3000)
    c_nufft, _ = compute_acf_rectangle_nufft(LAGS, t, x, bin_width=0.5)
    c_real, _ = compute_acf_rectangle_realspace(LAGS, t, x, bin_width=0.5)
    diff = np.abs(c_nufft[1:] - c_real[1:])
    assert np.nanmean(diff) < 0.05
