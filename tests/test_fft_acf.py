"""
Tests for the `fft_acf` module (regularly-sampled fast path).

Validation strategy: cross-check against the already-tested `realspace`
estimators (exact, O(n) per lag) on the SAME regular data -- these should
agree almost exactly, since both implement the same kernel definitions, one
via direct summation and the other via FFT correlation + filtering. The
`regular` (no-kernel) estimator is additionally checked against Pastas
itself, since it specifically targets numerical agreement with Pastas'
`bin_method="regular"`.
"""

import numpy as np
import pandas as pd
import pytest

from nufftcf import (
    compute_acf_regular_fft,
    compute_acf_rectangle_fft,
    compute_acf_gaussian_fft,
    compute_acf_rectangle_realspace,
    compute_acf_gaussian_realspace,
)

LAGS = np.arange(0.0, 366.0)


def _make_correlated_series(n=3000, alpha=10.0, seed=0):
    """Same AR(1)-like construction as test_nufftcf.py, on a regular
    integer grid (delta_t=1)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n + 200)
    kernel = np.exp(-np.arange(200) / alpha)
    values = np.convolve(noise, kernel, mode="valid")[:n]
    t = np.arange(n, dtype=float)
    return t, values


def test_rejects_irregular_grid():
    rng = np.random.default_rng(0)
    t_irregular = np.sort(rng.uniform(0, 3000, 3000))
    x = rng.standard_normal(3000)
    for fn in (
        compute_acf_regular_fft,
        compute_acf_rectangle_fft,
        compute_acf_gaussian_fft,
    ):
        with pytest.raises(ValueError):
            fn(LAGS, t_irregular, x)


@pytest.mark.parametrize(
    "fn", [compute_acf_rectangle_fft, compute_acf_gaussian_fft, compute_acf_regular_fft]
)
def test_lag0_is_close_to_one(fn):
    t, x = _make_correlated_series()
    c, b = fn(LAGS, t, x)
    assert np.isclose(c[0], 1.0, atol=0.05)


def test_rectangle_fft_matches_realspace_default_bin_width():
    """At the package's documented default bin_width=0.5 (1-day spacing),
    the FFT path should match the realspace reference to numerical
    precision (kernel_size resolves to a single sample on both sides)."""
    t, x = _make_correlated_series()
    c_fft, _ = compute_acf_rectangle_fft(LAGS, t, x, bin_width=0.5)
    c_real, _ = compute_acf_rectangle_realspace(LAGS, t, x, bin_width=0.5)
    assert np.allclose(c_fft, c_real, atol=1e-9)


@pytest.mark.parametrize("bin_width", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
def test_rectangle_fft_matches_realspace_various_bin_widths(bin_width):
    """For non-default bin widths, the discrete box filter can't always
    land exactly on the continuous reference window (see fft_acf.py
    docstring) -- away from lag=0 the agreement should still be tight."""
    t, x = _make_correlated_series()
    c_fft, _ = compute_acf_rectangle_fft(LAGS, t, x, bin_width=bin_width)
    c_real, _ = compute_acf_rectangle_realspace(LAGS, t, x, bin_width=bin_width)
    assert np.nanmean(np.abs(c_fft[1:] - c_real[1:])) < 0.005


@pytest.mark.parametrize("bin_width", [0.3, 0.5, 1.0, 2.0])
def test_gaussian_fft_matches_realspace(bin_width):
    t, x = _make_correlated_series()
    c_fft, _ = compute_acf_gaussian_fft(LAGS, t, x, bin_width=bin_width)
    c_real, _ = compute_acf_gaussian_realspace(LAGS, t, x, bin_width=bin_width)
    assert np.allclose(c_fft, c_real, atol=1e-4)


def test_regular_fft_matches_pastas():
    """The whole point of `compute_acf_regular_fft` is to reproduce Pastas'
    bin_method="regular" (a windowed Pearson correlation) exactly, just
    vectorized instead of one np.corrcoef call per lag. Skips cleanly if
    pastas isn't installed (it's an optional/benchmark dependency)."""
    pastas = pytest.importorskip("pastas")

    t, x = _make_correlated_series(n=2000)
    idx = pd.date_range("2000-01-01", periods=len(x), freq="D")
    sts = pd.Series(x, index=idx)

    acf_pastas = pastas.stats.acf(
        sts, lags=LAGS, bin_method="regular", max_gap=30, min_obs=0
    )
    c_mine, _ = compute_acf_regular_fft(LAGS, t, x)

    acf_pastas_aligned = acf_pastas.reindex(pd.to_timedelta(LAGS, unit="D")).to_numpy()
    assert np.allclose(c_mine, acf_pastas_aligned, atol=1e-9, equal_nan=True)


def test_regular_fft_runs_on_each_demo_signal_without_crashing():
    """Smoke test across the signal types used in the regular-sampling
    notebook, including the degenerate constant series (zero variance)."""
    n = 1000
    t = np.arange(n, dtype=float)
    f = 1.0 / (n / 10)
    signals = {
        "sinus": np.sin(2 * np.pi * f * t),
        "sinus_noisy": np.sin(2 * np.pi * f * t)
        + np.random.default_rng(0).standard_normal(n) * 0.1,
        "exponential": np.exp(-t / (n / 5)),
        "constant": np.ones(n),
        "square": np.sign(np.sin(2 * np.pi * f * t)),
    }
    lags = np.arange(0.0, 100.0)
    for name, x in signals.items():
        for fn, kwargs in [
            (compute_acf_regular_fft, {}),
            (compute_acf_rectangle_fft, dict(bin_width=0.5)),
            (compute_acf_gaussian_fft, dict(bin_width=0.5)),
        ]:
            with np.errstate(divide="ignore", invalid="ignore"):
                c, b = fn(lags, t, x, **kwargs)
            assert c.shape == lags.shape
            if name == "constant":
                assert np.all(np.isnan(c))  # zero variance -> ACF undefined
            else:
                assert np.isfinite(c[0])
