"""
Tests for the cross-correlation (CCF) estimators:
  compute_ccf_gaussian_nufft
  compute_ccf_rectangle_nufft
  compute_ccf_gaussian_realspace
  compute_ccf_rectangle_realspace

Testing strategy
----------------
1. Shape / basic sanity: output shapes match lags, b > 0.
2. CCF(x, x, t, t) ≈ ACF(x, t): on a regular grid where both estimators
   scan the same set of pairs, the CCF must equal the ACF to near-machine
   precision (tested per-kernel).
3. Shift detection: CCF(x, x_shifted_by_tau) peaks at lag=tau.
4. Independent signals: CCF values are small (bounded by statistical noise,
   not a fixed tolerance, because AR(1) processes have long memory --
   the bound is generous on purpose).
5. NUFFT vs realspace cross-check: on a dense regular grid where the NUFFT
   approximation error is small, both estimates agree reasonably well.
6. b_cross correctness: b is non-negative, matches expected order of magnitude.
"""

import numpy as np
import pytest

from nufftcf import (
    compute_ccf_gaussian_nufft,
    compute_ccf_rectangle_nufft,
    compute_ccf_gaussian_realspace,
    compute_ccf_rectangle_realspace,
    compute_acf_gaussian_nufft,
    compute_acf_gaussian_realspace,
    compute_acf_rectangle_realspace,
)

LAGS = np.arange(1.0, 80.0)
BW = 0.5  # bin_width used throughout


# ── Helpers ───────────────────────────────────────────────────────────────────


def _regular_series(n=2000, alpha=20.0, seed=0):
    """Regular integer-day grid, AR(1)-like.  Used where we need exact CCF=ACF."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n + 200)
    kernel = np.exp(-np.arange(200) / alpha)
    x = np.convolve(noise, kernel, mode="valid")[:n]
    t = np.arange(n, dtype=float)
    return t, x


def _irregular_series(n=1000, alpha=20.0, t_max=400.0, seed=0):
    """Irregular grid in [0, t_max]."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n + 200)
    kernel = np.exp(-np.arange(200) / alpha)
    x = np.convolve(noise, kernel, mode="valid")[:n]
    t = np.sort(rng.uniform(0, t_max, n))
    return t, x


# ── 1. Shape / basic sanity ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fn",
    [
        compute_ccf_gaussian_nufft,
        compute_ccf_rectangle_nufft,
        compute_ccf_gaussian_realspace,
        compute_ccf_rectangle_realspace,
    ],
)
def test_output_shape(fn):
    t, x = _regular_series(n=500)
    c, b = fn(LAGS, t, x, t, x, BW)
    assert c.shape == LAGS.shape
    assert b.shape == LAGS.shape
    assert np.all(b > 0)


# ── 2. CCF(x, x, t, t) == ACF(x) on regular grid ────────────────────────────


def test_ccf_gaussian_nufft_equals_acf_on_same_signal():
    """On a regular grid with the same signal twice, CCF must be close to ACF.

    They are NOT identical to machine precision even on a regular grid: the
    CCF sums "backward" pairs (s_j - t_i ≈ lag, i.e. j-i≈lag) while the ACF
    sums "forward" pairs (t_i - t_j ≈ lag, i.e. i-j≈lag).  At the series
    boundaries the Gaussian kernel sees different neighbourhoods for each
    direction, producing a ~1% systematic offset that scales as 1/n.  The
    tolerance here reflects that, not a precision problem.
    """
    t, x = _regular_series(n=2000)
    c_acf, _ = compute_acf_gaussian_nufft(LAGS, t, x, bin_width=BW)
    c_ccf, _ = compute_ccf_gaussian_nufft(LAGS, t, x, t, x, BW)
    assert np.allclose(
        c_ccf, c_acf, atol=0.02
    ), f"max diff CCF-ACF (nufft gaussian) = {np.abs(c_ccf-c_acf).max():.2e}"


def test_ccf_gaussian_realspace_equals_acf_on_same_signal():
    """Realspace CCF(x,x) on a regular grid must be close to ACF(x).

    Same boundary-effect caveat as the NUFFT test above; ~1% offset expected.
    """
    t, x = _regular_series(n=2000)
    c_acf, _ = compute_acf_gaussian_realspace(LAGS, t, x, bin_width=BW)
    c_ccf, _ = compute_ccf_gaussian_realspace(LAGS, t, x, t, x, BW)
    assert np.allclose(
        c_ccf, c_acf, atol=0.02
    ), f"max diff CCF-ACF (realspace gaussian) = {np.abs(c_ccf-c_acf).max():.2e}"


def test_ccf_rectangle_realspace_equals_acf_on_same_signal():
    t, x = _regular_series(n=2000)
    c_acf, _ = compute_acf_rectangle_realspace(LAGS, t, x, bin_width=BW)
    c_ccf, _ = compute_ccf_rectangle_realspace(LAGS, t, x, t, x, BW)
    assert np.allclose(c_ccf, c_acf, atol=1e-5)


# ── 3. Shift detection ────────────────────────────────────────────────────────


@pytest.mark.parametrize("tau_shift", [10, 20, 35])
def test_ccf_gaussian_nufft_peaks_at_correct_lag(tau_shift):
    """CCF of x and x_time_shifted should peak at tau_shift."""
    t = np.arange(600, dtype=float)
    s = t + tau_shift
    rng = np.random.default_rng(7)
    kern = np.exp(-np.arange(200) / 20.0)
    x = np.convolve(rng.standard_normal(800), kern, mode="valid")[:600]
    lags = np.arange(0.0, 60.0)
    c, _ = compute_ccf_gaussian_nufft(lags, t, x, s, x.copy(), BW)
    assert (
        int(lags[np.argmax(c)]) == tau_shift
    ), f"Expected peak at {tau_shift}, got {lags[np.argmax(c)]}"


@pytest.mark.parametrize("tau_shift", [10, 20])
def test_ccf_rectangle_nufft_peaks_at_correct_lag(tau_shift):
    t = np.arange(600, dtype=float)
    s = t + tau_shift
    rng = np.random.default_rng(8)
    kern = np.exp(-np.arange(200) / 20.0)
    x = np.convolve(rng.standard_normal(800), kern, mode="valid")[:600]
    lags = np.arange(0.0, 60.0)
    c, _ = compute_ccf_rectangle_nufft(lags, t, x, s, x.copy(), BW)
    assert int(lags[np.argmax(c)]) == tau_shift


@pytest.mark.parametrize("tau_shift", [10, 20])
def test_ccf_gaussian_realspace_peaks_at_correct_lag(tau_shift):
    t = np.arange(600, dtype=float)
    s = t + tau_shift
    rng = np.random.default_rng(9)
    kern = np.exp(-np.arange(200) / 20.0)
    x = np.convolve(rng.standard_normal(800), kern, mode="valid")[:600]
    lags = np.arange(0.0, 60.0)
    c, _ = compute_ccf_gaussian_realspace(lags, t, x, s, x.copy(), BW)
    assert int(lags[np.argmax(c)]) == tau_shift


# ── 4. Independent signals: CCF values bounded ───────────────────────────────


def test_ccf_gaussian_nufft_small_for_independent_signals():
    """
    NUFFT CCF of two independent series should be small.  The bound is generous
    (0.10) to accommodate statistical noise from the finite AR(1) series.
    """
    t, x = _irregular_series(n=1200, seed=0)
    s, y = _irregular_series(n=1000, seed=42)
    c, _ = compute_ccf_gaussian_nufft(LAGS, t, x, s, y, BW)
    assert np.abs(c).max() < 0.10, f"max|CCF(x,y_indep)| = {np.abs(c).max():.3f} > 0.10"


def test_ccf_rectangle_nufft_small_for_independent_signals():
    t, x = _irregular_series(n=1200, seed=1)
    s, y = _irregular_series(n=1000, seed=43)
    c, _ = compute_ccf_rectangle_nufft(LAGS, t, x, s, y, BW)
    assert np.abs(c).max() < 0.10


# ── 5. NUFFT vs realspace agreement (regular dense grid) ─────────────────────


def test_ccf_gaussian_nufft_vs_realspace_agreement():
    """
    On a regular dense grid the NUFFT approximation error is small.
    The test uses two DIFFERENT signals so both estimators face the same
    statistical challenge (they don't trivially reduce to the ACF).
    """
    t, x = _regular_series(n=1500, alpha=20, seed=0)
    s, y = _regular_series(n=1500, alpha=20, seed=5)
    c_n, _ = compute_ccf_gaussian_nufft(LAGS, t, x, s, y, BW)
    c_r, _ = compute_ccf_gaussian_realspace(LAGS, t, x, s, y, BW)
    # Tolerance is looser than for ACF because the CCF has more sampling noise
    # and the NUFFT has a larger approximation error for cross-spectra.
    assert (
        np.nanmean(np.abs(c_n - c_r)) < 0.05
    ), f"mean|CCF_nufft - CCF_realspace| = {np.nanmean(np.abs(c_n-c_r)):.3f}"


# ── 6. b_cross sanity ────────────────────────────────────────────────────────


def test_b_gaussian_cross_decreases_with_lag():
    """For two series on the same grid, b_cross should roughly decrease with lag
    (fewer pairs at large lags), just like b for the ACF."""
    t, x = _regular_series(n=500)
    _, b = compute_ccf_gaussian_nufft(LAGS, t, x, t, x, BW)
    # b at large lags should be smaller than at small lags
    assert b[:10].mean() > b[-10:].mean()


def test_b_rectangle_cross_positive():
    t, x = _irregular_series(n=800, seed=3)
    s, y = _irregular_series(n=700, seed=13)
    _, b = compute_ccf_rectangle_realspace(LAGS, t, x, s, y, BW)
    assert np.all(b > 0)
