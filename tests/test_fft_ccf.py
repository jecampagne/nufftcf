"""
Tests for the `fft_ccf` module (regularly-sampled CCF fast path):
  compute_ccf_rectangle_fft
  compute_ccf_gaussian_fft

Validation strategy: cross-check against the already-tested `realspace` CCF
estimators (exact, O(n) per lag) on the SAME regular data -- these should
agree almost exactly, since both implement the same kernel definitions, one
via direct summation and the other via FFT cross-correlation + filtering
(unlike the `nufft` family, there is no NUFFT approximation error in the
picture, so agreement here is expected to be much tighter than the
`nufft`-vs-`realspace` tolerances used in `test_ccf.py`).

Also exercises: CCF(x, x) == ACF(x) on the same regular grid, shift
detection, boundedness for independent signals, grid-compatibility error
handling (mismatched dt, non-lattice offset, irregular grid), and support
for two series of different lengths / with a nonzero integer lattice offset.
"""

import numpy as np
import pytest

from nufftcf import (
    compute_ccf_rectangle_fft,
    compute_ccf_gaussian_fft,
    compute_ccf_rectangle_realspace,
    compute_ccf_gaussian_realspace,
    compute_acf_rectangle_fft,
    compute_acf_gaussian_fft,
    compute_acf_rectangle_realspace,
    compute_acf_gaussian_realspace,
)

LAGS = np.arange(-60.0, 61.0)
BW = 0.5  # bin_width used throughout


# ── Helpers ───────────────────────────────────────────────────────────────────


def _regular_series(n=2000, alpha=20.0, seed=0):
    """Regular integer-day grid, AR(1)-like -- same construction as
    test_ccf.py / test_fft_acf.py."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n + 200)
    kernel = np.exp(-np.arange(200) / alpha)
    x = np.convolve(noise, kernel, mode="valid")[:n]
    t = np.arange(n, dtype=float)
    return t, x


# ── 1. Shape / basic sanity ───────────────────────────────────────────────────


@pytest.mark.parametrize("fn", [compute_ccf_rectangle_fft, compute_ccf_gaussian_fft])
def test_output_shape(fn):
    t, x = _regular_series(n=500)
    c, b = fn(LAGS, t, x, t, x, BW)
    assert c.shape == LAGS.shape
    assert b.shape == LAGS.shape
    assert np.all(b > 0)


# ── 2. Grid-compatibility error handling ─────────────────────────────────────


@pytest.mark.parametrize("fn", [compute_ccf_rectangle_fft, compute_ccf_gaussian_fft])
def test_rejects_irregular_grid(fn):
    rng = np.random.default_rng(0)
    t_irregular = np.sort(rng.uniform(0, 1000, 500))
    x = rng.standard_normal(500)
    with pytest.raises(ValueError):
        fn(LAGS, t_irregular, x, t_irregular, x, BW)


@pytest.mark.parametrize("fn", [compute_ccf_rectangle_fft, compute_ccf_gaussian_fft])
def test_rejects_mismatched_dt(fn):
    rng = np.random.default_rng(0)
    t = np.arange(200, dtype=float) * 1.0
    s = np.arange(200, dtype=float) * 2.0
    x = rng.standard_normal(200)
    y = rng.standard_normal(200)
    with pytest.raises(ValueError):
        fn(LAGS, t, x, s, y, BW)


@pytest.mark.parametrize("fn", [compute_ccf_rectangle_fft, compute_ccf_gaussian_fft])
def test_rejects_non_lattice_offset(fn):
    rng = np.random.default_rng(0)
    t = np.arange(200, dtype=float)
    s = np.arange(200, dtype=float) + 0.37  # not a multiple of dt=1
    x = rng.standard_normal(200)
    y = rng.standard_normal(200)
    with pytest.raises(ValueError):
        fn(LAGS, t, x, s, y, BW)


@pytest.mark.parametrize("fn", [compute_ccf_rectangle_fft, compute_ccf_gaussian_fft])
def test_accepts_integer_lattice_offset_and_different_lengths(fn):
    """A nonzero-but-commensurate offset between `t` and `s`, and two series
    of different lengths, must both be handled without error."""
    rng = np.random.default_rng(1)
    t = np.arange(300, dtype=float)
    s = np.arange(180, dtype=float) + 5.0  # k0 = 5, shorter series
    x = rng.standard_normal(300)
    y = rng.standard_normal(180)
    c, b = fn(np.arange(-20.0, 21.0), t, x, s, y, BW)
    assert np.all(np.isfinite(c))
    assert np.all(b > 0)


# ── 3. CCF(x, x, t, t) == ACF(x, t) on regular grid ─────────────────────────


# compute_acf_*_fft only evaluates non-negative lags (valid = lag_idx in
# [0, n)); restrict the CCF-vs-ACF comparison to that range so we aren't
# comparing a real CCF value against an ACF NaN.
LAGS_NONNEG = np.arange(0.0, 61.0)


def test_ccf_gaussian_fft_equals_acf_fft_on_same_signal():
    """On a regular grid with the same signal twice, CCF must be close to
    ACF (same ~1% boundary-effect caveat noted in test_ccf.py: the CCF sums
    "backward" pairs while the ACF sums "forward" pairs, so the two are not
    identical to machine precision, but very close)."""
    t, x = _regular_series(n=2000)
    c_acf, _ = compute_acf_gaussian_fft(LAGS_NONNEG, t, x, bin_width=BW)
    c_ccf, _ = compute_ccf_gaussian_fft(LAGS_NONNEG, t, x, t, x, BW)
    assert np.allclose(
        c_ccf, c_acf, atol=0.02
    ), f"max diff CCF-ACF (fft gaussian) = {np.nanmax(np.abs(c_ccf - c_acf)):.2e}"


def test_ccf_rectangle_fft_equals_acf_fft_on_same_signal():
    """The rectangle kernel has no discarded-half-then-reconstructed step
    (unlike gaussian), so CCF(x,x) should match ACF(x) far more tightly."""
    t, x = _regular_series(n=2000)
    c_acf, _ = compute_acf_rectangle_fft(LAGS_NONNEG, t, x, bin_width=BW)
    c_ccf, _ = compute_ccf_rectangle_fft(LAGS_NONNEG, t, x, t, x, BW)
    assert np.allclose(
        c_ccf, c_acf, atol=1e-9
    ), f"max diff CCF-ACF (fft rectangle) = {np.nanmax(np.abs(c_ccf - c_acf)):.2e}"


# ── 4. FFT vs realspace agreement (exact on a regular grid) ─────────────────


def test_ccf_gaussian_fft_matches_realspace():
    """No NUFFT approximation error is involved here (unlike
    `nufft`-vs-`realspace` in test_ccf.py), so agreement on `c` should be
    tight. `b` itself is NOT compared: like `compute_acf_gaussian_fft`
    (see test_fft_acf.py, which only ever checks `c`), the Gaussian-kernel
    `b` is a discretized-filter approximation of a continuous kernel-weighted
    pair count, and only its RATIO with the correlation numerator -- computed
    through the exact same discrete filter -- is guaranteed to cancel
    discretization artifacts and match the realspace/NUFFT definition; the
    two families' raw `b` values are not expected to agree bit-for-bit."""
    t, x = _regular_series(n=1500, alpha=20, seed=0)
    s, y = _regular_series(n=1500, alpha=20, seed=5)
    c_fft, _ = compute_ccf_gaussian_fft(LAGS, t, x, s, y, BW)
    c_real, _ = compute_ccf_gaussian_realspace(LAGS, t, x, s, y, BW)
    assert np.allclose(c_fft, c_real, atol=1e-6)


def test_ccf_rectangle_fft_matches_realspace():
    t, x = _regular_series(n=1500, alpha=20, seed=0)
    s, y = _regular_series(n=1500, alpha=20, seed=5)
    c_fft, b_fft = compute_ccf_rectangle_fft(LAGS, t, x, s, y, BW)
    c_real, b_real = compute_ccf_rectangle_realspace(LAGS, t, x, s, y, BW)
    assert np.allclose(c_fft, c_real, atol=1e-9)
    assert np.allclose(b_fft, b_real, atol=1e-9)


@pytest.mark.parametrize("bin_width", [0.3, 0.5, 1.0, 2.0])
def test_ccf_gaussian_fft_matches_realspace_various_bin_widths(bin_width):
    t, x = _regular_series(n=1200, alpha=15, seed=2)
    s, y = _regular_series(n=1200, alpha=15, seed=9)
    c_fft, _ = compute_ccf_gaussian_fft(LAGS, t, x, s, y, bin_width)
    c_real, _ = compute_ccf_gaussian_realspace(LAGS, t, x, s, y, bin_width)
    assert np.allclose(c_fft, c_real, atol=1e-5)


# ── 5. Shift detection ────────────────────────────────────────────────────────


@pytest.mark.parametrize("tau_shift", [10, 20, 35])
def test_ccf_gaussian_fft_peaks_at_correct_lag(tau_shift):
    """CCF of x and x_time_shifted should peak at tau_shift (positive lag
    means the second signal lags behind the first -- see module docstring)."""
    t = np.arange(600, dtype=float)
    s = t + tau_shift
    rng = np.random.default_rng(7)
    kern = np.exp(-np.arange(200) / 20.0)
    x = np.convolve(rng.standard_normal(800), kern, mode="valid")[:600]
    lags = np.arange(0.0, 60.0)
    c, _ = compute_ccf_gaussian_fft(lags, t, x, s, x.copy(), BW)
    assert (
        int(lags[np.argmax(c)]) == tau_shift
    ), f"Expected peak at {tau_shift}, got {lags[np.argmax(c)]}"


@pytest.mark.parametrize("tau_shift", [10, 20])
def test_ccf_rectangle_fft_peaks_at_correct_lag(tau_shift):
    t = np.arange(600, dtype=float)
    s = t + tau_shift
    rng = np.random.default_rng(8)
    kern = np.exp(-np.arange(200) / 20.0)
    x = np.convolve(rng.standard_normal(800), kern, mode="valid")[:600]
    lags = np.arange(0.0, 60.0)
    c, _ = compute_ccf_rectangle_fft(lags, t, x, s, x.copy(), BW)
    assert int(lags[np.argmax(c)]) == tau_shift


# ── 6. Independent signals: CCF values bounded ───────────────────────────────


def test_ccf_gaussian_fft_small_for_independent_signals():
    """CCF of two independent series should be small. The bound is generous
    to accommodate statistical noise from the finite AR(1) series (same
    reasoning as test_ccf.py's nufft equivalent); `n=3000`/`alpha=8` here
    (vs. `n=1200`/`alpha=20` in test_ccf.py) because this test's `LAGS`
    spans +-60 on a full regular grid (denser sampling of a long-memory
    AR(1) process needs a somewhat longer/less persistent series to keep
    spurious correlation this generous bound away from)."""
    t, x = _regular_series(n=3000, alpha=8.0, seed=0)
    s, y = _regular_series(n=3000, alpha=8.0, seed=42)
    c, _ = compute_ccf_gaussian_fft(LAGS, t, x, s, y, BW)
    assert np.abs(c).max() < 0.20, f"max|CCF(x,y_indep)| = {np.abs(c).max():.3f} > 0.20"


def test_ccf_rectangle_fft_small_for_independent_signals():
    t, x = _regular_series(n=3000, alpha=8.0, seed=1)
    s, y = _regular_series(n=3000, alpha=8.0, seed=43)
    c, _ = compute_ccf_rectangle_fft(LAGS, t, x, s, y, BW)
    assert np.abs(c).max() < 0.20


# ── 7. b sanity ───────────────────────────────────────────────────────────────


def test_b_gaussian_fft_decreases_away_from_best_overlap():
    """For two series on the same grid (`s == t`), b should be largest at
    lag=0 and decrease as |lag| grows (fewer overlapping pairs)."""
    t, x = _regular_series(n=500)
    _, b = compute_ccf_gaussian_fft(LAGS, t, x, t, x, BW)
    mid = len(LAGS) // 2
    assert LAGS[mid] == 0.0
    assert b[mid] > b[0]
    assert b[mid] > b[-1]


def test_b_rectangle_fft_positive_with_offset_and_different_lengths():
    rng = np.random.default_rng(3)
    t = np.arange(400, dtype=float)
    s = np.arange(350, dtype=float) + 3.0
    x = rng.standard_normal(400)
    y = rng.standard_normal(350)
    _, b = compute_ccf_rectangle_fft(LAGS, t, x, s, y, BW)
    assert np.all(b > 0)


# ── 8. Smoke test across demo-style signals ──────────────────────────────────


def test_ccf_fft_runs_on_each_demo_signal_without_crashing():
    n = 800
    t = np.arange(n, dtype=float)
    f = 1.0 / (n / 10)
    signals = {
        "sinus": np.sin(2 * np.pi * f * t),
        "sinus_noisy": np.sin(2 * np.pi * f * t)
        + np.random.default_rng(0).standard_normal(n) * 0.1,
        "exponential": np.exp(-t / (n / 5)),
        "square": np.sign(np.sin(2 * np.pi * f * t)),
    }
    lags = np.arange(-50.0, 51.0)
    for name, x in signals.items():
        for fn in (compute_ccf_rectangle_fft, compute_ccf_gaussian_fft):
            with np.errstate(divide="ignore", invalid="ignore"):
                c, b = fn(lags, t, x, t, x, BW)
            assert c.shape == lags.shape
            assert np.isfinite(c).all()
