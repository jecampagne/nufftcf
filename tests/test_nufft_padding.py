"""
Regression tests for the NUFFT periodic wrap-around fix (`eff_span` /
`padded_angular_map` in `utils.py`, used by `nufft_ccf.py` and
`nufft_acf.py`).

Background
----------
Before the fix, `compute_ccf_rectangle_nufft` / `compute_ccf_gaussian_nufft`
/ `compute_acf_*_nufft` mapped the data's time axis onto the FULL NUFFT
periodic domain [0, 2*pi), of period exactly equal to the observed data
span (`t.max() - t.min()`). Requested lags live *inside* that same period,
not in a padding margin beyond it (unlike `compute_ccf_*_fft`, which zero-
pads to `n1+n2-1` samples before its FFT round-trip -- see
`fft_ccf.py::compute_ccf_rectangle_fft`, which is used here as the
reference/ground truth on the regular-grid case).

As a result, a requested lag approaching the data span aliased with real
data from the *other* end of the record: the estimate could leave the
Pearson-valid range [-1, 1] entirely (values > 1 or < -1), and diverge by
an order of magnitude or more from the FFT/realspace reference as |lag|
approached the span. See CHANGELOG / GH issue for the full diagnosis.

The fix enlarges the periodic domain to
`eff_span = span + 2 * max(|lags|)` (mirroring the `n1+n2-1` margin used
on the FFT side) and maps the real data onto a *centered sub-arc* of the
circle instead of the whole circle -- no fictitious zero-valued sample is
added (a NUFFT type-1 transform already treats "no point here" as a zero
contribution, so this is exact, not an approximation).

These tests exercise the specific regime that was previously broken: lags
covering a large fraction of the total time span, on both a regular grid
(compared against `_fft`, which is exact here) and an irregular grid
(compared against `_realspace`, which is exact everywhere by construction).
"""

import numpy as np
import pytest

from nufftcf import (
    compute_ccf_rectangle_fft,
    compute_ccf_rectangle_nufft,
    compute_ccf_gaussian_nufft,
    compute_ccf_rectangle_realspace,
    compute_ccf_gaussian_realspace,
    compute_acf_rectangle_fft,
    compute_acf_rectangle_nufft,
    compute_acf_gaussian_nufft,
    compute_acf_rectangle_realspace,
    compute_acf_gaussian_realspace,
)

BW = 0.5


# ── Fixtures: the two datasets used throughout the original diagnosis ──────────


def _regular_sine_dataset(seed=0, n=128, dt_shift_frac=0.2, noise=0.2):
    """Small, densely-sampled, EXACTLY periodic signal on a regular grid --
    empirically the most unfavorable case found for the wrap-around bug
    (agreement used to degrade already beyond lag/span ~ 0.05-0.08)."""
    rng = np.random.default_rng(seed)
    ts = np.arange(n).astype(float)
    sig = np.sin(2 * np.pi * ts / n)
    shift = dt_shift_frac * n
    sig_noise = np.sin(2 * np.pi * (ts - shift) / n) + noise * rng.standard_normal(n)
    return ts, sig, sig_noise, shift


def _irregular_ou_dataset(seed=123, n_days=3650, tau0=60, alpha=10.0, keep_frac=0.5):
    """Larger, irregularly-sampled, decaying-autocorrelation (OU-like)
    process -- representative of the package's typical/validated use case,
    used here far outside the previously-safe lag/span range."""
    rng = np.random.default_rng(seed)
    buffer = 200
    noise = rng.standard_normal(n_days + tau0 + buffer)
    kern = np.exp(-np.arange(buffer) / alpha)
    e = np.convolve(noise, kern, mode="valid")[: n_days + tau0]
    e = (e - e.mean()) / e.std()
    x_full = e[tau0:]
    y_full = e[:n_days]
    mask = rng.random(n_days) > (1 - keep_frac)
    t_num = np.arange(n_days).astype(float)[mask]
    x = x_full[mask]
    y = y_full[mask]
    return t_num, x, y, tau0


# ── Regular grid: NUFFT must track FFT (exact here) over the FULL lag range ────


@pytest.mark.parametrize("kernel", ["rectangle", "gaussian"])
def test_ccf_nufft_matches_fft_full_lag_range(kernel):
    """Previously: max|fft - nufft| ~ 35 (nufft values up to ~38, far
    outside [-1, 1]) once lags approached the full span. After the fix,
    NUFFT must track the exact FFT reference tightly everywhere."""
    ts, sig, sig_noise, _ = _regular_sine_dataset()
    n = len(ts)
    lags = np.arange(-(n - 1), n, dtype=float)  # full valid range, |lag| up to n-1

    if kernel == "rectangle":
        c_fft, _ = compute_ccf_rectangle_fft(lags, ts, sig, ts, sig_noise, bin_width=BW)
        c_nufft, _ = compute_ccf_rectangle_nufft(
            lags, ts, sig, ts, sig_noise, bin_width=BW
        )
    else:
        from nufftcf import compute_ccf_gaussian_fft

        c_fft, _ = compute_ccf_gaussian_fft(lags, ts, sig, ts, sig_noise, bin_width=BW)
        c_nufft = compute_ccf_gaussian_nufft(
            lags, ts, sig, ts, sig_noise, bin_width=BW
        )[0]

    # NOTE: no hard [-1, 1] bound here -- the smoothed-kernel estimator can
    # itself slightly exceed 1 near a strong peak even in the `_fft`
    # reference (a known ~2-3% characteristic of this normalisation, see
    # `_acf_scale_at_lag0`'s docstring); that is not what this test is
    # about. What matters is that nufft TRACKS the fft reference closely
    # everywhere, including at |lag| approaching the full span (where it
    # used to diverge by an order of magnitude or more).
    assert np.all(np.isfinite(c_nufft))
    assert np.nanmax(np.abs(c_fft - c_nufft)) < 0.01


@pytest.mark.parametrize("kernel", ["rectangle", "gaussian"])
def test_acf_nufft_matches_fft_full_lag_range(kernel):
    ts, sig, _, _ = _regular_sine_dataset()
    n = len(ts)
    lags = np.arange(-(n - 1), n, dtype=float)

    if kernel == "rectangle":
        c_fft, _ = compute_acf_rectangle_fft(lags, ts, sig, bin_width=BW)
        c_nufft, _ = compute_acf_rectangle_nufft(lags, ts, sig, bin_width=BW)
    else:
        from nufftcf import compute_acf_gaussian_fft

        c_fft, _ = compute_acf_gaussian_fft(lags, ts, sig, bin_width=BW)
        c_nufft, _ = compute_acf_gaussian_nufft(lags, ts, sig, bin_width=BW)

    assert np.all(np.isfinite(c_nufft))
    assert np.nanmax(np.abs(c_fft - c_nufft)) < 0.01


# ── Irregular grid: NUFFT must track realspace (exact) up to lag/span ~ 0.9 ────

_OU_LAGS = np.array(
    [20, 60, 100, 200, 400, 700, 1000, 1400, 1800, 2200, 2600, 3000, 3300], dtype=float
)

# Rectangle: tight tolerance, this is exactly what this branch fixes.
# Gaussian: a separate, PRE-EXISTING NUFFT+gaussian-kernel approximation
# residual near the true peak (lag=tau0) remains on gappy irregular data
# -- present identically before and after this branch's two fixes (see
# discussion / CHANGELOG), NOT caused or fixed by the padding/sort changes
# here. The far-lag values (where wrap-around used to dominate, up to
# diff~0.44 before this branch) ARE tightly fixed; only the near-peak
# residual keeps a looser bound, tracked separately as a follow-up.
_TOL = {"rectangle": 0.1, "gaussian": 0.25}


@pytest.mark.parametrize("kernel", ["rectangle", "gaussian"])
def test_ccf_nufft_matches_realspace_large_lag_fraction(kernel):
    """Previously: at lag/span ~ 0.9, |nufft - realspace| reached ~0.3-0.4
    (and grew further beyond, e.g. ~0.44 at lag=3000 for gaussian). After
    the fix, the far-lag wrap-around contamination is gone; what tolerance
    remains reflects ordinary NUFFT approximation error (see `_TOL`)."""
    t_num, x, y, tau0 = _irregular_ou_dataset()
    span = t_num.max() - t_num.min()
    lags = _OU_LAGS
    assert (
        lags.max() / span > 0.85
    )  # sanity: this test IS in the previously-broken regime

    if kernel == "rectangle":
        c_real, _ = compute_ccf_rectangle_realspace(
            lags, t_num, x, t_num, y, bin_width=BW
        )
        c_nufft, _ = compute_ccf_rectangle_nufft(lags, t_num, x, t_num, y, bin_width=BW)
    else:
        c_real, _ = compute_ccf_gaussian_realspace(
            lags, t_num, x, t_num, y, bin_width=BW
        )
        c_nufft, _ = compute_ccf_gaussian_nufft(lags, t_num, x, t_num, y, bin_width=BW)

    assert np.all(np.isfinite(c_nufft))
    # far lags specifically (>50% of span): this is the previously-broken
    # wrap-around regime and must now be tight regardless of kernel.
    far = lags / span > 0.5
    assert np.nanmax(np.abs(c_real[far] - c_nufft[far])) < 0.1
    assert np.nanmax(np.abs(c_real - c_nufft)) < _TOL[kernel]

    # the true peak (lag=tau0) should still be the clear maximum
    peak_lag = lags[np.argmax(c_nufft)]
    assert abs(peak_lag - tau0) <= 20


@pytest.mark.parametrize("kernel", ["rectangle", "gaussian"])
def test_acf_nufft_matches_realspace_large_lag_fraction(kernel):
    t_num, x, _, _ = _irregular_ou_dataset()
    span = t_num.max() - t_num.min()
    lags = _OU_LAGS
    assert lags.max() / span > 0.85

    if kernel == "rectangle":
        c_real, _ = compute_acf_rectangle_realspace(lags, t_num, x, bin_width=BW)
        c_nufft, _ = compute_acf_rectangle_nufft(lags, t_num, x, bin_width=BW)
    else:
        c_real, _ = compute_acf_gaussian_realspace(lags, t_num, x, bin_width=BW)
        c_nufft, _ = compute_acf_gaussian_nufft(lags, t_num, x, bin_width=BW)

    assert np.all(np.isfinite(c_nufft))
    far = lags / span > 0.5
    assert np.nanmax(np.abs(c_real[far] - c_nufft[far])) < 0.1
    assert np.nanmax(np.abs(c_real - c_nufft)) < _TOL[kernel]


# ── Distinct bug found while writing the above: array-index vs physical-lag ────
# `lags_eval = [0.0] + lags` puts a huge lag=0 value right next to
# `lags[0]` in ARRAY position. `gaussian_filter1d`/`uniform_filter1d`
# smooth along array index, so if `lags[0]` is physically far from 0 (as
# in the full-range test above, which starts at lag=-(n-1)), the smoothing
# kernel blends the two -- independently of, and in addition to, the
# wrap-around bug. Confirmed pre-existing (reproduced on the original,
# un-padded code too). Fixed by sorting `lags_eval` by physical value
# before smoothing and un-sorting after.


@pytest.mark.parametrize("kernel", ["rectangle", "gaussian"])
def test_ccf_nufft_lag0_does_not_leak_into_extreme_lag(kernel):
    """Regression test for the array-index/physical-lag smoothing bug,
    isolated from the wrap-around fix: a large bin_width is used so even
    the rectangle kernel's uniform_filter1d actually smooths (kernel_size
    > 1), which the default bin_width=0.5 does not exercise."""
    ts, sig, sig_noise, _ = _regular_sine_dataset()
    n = len(ts)
    bw = 3.0  # large enough that the rectangle kernel also smooths
    lags = np.arange(-(n - 1), n, dtype=float)

    if kernel == "rectangle":
        c_fft, _ = compute_ccf_rectangle_fft(lags, ts, sig, ts, sig_noise, bin_width=bw)
        c_nufft, _ = compute_ccf_rectangle_nufft(
            lags, ts, sig, ts, sig_noise, bin_width=bw
        )
    else:
        from nufftcf import compute_ccf_gaussian_fft

        c_fft, _ = compute_ccf_gaussian_fft(lags, ts, sig, ts, sig_noise, bin_width=bw)
        c_nufft, _ = compute_ccf_gaussian_nufft(
            lags, ts, sig, ts, sig_noise, bin_width=bw
        )

    # index 0 (lag = -(n-1)) is the one directly adjacent, in lags_eval,
    # to the injected lag=0 -- exactly where the leak previously showed up.
    # Before this fix: diff ~28.8 (rectangle) / much worse (gaussian) --
    # the lag=0 spectrum value dominating a single-pair (b=1) lag entirely.
    # After: a small residual remains (~0.08-0.14), consistent with an
    # ordinary filter *edge* effect (scipy's `uniform_filter1d` /
    # `gaussian_filter1d` boundary handling at the edge of the short
    # `lags_eval` array, at the single least-significant b=1 point) rather
    # than the lag=0 leak -- two orders of magnitude smaller, and not
    # growing with bin_width the way the original leak did.
    assert abs(c_fft[0] - c_nufft[0]) < 0.2


# ── Sanity: eff_span helper itself ──────────────────────────────────────────────


def test_effective_span_helper():
    from nufftcf.utils import effective_span, padded_angular_map

    span = 100.0
    lags = np.array([-30.0, 10.0, 50.0])
    eff = effective_span(span, lags)
    assert eff == pytest.approx(span + 2 * 50.0)

    # with lag_max=0 (e.g. ACF at lag=0 only), no padding is added
    assert effective_span(span, np.array([0.0])) == pytest.approx(span)

    # mapped real-data points must land strictly inside (0, 2*pi)
    t = np.linspace(0.0, span, 50)
    mapped = padded_angular_map(t, 0.0, span, eff)
    assert mapped.min() > 0.0
    assert mapped.max() < 2 * np.pi
    # and, with padding, they must NOT cover the full circle
    assert (mapped.max() - mapped.min()) < 2 * np.pi - 1e-9
