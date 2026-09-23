# Changelog

All notable changes to `nufftcf` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed — misleading $O(n\log n)$ complexity claims for the `_nufft` estimators (docs only)

The README's scaling table and "Method" section stated the `_nufft`
estimator family (`compute_acf_gaussian_nufft`, `compute_acf_rectangle_nufft`,
`compute_ccf_gaussian_nufft`, `compute_ccf_rectangle_nufft`) scales as
$O(n\log n)$. That is true of the FINUFFT type-1/type-2 calls alone, but
`compute_*_nufft` also normalizes by the pair count returned by
`kernels.py`'s two-pointer scan, which costs $O(n)$ **per lag** -- i.e.
$O(nK)$ overall for $K$ requested lags, not accounted for by the
$O(n\log n)$ label. For a fixed $K$ (the common case: a benchmark or a
production pipeline evaluating the same lag grid across many series
lengths), $n\log n$ and $nK$ are only weakly distinguishable over 1-2
decades of $n$, and the crossover where $n\log n$ would overtake $nK$
sits at $n \sim e^{K}$ -- unreachable for any realistic $K$.

Refitting `benchmark/*_results_macosx.csv` with both terms free
(`a_lin*n + a_nlogn*n*ln(n) + overhead`, both coefficients constrained
$\geq 0$, see `benchmark/fit_benchmark_acf*.py`) shows the pair-count term
dominates (~100%) for the **Gaussian** kernel (whose two-pointer inner
loop evaluates `exp()` per point in the window) and is comparable to the
NUFFT term (~45%/55% split at $n\sim2.6\times10^5$) for the **rectangle**
kernel (whose inner loop is a plain pointer/cumulative-sum lookup, no
`exp()`), on regularly-sampled data. See the updated tables and Method
section in [README.md](README.md) for the corrected, footnoted claims,
and the discussion around lines 240-270 for the `_realspace`-vs-`_nufft`
comparison, similarly corrected (the two families share the *same*
$O(nK)$ pair-count denominator; `_nufft` only saves the numerator's
$O(nK)$ term, which mattered more for the Gaussian kernel than for the
rectangle one).

**No code or numerical behavior changes** -- this release only corrects
comments and documentation (README.md) and the two benchmark-fitting
scripts (`benchmark/fit_benchmark_acf.py`,
`benchmark/fit_benchmark_acf_regular.py`), which now fit and report both
terms instead of assuming the NUFFT term alone. Outputs of
`compute_*_nufft`/`compute_*_fft`/`compute_*_realspace` are bit-for-bit
identical to v0.2.1 for the same inputs. Benchmark CSVs
(`benchmark/*_results*.csv`) are unchanged (real measurements, nothing to
correct); only the derived `*_fit_summary.csv` and `*_fit.pdf` outputs
differ, reflecting the corrected fit model.

## [0.2.1] - 2026-09-20

### Added

- `nufftcf.default_N1(n_points, span, lags)`: given the same
  `(n_points, span, lags)` a `compute_*_nufft` call will see, returns the
  exact `N1` (NUFFT frequency-grid size) that call will use by default.
  Since v0.2.0, that default depends on `lags` (through `eff_span`, see
  [CHANGELOG](CHANGELOG.md#020---2026-09-19)) rather than being the fixed
  `32 * n_points` it used to be, and there was previously no way to know
  its value ahead of a call other than reimplementing the formula
  yourself. Useful to log/report alongside results, or as a starting point
  before passing a larger `N1` explicitly for extra precision.
- `nufftcf.effective_span` is now also exported at the top level (it was
  previously only reachable via `nufftcf.utils.effective_span`), alongside
  the new `default_N1`.

### Changed

- Internal consolidation: the `N1 = 32 * n_points * eff_span / span`
  formula was independently duplicated at four call sites across
  `nufft_ccf.py` and `nufft_acf.py` (a leftover from how the v0.2.0 fixes
  were added incrementally). All four now call the single
  `utils.default_N1` helper described above -- no behavior change, just
  removes the risk of the copies silently drifting apart in a future edit.

No numerical behavior changes in this release: outputs are bit-for-bit
identical to v0.2.0 for the same inputs.

## [0.2.0] - 2026-09-19

### Fixed — periodic wrap-around in `*_nufft` estimators (CCF and ACF)

`compute_ccf_rectangle_nufft`, `compute_ccf_gaussian_nufft`,
`compute_acf_rectangle_nufft`, and `compute_acf_gaussian_nufft` mapped the
data's time axis onto the full NUFFT periodic domain `[0, 2*pi)`, of period
exactly equal to the observed data span (`t.max() - t.min()`). Requested
lags live *inside* that same period, not in a padding margin beyond it
(unlike `compute_ccf_*_fft`, which zero-pads to `n1+n2-1` samples before
its FFT round-trip specifically to avoid this).

As a result, a requested lag approaching the data span aliased with real
data from the *other* end of the record: the estimate could leave the
Pearson-valid range `[-1, 1]` entirely, and diverge from the `_fft`/
`_realspace` reference by an order of magnitude or more as `|lag|`
approached the span. The bug was invisible in the regime validated by the
existing notebooks (`lag_max` a few % of the span), which is why it went
unnoticed until now.

**Fix**: the NUFFT periodic domain is now padded to
`eff_span = span + 2*max(|lags|)` (mirroring the `n1+n2-1` margin `_fft`
already used), and the real data is mapped onto a centered sub-arc of the
circle instead of the whole circle. No fictitious zero-valued sample is
added — a NUFFT type-1 transform already treats "no point placed here" as
a zero contribution, exactly like a genuine zero-padding sample would.
New helpers: `utils.effective_span`, `utils.padded_angular_map`.

### Fixed — array-index vs. physical-lag smoothing order (CCF and ACF)

A second, independent, pre-existing bug was found while writing regression
tests for the fix above (confirmed present before it too, so it is not a
regression introduced by the padding fix). All four `*_nufft` estimators
build `lags_eval = [0.0] + lags` before smoothing. `gaussian_filter1d` /
`uniform_filter1d` smooth along **array index**, not physical lag value —
so whenever `lags[0]` was not itself close to `0` (e.g. `lags` starting at
a large negative value, or any lag array where 0 falls near an edge rather
than being embedded in it), the huge lag=0 spectrum value leaked into
whatever lag happened to be array-adjacent to it, contaminating an
estimate that could otherwise have very few real sample pairs behind it.

**Fix**: `lags_eval` is now sorted by physical value before smoothing, and
the result is un-sorted back to the caller's order afterwards, so
array-adjacency always matches physical-adjacency regardless of the order
`lags` was supplied in.

### Changed — `N1` (NUFFT frequency-grid size) default

Padding the periodic domain to `eff_span` (see above) compresses the real
data into a narrower arc of that domain, which — at a fixed `N1` — reduces
the NUFFT resolution available per unit of *physical* time, even at lags
far from the domain edge. The default `N1` (previously `32 * n`,
unconditionally) is now scaled by `eff_span / span` to compensate.

This does not fully restore the estimator's baseline precision (see
"Known follow-up" below) — it specifically offsets the *additional*
resolution loss introduced by the padding margin itself. This also
consolidates two previously-duplicated `N1`-default computations (the main
NUFFT round-trip and the CCF's `_acf_scale_at_lag0` helper independently
computed the same default; they were coincidentally always identical
before this change and are now computed once and shared).

### ⚠️ Numerical behavior change

Because of the three changes above, values returned by `compute_ccf_*_nufft`
and `compute_acf_*_nufft` can shift by `O(10⁻³ – 10⁻²)` relative to
`v0.1.x`, **even for lags well within the previously-documented "safe"
range** (`lag_max` a few % of the span). This is expected: the old values
included a small, unpredictable-in-sign bias from the wrap-around bug that
partially masked a separate, pre-existing `N1`-resolution limitation; the
new values no longer have that bias.

Validated on `notebook/nufftcf_ccf_demo.ipynb`'s `rho` sweep (5 values ×
2 kernels): the maximum deviation from the exact `realspace` reference
across the sweep went from `+0.016` (`v0.1.x`) to `±0.007` (this release)
— i.e. the new values are, on average, *closer* to ground truth than
before, not just differently biased. `pip install -e ".[benchmark]"` +
`benchmark/` results confirm no performance regression at realistic
dataset sizes (see "Performance" below).

If you have saved reference outputs from `compute_ccf_*_nufft` /
`compute_acf_*_nufft` (e.g. regression tests in downstream projects, or
published results), expect to regenerate them against this release.
`compute_ccf_*_fft`, `compute_acf_*_fft`, and the `*_realspace` estimators
are entirely unaffected (unchanged code paths, unchanged outputs).

### Added

- `tests/test_nufft_padding.py`: 11 new regression tests covering both
  fixes above, on a regular-grid case (compared against the exact
  `compute_ccf_rectangle_fft` / `compute_acf_rectangle_fft` reference over
  the *full* valid lag range, `|lag|` up to `n-1`) and an irregular-grid
  case (compared against `compute_ccf_*_realspace` / `compute_acf_*_realspace`
  up to `lag/span ≈ 0.9`).

### Performance

`N1` scaling by `eff_span/span` only matters when `lag_max` is a
significant fraction of the span. Benchmarked (`benchmark/`, plus an
additional direct `main`-vs-this-release timing comparison):

- Negligible impact (`~0.8×–1.1×`, within measurement noise) for
  `lag_max ≪ span` — the regime the package targets and that the existing
  benchmarks exercise.
- Up to `~2–3×` slower in the worst case tested (`lag_max` comparable to
  `span`, on a very short series), but the absolute added cost stays in
  the low milliseconds.
- The `O(n log n)` NUFFT advantage over Pastas' `O(n²)` (the package's
  core value proposition) is unaffected: still ~60–280× faster than Pastas
  on `benchmark/benchmark_acf.py` across the tested range.

### Known follow-up (not fixed in this release)

A separate, pre-existing NUFFT+gaussian-kernel approximation residual
(~10–20% relative, on one tested strongly-gappy irregular Ornstein-
Uhlenbeck configuration) remains near the true CCF/ACF peak on irregular
sampling. It is present identically before and after both fixes above, and
is unrelated to either — tracked separately, not addressed here.

## [0.1.4] - 2026-09-16
- CITATION.cff, DOI Zenodo, README

## [0.1.2] - 2026-09-15

### Fixed
- `nufftcf.__version__` was a string hardcoded in `src/nufftcf/__init__.py`,
  independent from the `version` field in `pyproject.toml` (the one actually
  used to build the PyPI release). It had drifted after the 0.1.1 release
  (still reported `0.1.0`). `__version__` is now read dynamically from the
  installed package's metadata (`importlib.metadata.version("nufftcf")`),
  so it can no longer go out of sync with `pyproject.toml`.

## [0.1.1] - 2026-09-15

### Added
- `compute_ccf_rectangle_fft` and `compute_ccf_gaussian_fft` in a new
  `fft_ccf.py` module: FFT-based cross-correlation (CCF) estimators for two
  REGULARLY-sampled series sharing the same sampling step and a common
  integer sampling lattice. These are the CCF counterparts of the existing
  `compute_acf_rectangle_fft` / `compute_acf_gaussian_fft` (`fft_acf.py`),
  completing the FFT estimator family (which previously covered ACF only).
  Same `gaussian`/`rectangle` kernel definitions, and same lag-0
  normalisation approach, as `compute_ccf_*_nufft` / `compute_ccf_*_realspace`;
  on a shared regular grid, results match `compute_ccf_*_realspace` almost
  exactly (no NUFFT approximation error involved), and are computed via
  `scipy.signal.correlate` + `scipy.ndimage` filtering instead of NUFFT or a
  numba two-pointer scan.
- `tests/test_fft_ccf.py`: 28 new tests covering shape/sanity checks, grid-
  compatibility error handling (mismatched `dt`, non-lattice offset,
  irregular grid), agreement with `compute_ccf_*_realspace`, agreement with
  `compute_acf_*_fft` when both signals are identical, lag-shift detection,
  boundedness for independent signals, and a multi-signal smoke test.

### Fixed
- `compute_b_gaussian_cross`, `compute_b_rectangle_cross`,
  `compute_c_gaussian_cross`, and `compute_c_rectangle_cross` (in
  `kernels.py`) were already listed in `nufftcf.__all__` but never actually
  imported into the package namespace in `src/nufftcf/__init__.py`, so
  `from nufftcf import compute_b_gaussian_cross` (and the other three)
  raised an `ImportError`. They are now imported and exported correctly.

## [0.1.0] - Initial release

- First public release: NUFFT- and real-space (numba)-based ACF and CCF
  estimators (`gaussian` and `rectangle` kernels) for irregularly-sampled
  time series, plus a fast FFT-based ACF path (`regular`, `rectangle`,
  `gaussian`) for regularly-sampled data.

[0.2.1]: https://github.com/jecampagne/nufftcf/releases/tag/v0.2.1
[0.2.0]: https://github.com/jecampagne/nufftcf/releases/tag/v0.2.0
[0.1.4]: https://github.com/jecampagne/nufftcf/releases/tag/v0.1.4
[0.1.2]: https://github.com/jecampagne/nufftcf/releases/tag/v0.1.2
[0.1.1]: https://github.com/jecampagne/nufftcf/releases/tag/v0.1.1
[0.1.0]: https://github.com/jecampagne/nufftcf/releases/tag/v0.1.0
