# Changelog

All notable changes to `nufftcf` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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

[0.1.4]: https://github.com/jecampagne/nufftcf/releases/tag/v0.1.4
[0.1.2]: https://github.com/jecampagne/nufftcf/releases/tag/v0.1.2
[0.1.1]: https://github.com/jecampagne/nufftcf/releases/tag/v0.1.1
[0.1.0]: https://github.com/jecampagne/nufftcf/releases/tag/v0.1.0
