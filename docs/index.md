# nufftcf

[![Tests](https://github.com/jecampagne/nufftcf/actions/workflows/tests.yml/badge.svg)](https://github.com/jecampagne/nufftcf/actions/workflows/tests.yml)
[![Lint](https://github.com/jecampagne/nufftcf/actions/workflows/lint.yml/badge.svg)](https://github.com/jecampagne/nufftcf/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/jecampagne/nufftcf/blob/main/LICENSE)

[![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13%20|%203.14-blue)](https://github.com/jecampagne/nufftcf/actions/workflows/tests.yml)
[![Linux](https://img.shields.io/badge/linux-ubuntu--latest-orange?logo=linux)](https://github.com/jecampagne/nufftcf/actions/workflows/tests.yml)
[![macOS](https://img.shields.io/badge/macOS-latest%20(arm64)%20%7C%20intel-orange?logo=apple)](https://github.com/jecampagne/nufftcf/actions/workflows/tests.yml)
[![macOS](https://img.shields.io/badge/macOS-arm64%20%7C%20x86__64-orange?logo=apple)](https://github.com/jecampagne/nufftcf/actions/workflows/tests.yml)
[![Windows](https://img.shields.io/badge/windows-latest-orange?logo=windows)](https://github.com/jecampagne/nufftcf/actions/workflows/tests.yml)

Fast **autocorrelation** (ACF) and **cross-correlation** (CCF) function estimation for
**irregularly- and regularly-sampled** 1D time series, scaling as $\sim~O(n\log n)$.

## Three ACF estimator families

  Function(s) | Sampling | Method | Scaling |
 |:-----------|:---------:|:-------:|:-------:|
 | `compute_acf_gaussian_nufft` / `_rectangle_nufft` | irregular | NUFFT + Wiener-Khinchin | $\sim~O(n\log n)$ |
 | `compute_acf_gaussian_realspace` / `_rectangle_realspace` | irregular or regular | direct weighted sum | $O(n)$ per lag |
 | `compute_acf_gaussian_fft` / `_rectangle_fft` / `_regular_fft` | **regular only** | classic FFT + filter | $\sim~O(n\log n)$ |

All seven ACF functions share the same calling convention: `fn(lags, t, x, bin_width=0.5)`
(`compute_acf_regular_fft` has no `bin_width`, since it applies no smoothing kernel),
returning `(c, b)` — the ACF estimate and effective pair count.

## Cross-correlation (CCF) estimators

The same NUFFT + Wiener-Khinchin and real-space machinery is also available for the
**cross-correlation function** between two **irregularly-sampled** series `(t, x)` and
`(s, y)`, which may have different lengths and different sampling times:

  Function | Sampling | Method | Scaling |
 |:--------|:---------:|:-------:|:-------:|
 | `compute_ccf_gaussian_nufft` / `_rectangle_nufft` | irregular | NUFFT + Wiener-Khinchin | $\sim~O(n\log n)$ |
 | `compute_ccf_gaussian_realspace` / `_rectangle_realspace` | irregular or regular | direct weighted sum | $O(n)$ per lag |

All four share the calling convention `fn(lags, t, x, s, y, bin_width=0.5)` and return
`(c, b)` — the CCF estimate (Pearson-normalised, `c ~ 1` at perfect correlation) and the
effective pair count. By convention, a positive lag means `y` lags behind `x` (the CCF
peaks at `lag = tau0` when `y(t) ~ x(t - tau0)`).

**Important:** `t` and `s` must share a *common* time origin (e.g. elapsed days since the
same reference date for both series) — using `t_numeric_of` independently on each series
would silently misalign the lags. See [Usage](usage.md) for a worked example.

## Documentation

- [Installation](installation.md)
- [Usage](usage.md)
- [API Reference](api.md)
- [License](license.md)

## Quick start

```python
import numpy as np
from nufftcf import compute_acf_gaussian_nufft, t_numeric_of
import pandas as pd

sts = pd.Series(...)  # irregularly-sampled pandas Series with DatetimeIndex
lags = np.arange(1.0, 366.0)
c, b = compute_acf_gaussian_nufft(lags, t_numeric_of(sts), sts.to_numpy(), bin_width=0.5)
```

```python
# Regularly-sampled series: use the faster classic-FFT path
from nufftcf import compute_acf_gaussian_fft
t = np.arange(len(x), dtype=float)
c, b = compute_acf_gaussian_fft(lags, t, x, bin_width=0.5)
```

```python
# Cross-correlation between two irregularly-sampled series on a COMMON time
# origin (t and s must be elapsed time since the same reference date)
from nufftcf import compute_ccf_gaussian_nufft

lags = np.arange(1.0, 181.0)
c, b = compute_ccf_gaussian_nufft(lags, t, x, s, y, bin_width=0.5)
# c: CCF estimate per lag (peaks at lag = tau0 when y(t) ~ x(t - tau0))
```

See [Usage](usage.md) for a full worked CCF example built from two coupled
Ornstein-Uhlenbeck-like series with a known lag `tau0`.

## Method

Built on two ingredients:

1. **[FINUFFT](https://github.com/flatironinstitute/finufft)** (Flatiron Institute)
   evaluates the power spectrum of the irregularly-sampled signal via a type-1 NUFFT,
   then inverts it at the requested lags (Wiener-Khinchin theorem) — giving $\sim~O(n\log n)$
   scaling instead of the $O(n^2)$ real-space sum. The same machinery extends directly to
   the CCF between two irregularly-sampled series with different sampling times.
2. An analytical **"b" correction** for the effective pair count per lag, specific to
   the Gaussian and rectangular kernels, computed in $O(n)$ via a two-pointer scan
   (`kernels.py`) — without this, the raw NUFFT spectrum would not be correctly normalised.

On a **regular** grid, `fft_acf.py` uses plain `scipy.signal.correlate` and recovers
the same `b` denominator by smoothing the triangular "raw pair-count" ramp with the
same discrete filter as the numerator — making any discretisation artefact cancel exactly
in the ratio.

## Notebooks

- [pastas_vs_nufftcf.ipynb](https://github.com/jecampagne/nufftcf/blob/main/notebook/pastas_vs_nufftcf.ipynb) —
  **irregularly-sampled** series, NUFFT path vs Pastas (Colab-ready).
- [pastas_vs_nufftcf_regular.ipynb](https://github.com/jecampagne/nufftcf/blob/main/notebook/pastas_vs_nufftcf_regular.ipynb) —
  **regularly-sampled** series, classic-FFT path vs Pastas (Colab-ready).
- [zdcf_vs_nufftcf.ipynb](https://github.com/jecampagne/nufftcf/blob/main/notebook/zdcf_vs_nufftcf.ipynb) —
  **nufftcf** vs **pyZDCF** on the same irregularly-sampled series as above.
- [nufftcf_ccf_demo.ipynb](https://github.com/jecampagne/nufftcf/blob/main/notebook/nufftcf_ccf_demo.ipynb) —
  the **cross-correlation (CCF)** functions vs **pyZDCF**, including a case with two
  coupled Ornstein-Uhlenbeck series for which the theoretical CCF is known analytically.

## Benchmark

`benchmark/` contains timing results on an Apple Silicon Mac (irregular case)
and a Linux sandbox (regular case), along with the scripts to reproduce them.
See [Usage](usage.md) for the commands.

## Citing

If you use `nufftcf`, please also cite FINUFFT:

> A. H. Barnett, J. F. Magland, and L. af Klinteberg (2019).
> *A parallel non-uniform fast Fourier transform library based on an
> "exponential of semicircle" kernel.* SIAM J. Sci. Comput.  41(5), C479-C504. 
> [github.com/flatironinstitute/finufft](https://github.com/flatironinstitute/finufft)

> J.E Campagne (2026): *"Non Uniform FFT based Auto Correlation functions"*.  https://github.com/jecampagne/nufftcf

