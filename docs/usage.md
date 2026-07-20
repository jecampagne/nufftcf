# Usage

## Choosing the right estimator

| Your data | Recommended function | Notes |
|---|---|:---:|
| Irregularly sampled, long ($\gtrsim~10,000$ pts) | `compute_acf_gaussian_nufft` / `compute_acf_rectangle_nufft` | $O(n\log n)$ fastest |
| Irregularly sampled, any length | `compute_acf_gaussian_realspace` / `compute_acf_rectangle_realspace` | $O(n)$ per lag, artifact-free |
| **Regularly sampled**, any method | `compute_acf_gaussian_fft` / `compute_acf_rectangle_fft` / `compute_acf_regular_fft` | 100–400× faster than NUFFT on regular grids |
| **Cross-correlation**, long ($\gtrsim~10,000$ pts) | `compute_ccf_gaussian_nufft` / `compute_ccf_rectangle_nufft` | $O(n\log n)$ fastest |
| **Cross-correlation**, any length | `compute_ccf_gaussian_realspace` / `compute_ccf_rectangle_realspace` | $O(n)$ per lag, artifact-free |

All ACF functions share the same signature:

```python
c, b = fn(lags, t, x)                     # no-kernel ("regular") variant
c, b = fn(lags, t, x, bin_width=0.5)      # gaussian / rectangle variants
```

- `lags` — array of lag values (same units as `t`)
- `t` — sample times, sorted ascending (float array, e.g. days since start)
- `x` — signal values, same length as `t`
- `bin_width` — kernel half-width (gaussian σ or rectangle half-width), same units as `t`
- Returns `(c, b)` — ACF estimate and effective pair count, both shape `(len(lags),)`

The CCF functions take a second `(s, y)` series in addition to `(t, x)`:

```python
c, b = fn(lags, t, x, s, y, bin_width=0.5)
```

- `s` — sample times of the second series, sorted ascending, on the **same time
  origin as `t`** (see the warning below)
- `y` — signal values of the second series, same length as `s`
- Returns `(c, b)` — CCF estimate (Pearson-normalised, `c ~ 1` at perfect correlation)
  and effective pair count, both shape `(len(lags),)`. By convention a positive lag
  means `y` lags behind `x` (the CCF peaks at `lag = tau0` when `y(t) ~ x(t - tau0)`)

## Irregularly-sampled series (NUFFT path)

```python
import numpy as np
import pandas as pd
from nufftcf import compute_acf_gaussian_nufft, t_numeric_of

# A pandas Series with a DatetimeIndex works directly
idx = pd.date_range("2005-01-01", periods=3650, freq="D")
# simulate some irregular gaps
mask = np.random.default_rng(0).random(3650) > 0.3
sts = pd.Series(np.random.randn(3650)[mask], index=idx[mask])

lags = np.arange(1.0, 366.0)          # lags 1..365 days
t    = t_numeric_of(sts)               # timedelta -> float days
x    = sts.to_numpy()

c, b = compute_acf_gaussian_nufft(lags, t, x, bin_width=0.5)
```

## Regularly-sampled series (classic FFT path)

```python
from nufftcf import compute_acf_gaussian_fft, compute_acf_regular_fft

n = 3650
t = np.arange(n, dtype=float)         # integer day indices
x = np.random.default_rng(0).standard_normal(n)
lags = np.arange(0.0, 366.0)

# With Gaussian smoothing kernel
c_gauss, b = compute_acf_gaussian_fft(lags, t, x, bin_width=0.5)

# Without smoothing kernel (matches Pastas' bin_method="regular" exactly)
c_reg, b = compute_acf_regular_fft(lags, t, x)
```

!!! warning
    The `_fft` estimators raise `ValueError` if `t` is not regularly spaced.
    Use the `_nufft` or `_realspace` estimators for irregular data.

## Cross-correlation between two irregularly-sampled series

The CCF estimators (`compute_ccf_gaussian_nufft`, `compute_ccf_rectangle_nufft`,
`compute_ccf_gaussian_realspace`, `compute_ccf_rectangle_realspace`) compare two
irregularly-sampled series `(t, x)` and `(s, y)`, which may have different lengths
and different sampling times.

!!! warning
    `t` and `s` must be expressed on a **common** time origin (e.g. elapsed days
    since the same reference date for both series). `t_numeric_of` alone uses each
    series' own first sample as origin, which is **not** suitable here — applying
    it separately to `x` and `y` would silently misalign the lags. Build `t`/`s`
    from a shared reference date instead, as in the example below.

```python
import numpy as np
import pandas as pd
from nufftcf import compute_ccf_gaussian_nufft

# Two irregularly-sampled series on a COMMON time origin (elapsed days since
# the same reference date), with y lagging behind x by tau0 = 60 days
ref_date = pd.Timestamp("2000-01-01")
n_days, tau0, alpha = 3650, 60, 10.0
rng = np.random.default_rng(0)

# shared latent Ornstein-Uhlenbeck-like signal (ACF ~ exp(-|u|/alpha))
phi = np.exp(-1.0 / alpha)
noise = rng.standard_normal(n_days + tau0)
z = np.empty(n_days + tau0)
z[0] = noise[0]
for i in range(1, n_days + tau0):
    z[i] = phi * z[i - 1] + noise[i]

mask_x = rng.random(n_days) > 0.6   # series 1: ~40% of days kept
mask_y = rng.random(n_days) > 0.4   # series 2: ~60% of days kept

t = np.arange(n_days)[mask_x].astype(float)   # elapsed days, series 1
s = np.arange(n_days)[mask_y].astype(float)   # elapsed days, series 2 (same origin as t)

x = z[tau0:][mask_x]     # x(t)   = z(t)
y = z[:n_days][mask_y]   # y(t)   = z(t - tau0)  -> y lags x by tau0 days

lags = np.arange(1.0, 181.0)   # 1 to 180 days
c, b = compute_ccf_gaussian_nufft(lags, t, x, s, y, bin_width=0.5)
# c: CCF estimate per lag, peaks at lag = tau0 = 60
# b: effective number of contributing pairs per lag
```

For a worked comparison against **pyZDCF**, including a case with a known
theoretical CCF, see
[`notebook/nufftcf_ccf_demo.ipynb`](https://github.com/jecampagne/nufftcf/blob/main/notebook/nufftcf_ccf_demo.ipynb).

## Notebooks

Colab-ready notebooks are included in `notebook/`:

- [`pastas_vs_nufftcf.ipynb`](https://github.com/jecampagne/nufftcf/blob/main/notebook/pastas_vs_nufftcf.ipynb) —
  **irregularly-sampled** series (sine and AR(1)-like, with gaps), NUFFT path vs Pastas.
- [`pastas_vs_nufftcf_regular.ipynb`](https://github.com/jecampagne/nufftcf/blob/main/notebook/pastas_vs_nufftcf_regular.ipynb) —
  **regularly-sampled** series (sine, noisy sine, exponential, constant, square wave),
  classic-FFT path vs Pastas for all three bin methods.
- [`zdcf_vs_nufftcf.ipynb`](https://github.com/jecampagne/nufftcf/blob/main/notebook/zdcf_vs_nufftcf.ipynb) —
  **nufftcf** vs **pyZDCF** on the same irregularly-sampled series as above.
- [`nufftcf_ccf_demo.ipynb`](https://github.com/jecampagne/nufftcf/blob/main/notebook/nufftcf_ccf_demo.ipynb) —
  the **cross-correlation (CCF)** functions vs **pyZDCF**, including a case with two
  coupled Ornstein-Uhlenbeck series for which the theoretical CCF is known analytically.

All install `nufftcf` directly from GitHub in their first cell — no local setup needed
to run them on Colab.

## Benchmark scripts

```bash
pip install -e ".[benchmark]"

# Irregular data: Pastas vs _nufft
python benchmark/benchmark_acf.py
python benchmark/fit_benchmark_acf.py

# Regular data: Pastas vs _fft vs _nufft, all three bin methods
python benchmark/benchmark_acf_regular.py
python benchmark/fit_benchmark_acf_regular.py 
```

Example results (Apple Silicon Mac) are stored in `benchmark/`.

No dedicated benchmark script is provided for the CCF estimators against
**pyZDCF**, but the plots in `nufftcf_ccf_demo.ipynb` show `nufftcf` running
roughly two orders of magnitude faster.
