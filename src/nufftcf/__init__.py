"""
nufftcf: fast ACF estimation for irregularly- AND regularly-sampled time series.

Three estimation families are provided, sharing the same (lags, t, x, bin_width)
calling convention:

- `compute_acf_*_nufft`      : NUFFT + Wiener-Khinchin, O(n log n)-ish, fastest
                                for long IRREGULAR series, ~1-3% residual
                                amplitude bias on strongly periodic signals
                                (see module docs).
- `compute_acf_*_realspace`  : direct real-space weighted sum, O(n) per lag,
                                artifact-free reference / alternative, works
                                for irregular AND regular sampling.
- `compute_acf_*_fft`        : classic FFT correlation, O(n log n), for
                                REGULARLY-sampled data only -- faster than
                                `_nufft` (no NUFFT overhead) and faster than
                                `_realspace` (no numba two-pointer scan) when
                                sampling happens to be regular. Also adds a
                                `regular` (no-kernel) variant matching
                                Pastas' `bin_method="regular"`.

All families come in `gaussian` and `rectangle` kernel variants; `_fft` also
has `regular` (no smoothing kernel).

Example
-------
>>> import numpy as np, pandas as pd
>>> from nufftcf import compute_acf_gaussian_nufft, t_numeric_of
>>> idx = pd.date_range("2020-01-01", periods=2000, freq="D")
>>> x = pd.Series(np.random.randn(2000), index=idx)
>>> lags = np.arange(0.0, 366.0)
>>> t = t_numeric_of(x)
>>> c, b = compute_acf_gaussian_nufft(lags, t, x.to_numpy(), bin_width=0.5)

>>> # regularly-sampled data -> use the faster classic-FFT path instead:
>>> from nufftcf import compute_acf_gaussian_fft, compute_acf_regular_fft
>>> c, b = compute_acf_gaussian_fft(lags, t, x.to_numpy(), bin_width=0.5)
>>> c, b = compute_acf_regular_fft(lags, t, x.to_numpy())  # no kernel
"""

from .kernels import (
    compute_b_gaussian,
    compute_b_rectangle,
    compute_c_gaussian,
    compute_c_rectangle,
)
from .nufft_acf import compute_acf_gaussian_nufft, compute_acf_rectangle_nufft
from .realspace_acf import (
    compute_acf_gaussian_realspace,
    compute_acf_rectangle_realspace,
)
from .nufft_ccf import compute_ccf_gaussian_nufft, compute_ccf_rectangle_nufft
from .realspace_ccf import (
    compute_ccf_gaussian_realspace,
    compute_ccf_rectangle_realspace,
)

from .fft_acf import (
    compute_acf_regular_fft,
    compute_acf_rectangle_fft,
    compute_acf_gaussian_fft,
)
from .utils import t_numeric_of, standardize

__version__ = "0.1.0"

__all__ = [
    "compute_acf_gaussian_nufft",
    "compute_acf_rectangle_nufft",
    "compute_acf_gaussian_realspace",
    "compute_acf_rectangle_realspace",
    "compute_acf_regular_fft",
    "compute_acf_rectangle_fft",
    "compute_acf_gaussian_fft",
    "compute_ccf_gaussian_nufft",
    "compute_ccf_rectangle_nufft",
    "compute_ccf_gaussian_realspace",
    "compute_ccf_rectangle_realspace",
    "compute_b_gaussian",
    "compute_b_rectangle",
    "compute_c_gaussian",
    "compute_c_rectangle",
    "compute_b_gaussian_cross",
    "compute_b_rectangle_cross",
    "compute_c_gaussian_cross",
    "compute_c_rectangle_cross",
    "t_numeric_of",
    "standardize",
]
