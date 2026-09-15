"""
nufftcf: fast ACF/CCF estimation for irregularly- AND regularly-sampled time series.

Three estimation families are provided, sharing the same (lags, t, x, bin_width) for ACF
and (lags, t, x, s, y, bin_width) for CCF calling convention:

- `compute_[acf/ccf]_*_nufft`      : NUFFT + Wiener-Khinchin, O(n log n)-ish, fastest
                                for long IRREGULAR series, ~1-3% residual
                                amplitude bias on strongly periodic signals
                                (see module docs).
- `compute_[acf/ccf]_*_realspace`  : direct real-space weighted sum, O(n) per lag,
                                artifact-free reference / alternative, works
                                for irregular AND regular sampling.
- `compute_[acf/ccf]_*_fft`        : classic FFT correlation, O(n log n), for
                                REGULARLY-sampled data only -- faster than
                                `_nufft` (no NUFFT overhead) and faster than
                                `_realspace` (no numba two-pointer scan) when
                                sampling happens to be regular. Also adds a
                                `regular` (no-kernel) variant matching
                                Pastas `bin_method="regular"` for ACF.

All families come in `gaussian` and `rectangle` kernel variants.
"""

from .kernels import (
    compute_b_gaussian,
    compute_b_rectangle,
    compute_c_gaussian,
    compute_c_rectangle,
    compute_b_gaussian_cross,
    compute_b_rectangle_cross,
    compute_c_gaussian_cross,
    compute_c_rectangle_cross,
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
from .fft_ccf import (
    compute_ccf_rectangle_fft,
    compute_ccf_gaussian_fft,
)
from .utils import t_numeric_of, standardize

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("nufftcf")
except _metadata.PackageNotFoundError:  # e.g. running from an uninstalled checkout
    __version__ = "unknown"

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
    "compute_ccf_rectangle_fft",
    "compute_ccf_gaussian_fft",
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
