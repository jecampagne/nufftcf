# API Reference

All public functions are importable directly from `nufftcf`:

```python
from nufftcf import compute_acf_gaussian_nufft, compute_acf_regular_fft, ...
```

---

## NUFFT estimators

### Autocorrelation (ACF)

::: nufftcf.nufft_acf
    options:
      show_root_heading: true
      show_signature: true
      show_signature_annotations: true
      separate_signature: true
      members:
        - compute_acf_gaussian_nufft
        - compute_acf_rectangle_nufft

### Cross-correlation (CCF)

::: nufftcf.nufft_ccf
    options:
      show_root_heading: true
      show_signature: true
      show_signature_annotations: true
      separate_signature: true
      members:
        - compute_ccf_gaussian_nufft
        - compute_ccf_rectangle_nufft

## Classic-FFT estimators (regular data only)

::: nufftcf.fft_acf
    options:
      show_root_heading: true
      show_signature: true
      show_signature_annotations: true
      separate_signature: true
      members:
        - compute_acf_regular_fft
        - compute_acf_rectangle_fft
        - compute_acf_gaussian_fft

## Real-space estimators

### Autocorrelation (ACF)

::: nufftcf.realspace_acf
    options:
      show_root_heading: true
      show_signature: true
      show_signature_annotations: true
      separate_signature: true
      members:
        - compute_acf_gaussian_realspace
        - compute_acf_rectangle_realspace

### Cross-correlation (CCF)

::: nufftcf.realspace_ccf
    options:
      show_root_heading: true
      show_signature: true
      show_signature_annotations: true
      separate_signature: true
      members:
        - compute_ccf_gaussian_realspace
        - compute_ccf_rectangle_realspace
---

## Kernel helpers
::: nufftcf.kernels
    options:
      show_root_heading: true
      show_signature: true
      show_signature_annotations: true
      separate_signature: true
      members:
        - compute_b_gaussian
        - compute_b_rectangle
        - compute_c_gaussian
        - compute_c_rectangle

---

## Utilities
::: nufftcf.utils
    options:
      show_root_heading: true
      show_signature: true
      show_signature_annotations: true
      separate_signature: true
      members:
        - t_numeric_of
        - standardize
