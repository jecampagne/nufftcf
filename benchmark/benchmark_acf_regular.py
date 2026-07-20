"""
Benchmark (data collection only): time Pastas vs nufftcf on REGULARLY
-sampled series of variable length, for three bin methods: "regular" (no
smoothing kernel), "rectangle", and "gaussian".

Unlike the irregular-sampling benchmark (`benchmark_acf.py`), this one
also includes the existing `_nufft` estimators (gaussian/rectangle) for
comparison, alongside the new dedicated `_fft` (classic-FFT) path -- on
regular data, NUFFT is a strict generalization of what's needed, so this
shows what (if anything) the dedicated regular-grid fast path buys over
just reusing the more general NUFFT estimator. There is no NUFFT
counterpart for "regular" (no-kernel), so that row only compares Pastas vs
`_fft`.

Pastas' "regular" bin_method is empirically ~O(n) (it's a windowed
np.corrcoef per lag, with a FIXED number of lags) and stays fast even for
large series, but its "gaussian"/"rectangle" bin methods are empirically
O(n^2) (same conclusion as the irregular-sampling case) and become
impractically slow well before n=20'000 -- confirmed by direct
measurement before writing this script, not assumed. Pastas is therefore
capped at a much higher n for "regular" than for "gaussian"/"rectangle"
(see `--pastas-max-n-regular` / `--pastas-max-n-kernel`).

Requires: pip install -e ".[benchmark]"  (from the package root)

Usage: python benchmark/benchmark_acf_regular.py [--output results.csv]
"""

import argparse
import time
import numpy as np
import pandas as pd
import pastas as ps

from nufftcf import (
    compute_acf_regular_fft,
    compute_acf_rectangle_fft,
    compute_acf_gaussian_fft,
    compute_acf_rectangle_nufft,
    compute_acf_gaussian_nufft,
)

FFT_FUNCS = {
    "regular": compute_acf_regular_fft,
    "rectangle": compute_acf_rectangle_fft,
    "gaussian": compute_acf_gaussian_fft,
}
NUFFT_FUNCS = {
    "rectangle": compute_acf_rectangle_nufft,
    "gaussian": compute_acf_gaussian_nufft,
}


# ============================================================
# 1. Regular series generator
# ============================================================
def generate_regular_series(n_points: int, seed: int = 0) -> pd.Series:
    """White-noise series on a regular numeric grid (delta_t=1). Kept as a
    plain numeric index (not a DatetimeIndex) so this works at any n_points
    -- a real DatetimeIndex starting in year 2000 overflows pandas' range
    well before n_points=130'000 (~292 years). The DatetimeIndex Pastas
    needs is built on demand, only for the n_points actually small enough
    to be sent to Pastas (see `time_pastas_once`)."""
    rng = np.random.default_rng(seed)
    idx = pd.Index(np.arange(n_points, dtype=float), name="t")
    return pd.Series(rng.standard_normal(n_points), index=idx, name=f"rand_{n_points}")


def _as_pastas_series(series: pd.Series) -> pd.Series:
    """Re-index a numeric-indexed series onto a DatetimeIndex (daily),
    required by Pastas. Only ever called for n_points small enough to stay
    well within pandas' datetime range (caller enforces the cap)."""
    idx = pd.date_range("2000-01-01", periods=len(series), freq="D")
    return pd.Series(series.to_numpy(), index=idx, name=series.name)


# ============================================================
# 2. Single-call timers
# ============================================================
def time_fft_once(series, lags, method="gaussian", bin_width=0.5):
    t = series.index.to_numpy(dtype=float)
    x = series.to_numpy()
    func = FFT_FUNCS[method]
    t0 = time.perf_counter()
    if method == "regular":
        func(lags, t, x)
    else:
        func(lags, t, x, bin_width=bin_width)
    return time.perf_counter() - t0


def time_nufft_once(series, lags, method="gaussian", bin_width=0.5):
    t = series.index.to_numpy(dtype=float)
    x = series.to_numpy()
    func = NUFFT_FUNCS[method]
    t0 = time.perf_counter()
    func(lags, t, x, bin_width=bin_width)
    return time.perf_counter() - t0


def time_pastas_once(series, lags, method="regular", bin_width=0.5, max_gap=30):
    series = _as_pastas_series(series)
    t0 = time.perf_counter()
    ps.stats.acf(
        series,
        lags=lags,
        bin_method=method,
        bin_width=bin_width,
        max_gap=max_gap,
        min_obs=0,
    )
    return time.perf_counter() - t0


# ============================================================
# 3. Main benchmark loop
# ============================================================
def run_benchmark(
    n_points_list,
    methods=("regular", "rectangle", "gaussian"),
    pastas_max_n_regular=50_000,
    pastas_max_n_kernel=8_000,
    seed=42,
    n_repeat_pastas=6,
    n_repeat_fast=3,
    order_seed=None,
):
    lags = np.arange(0.0, 366.0)
    rows = []

    # Warm-up (FFT planning / finufft plan cache), excluded from timings.
    warmup_series = generate_regular_series(2000, seed)
    for method in methods:
        time_fft_once(warmup_series, lags, method=method)
        if method != "regular":
            time_nufft_once(warmup_series, lags, method=method)
        time_pastas_once(warmup_series, lags, method=method)

    combos = [(method, n) for method in methods for n in n_points_list]
    rng_order = np.random.default_rng(order_seed)
    rng_order.shuffle(combos)

    run_order = 0
    for method, n_points in combos:
        series = generate_regular_series(n_points, seed=seed)
        n = len(series)
        pastas_max_n = (
            pastas_max_n_regular if method == "regular" else pastas_max_n_kernel
        )

        def _record(algo, dt, rep):
            nonlocal run_order
            run_order += 1
            rows.append(
                dict(
                    run_order=run_order,
                    method=method,
                    n_points=n,
                    algo=algo,
                    repeat=rep,
                    time_s=dt,
                )
            )

        for rep in range(n_repeat_fast):
            _record("fft", time_fft_once(series, lags, method=method), rep)
            if method != "regular":
                _record("nufft", time_nufft_once(series, lags, method=method), rep)

        if n <= pastas_max_n:
            for rep in range(n_repeat_pastas):
                _record("pastas", time_pastas_once(series, lags, method=method), rep)

        def _min_of(algo):
            vals = [
                r["time_s"]
                for r in rows
                if r["method"] == method and r["n_points"] == n and r["algo"] == algo
            ]
            return min(vals) if vals else None

        parts = [f"method={method:9s} n_points={n:8d}", f"fft={_min_of('fft'):.4f}s"]
        if method != "regular":
            parts.append(f"nufft={_min_of('nufft'):.4f}s")
        p_min = _min_of("pastas")
        parts.append(f"pastas={p_min:.4f}s" if p_min is not None else "pastas=skip")
        print("  ".join(parts))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect ACF benchmark timing data on REGULAR data (Pastas vs nufftcf)."
    )
    parser.add_argument("--output", default="benchmark_acf_regular_results.csv")
    parser.add_argument(
        "--pastas-max-n-regular",
        type=int,
        default=50_000,
        help="Skip Pastas 'regular' above this n (it's ~O(n), stays usable much longer).",
    )
    parser.add_argument(
        "--pastas-max-n-kernel",
        type=int,
        default=8_000,
        help="Skip Pastas 'gaussian'/'rectangle' above this n (O(n^2): ~48s already at n=16'000).",
    )
    parser.add_argument("--n-repeat-pastas", type=int, default=6)
    parser.add_argument("--n-repeat-fast", type=int, default=6)
    parser.add_argument("--order-seed", type=int, default=None)
    args = parser.parse_args()

    n_points_list = [500, 1000, 2000, 4000, 8000, 16000, 32000, 65000, 130000, 260000]
    df = run_benchmark(
        n_points_list,
        pastas_max_n_regular=args.pastas_max_n_regular,
        pastas_max_n_kernel=args.pastas_max_n_kernel,
        n_repeat_pastas=args.n_repeat_pastas,
        n_repeat_fast=args.n_repeat_fast,
        order_seed=args.order_seed,
    )
    df.to_csv(args.output, index=False)
    print(f"\nRaw timing data ({len(df)} rows) saved to: {args.output}")
    print("Run fit_benchmark_acf_regular.py on this file to fit and plot the results.")
