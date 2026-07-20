"""
Benchmark (data collection only): time Pastas vs the nufftcf package on
irregularly-sampled series of variable length, for both the gaussian and
rectangle kernels.

This script ONLY collects timing data and writes it to a CSV (long format:
one row per individual repeat). Fitting and plotting are done separately by
fit_benchmark_acf.py, which reads that CSV.

Design choices, informed by earlier runs on a laptop under heavy thermal
load (heatwave!) that showed a one-off, non-reproducible timing spike at a
single (method, n_years) point:

  - The execution ORDER of (method, n_years) combinations is randomized.
    A systematic thermal drift over the course of a long run would otherwise
    be confounded with the n-trend itself (the very thing we're trying to
    measure) if always run in increasing n order.
  - Pastas gets more repeats (default 6) than the NUFFT-based estimator
    (default 3), since its longer per-call duration makes it more exposed
    to transient external interference (OS scheduling, thermal throttling).
  - ALL individual repeat timings are kept in the output (not just the
    min), so the downstream fitting script can bootstrap over the actual
    observed measurement noise rather than assuming a noise model.

Requires: pip install -e ".[benchmark]"  (from the package root)

Usage: python benchmark/benchmark_acf.py [--output results.csv]
"""

import argparse
import time
import numpy as np
import pandas as pd
import pastas as ps

from nufftcf import (
    compute_acf_gaussian_nufft,
    compute_acf_rectangle_nufft,
    t_numeric_of,
)

NUFFT_FUNCS = {
    "gaussian": compute_acf_gaussian_nufft,
    "rectangle": compute_acf_rectangle_nufft,
}


# ============================================================
# 1. Random irregularly-sampled series generator
# ============================================================
def generate_irregular_series(
    n_years: float, drop_fraction: float = 0.3, seed: int = 0
) -> pd.Series:
    """Generate a white-noise series spanning n_years, with irregular sampling
    (a random fraction `drop_fraction` of days is removed)."""
    rng = np.random.default_rng(seed)
    n_days = int(n_years * 365)
    index_full = pd.to_datetime(np.arange(n_days), unit="D", origin="2000")
    keep_mask = rng.random(n_days) > drop_fraction
    index = index_full[keep_mask]
    values = rng.standard_normal(len(index))
    return pd.Series(values, index=index, name=f"rand_{n_years}y")


# ============================================================
# 2. Single-call timers (return ONE elapsed time; repeats are handled by
#    the caller so every individual repeat can be recorded)
# ============================================================
def time_nufft_once(series, lags, method="gaussian", bin_width=0.5):
    t = t_numeric_of(series)
    x = series.to_numpy()
    func = NUFFT_FUNCS[method]
    t0 = time.perf_counter()
    func(lags, t, x, bin_width=bin_width)
    return time.perf_counter() - t0


def time_pastas_once(series, lags, method="gaussian", bin_width=0.5, max_gap=30):
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
    durations_years,
    methods=("gaussian", "rectangle"),
    pastas_max_years=32,
    drop_fraction=0.3,
    seed=42,
    n_repeat_pastas=6,
    n_repeat_nufft=3,
    order_seed=None,
):
    lags = np.arange(0.0, 366.0)
    rows = []

    # JIT warm-up (excluded from recorded timings) for each method
    warmup_series = generate_irregular_series(1, drop_fraction, seed)
    for method in methods:
        time_nufft_once(warmup_series, lags, method=method)
        time_pastas_once(warmup_series, lags, method=method)

    # Build the full list of (method, n_years) combinations, then shuffle
    # the execution order -- avoids confounding a systematic thermal drift
    # over the course of the run with the n-trend being measured.
    combos = [(method, ny) for method in methods for ny in durations_years]
    rng_order = np.random.default_rng(order_seed)
    rng_order.shuffle(combos)

    run_order = 0
    for method, ny in combos:
        series = generate_irregular_series(ny, drop_fraction=drop_fraction, seed=seed)
        n = len(series)

        for rep in range(n_repeat_nufft):
            run_order += 1
            dt = time_nufft_once(series, lags, method=method)
            rows.append(
                dict(
                    run_order=run_order,
                    method=method,
                    n_years=ny,
                    n_points=n,
                    algo="nufft",
                    repeat=rep,
                    time_s=dt,
                )
            )

        if ny <= pastas_max_years:
            for rep in range(n_repeat_pastas):
                run_order += 1
                dt = time_pastas_once(series, lags, method=method)
                rows.append(
                    dict(
                        run_order=run_order,
                        method=method,
                        n_years=ny,
                        n_points=n,
                        algo="pastas",
                        repeat=rep,
                        time_s=dt,
                    )
                )

        n_min_nufft = min(
            r["time_s"]
            for r in rows
            if r["method"] == method and r["n_years"] == ny and r["algo"] == "nufft"
        )
        pastas_note = ""
        if ny <= pastas_max_years:
            n_min_pastas = min(
                r["time_s"]
                for r in rows
                if r["method"] == method
                and r["n_years"] == ny
                and r["algo"] == "pastas"
            )
            pastas_note = f"t_pastas(min)={n_min_pastas:8.4f}s"
        else:
            pastas_note = "t_pastas=skip"
        print(
            f"method={method:9s}  n_years={ny:6.0f}  n_points={n:7d}  "
            f"{pastas_note}  t_nufft(min)={n_min_nufft:8.4f}s"
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect ACF benchmark timing data (Pastas vs nufftcf)."
    )
    parser.add_argument(
        "--output",
        default="benchmark_acf_results.csv",
        help="Output CSV path (long format: one row per repeat).",
    )
    parser.add_argument(
        "--pastas-max-years",
        type=float,
        default=32,
        help="Skip Pastas above this series length (its O(n^2) scaling makes long series impractical).",
    )
    parser.add_argument(
        "--n-repeat-pastas",
        type=int,
        default=6,
        help="Number of repeats per point for Pastas (more, since it's slower and more exposed to transient interference).",
    )
    parser.add_argument(
        "--n-repeat-nufft",
        type=int,
        default=3,
        help="Number of repeats per point for the nufftcf package.",
    )
    parser.add_argument(
        "--order-seed",
        type=int,
        default=None,
        help="Seed for randomizing run order (None = different shuffle every run).",
    )
    args = parser.parse_args()

    durations_years = [1, 2, 4, 8, 16, 32, 64, 100]
    df = run_benchmark(
        durations_years,
        methods=("gaussian", "rectangle"),
        pastas_max_years=args.pastas_max_years,
        n_repeat_pastas=args.n_repeat_pastas,
        n_repeat_nufft=args.n_repeat_nufft,
        order_seed=args.order_seed,
    )
    df.to_csv(args.output, index=False)
    print(f"\nRaw timing data ({len(df)} rows) saved to: {args.output}")
    print("Run fit_benchmark_acf.py on this file to fit and plot the results.")
