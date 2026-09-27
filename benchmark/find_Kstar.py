"""
find_Kstar.py
=============

Determine the NUFFT-vs-real-space crossover K* for nufftcf, at a fixed
series length n (~8000 points), for the gaussian and rectangle kernels.

Adapted from benchmark/benchmark_acf.py: reuses the same irregular-series
generator, the same nufftcf compute_acf_* functions and calling
convention (lags, t, x, bin_width=...), and the same precautions
(execution order shuffled to avoid confounding thermal drift with the
trend being measured; JIT warm-up excluded from the timed loop; every
individual repeat kept in the output, not just the min/median, so the
crossover can be checked against the actual observed noise).

Here n (not n_years/lags) is the fixed quantity, and K (the number of
requested lags, lags = np.arange(0.0, K), same unit-spacing convention as
in benchmark_acf.py's lags = np.arange(0.0, 366)) is swept instead.

Requires: pip install -e ".[benchmark]"  (from the package root)

Usage: python benchmark/find_Kstar.py [--output Kstar_sweep_results.csv]
"""

import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nufftcf import (
    compute_acf_gaussian_nufft,
    compute_acf_rectangle_nufft,
    compute_acf_gaussian_realspace,
    compute_acf_rectangle_realspace,
    t_numeric_of,
)

NUFFT_FUNCS = {
    "gaussian": compute_acf_gaussian_nufft,
    "rectangle": compute_acf_rectangle_nufft,
}

REALSPACE_FUNCS = {
    "gaussian": compute_acf_gaussian_realspace,
    "rectangle": compute_acf_rectangle_realspace,
}


# ============================================================
# 1. Same irregular-series generator as benchmark_acf.py
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


def n_years_for_target_n(target_n: float, drop_fraction: float = 0.3) -> float:
    """n_years giving ~target_n points with the day-quantized, drop_fraction
    -thinned generator above (exact count also depends on the seed)."""
    return target_n / ((1.0 - drop_fraction) * 365.0)


# ============================================================
# 2. Single-call timer (one elapsed time per call; repeats handled by the
#    caller so every individual repeat can be recorded, as in
#    benchmark_acf.py)
# ============================================================
def time_once(func, lags, t, x, bin_width):
    t0 = time.perf_counter()
    func(lags, t, x, bin_width=bin_width)
    return time.perf_counter() - t0


# ============================================================
# 3. K-sweep at fixed n, for one kernel
# ============================================================
def sweep_K(
    method,
    K_list,
    n_target=8000,
    drop_fraction=0.3,
    bin_width=0.5,
    seed=42,
    n_repeat=7,
    order_seed=None,
):
    ny = n_years_for_target_n(n_target, drop_fraction)
    series = generate_irregular_series(ny, drop_fraction=drop_fraction, seed=seed)
    n = len(series)
    t = t_numeric_of(series)
    x = series.to_numpy()

    fn_nufft = NUFFT_FUNCS[method]
    fn_realspace = REALSPACE_FUNCS[method]

    # JIT warm-up, excluded from recorded timings.
    warm_lags = np.arange(0.0, 5.0)
    fn_nufft(warm_lags, t, x, bin_width=bin_width)
    fn_realspace(warm_lags, t, x, bin_width=bin_width)

    # Shuffle (K, algo, repeat) execution order -- avoids confounding a
    # systematic thermal drift over the course of the run with the
    # K-trend being measured, same rationale as benchmark_acf.py.
    combos = [
        (K, algo, rep)
        for K in K_list
        for algo in ("nufft", "realspace")
        for rep in range(n_repeat)
    ]
    rng_order = np.random.default_rng(order_seed)
    rng_order.shuffle(combos)

    rows = []
    run_order = 0
    for K, algo, rep in combos:
        run_order += 1
        lags = np.arange(0.0, float(K))
        func = fn_nufft if algo == "nufft" else fn_realspace
        dt = time_once(func, lags, t, x, bin_width)
        rows.append(
            dict(
                run_order=run_order,
                method=method,
                n_points=n,
                K=K,
                algo=algo,
                repeat=rep,
                time_s=dt,
            )
        )

    df = pd.DataFrame(rows)
    med = df.groupby(["K", "algo"]).time_s.median().unstack()
    for K in K_list:
        med_n, med_r = med.loc[K, "nufft"], med.loc[K, "realspace"]
        print(
            f"method={method:9s} n={n:6d} K={K:5d}  "
            f"t_nufft(median)={med_n*1e3:8.3f} ms  "
            f"t_realspace(median)={med_r*1e3:8.3f} ms  "
            f"ratio(nufft/realspace)={med_n/med_r:5.2f}"
        )
    return df


# ============================================================
# 4. Crossover K* (t_nufft == t_realspace) by log-log interpolation of the
#    median-timing ratio, around the K where it crosses 1
# ============================================================
def find_crossover(df_method):
    med = df_method.groupby(["K", "algo"]).time_s.median().unstack().sort_index()
    K = med.index.to_numpy(dtype=float)
    ratio = (med["nufft"] / med["realspace"]).to_numpy()
    log_ratio = np.log(ratio)
    for i in range(len(K) - 1):
        if log_ratio[i] == 0.0:
            return K[i]
        if (log_ratio[i] < 0.0 < log_ratio[i + 1]) or (
            log_ratio[i] > 0.0 > log_ratio[i + 1]
        ):
            x0, x1 = np.log(K[i]), np.log(K[i + 1])
            y0, y1 = log_ratio[i], log_ratio[i + 1]
            xk = x0 - y0 * (x1 - x0) / (y1 - y0)
            return float(np.exp(xk))
    return None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find the NUFFT-vs-real-space crossover K*, at fixed n, for the "
            "gaussian and rectangle kernels."
        )
    )
    parser.add_argument("--output", default="Kstar_sweep_results.csv")
    parser.add_argument(
        "--n-target", type=int, default=8000, help="Target series length n."
    )
    parser.add_argument("--drop-fraction", type=float, default=0.3)
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument(
        "--n-repeat",
        type=int,
        default=7,
        help="Repeats per (K, algo) point; both estimators are cheap enough here "
        "to use the same repeat count for both.",
    )
    parser.add_argument(
        "--order-seed",
        type=int,
        default=None,
        help="Seed for randomizing run order (None = different shuffle every run).",
    )
    args = parser.parse_args()

    # Dense enough around 300-600 to pin down K* precisely given the
    # K=549 rectangle observation; extend (e.g. up to 3000-4000) if no
    # crossover is found in this range.
    K_list = np.array(
        [
            10,
            20,
            30,
            50,
            75,
            100,
            150,
            200,
            250,
            300,
            350,
            366,
            400,
            450,
            500,
            549,
            600,
            700,
            800,
            1000,
            1200,
            1500,
            2000,
            2500,
            3000,
            3500,
            4000,
        ]
    )
    mask = K_list < args.n_target  # coherence of lags_max value wrt n_target
    K_list = K_list[mask]

    all_dfs = []
    for method in ("gaussian", "rectangle"):
        print(f"\n=== method={method} ===")
        df = sweep_K(
            method,
            K_list,
            n_target=args.n_target,
            drop_fraction=args.drop_fraction,
            bin_width=args.bin_width,
            n_repeat=args.n_repeat,
            order_seed=args.order_seed,
        )
        all_dfs.append(df)

    result = pd.concat(all_dfs, ignore_index=True)
    result.to_csv(args.output, index=False)
    print(f"\nRaw timing data ({len(result)} rows) saved to: {args.output}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, method in zip(axes, ("gaussian", "rectangle")):
        sub = result[result.method == method]
        med = sub.groupby(["K", "algo"]).time_s.median().unstack().sort_index()
        ax.plot(med.index, med["nufft"] * 1e3, "o--", label="nufft")
        ax.plot(med.index, med["realspace"] * 1e3, "s--", label="realspace")

        kstar = find_crossover(sub)
        if kstar is not None:
            ax.axvline(kstar, color="k", ls=":", lw=1)
            ax.set_title(f"{method} kernel  (K*$\\simeq${kstar:.0f})")
            print(f"K* for {method}: {kstar:.1f}")
        else:
            ax.set_title(f"{method} kernel  (no crossover in range)")
            print(f"K* for {method}: not found in sampled range -- extend K_list.")

        ax.set_xlabel("K (number of lags)")
        ax.set_ylabel("Computation time [ms]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    n_actual = result["n_points"].iloc[0]
    fig.suptitle(f"NUFFT vs real-space crossover, n\u2248{n_actual}")
    fig.tight_layout()
    f_name = "Kstar_sweep_" + str(args.n_target) + ".pdf"
    fig.savefig(f_name, format="pdf", bbox_inches="tight", dpi=200)
    print(f"Saved {f_name}")


if __name__ == "__main__":
    main()
