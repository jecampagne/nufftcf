"""
Fit and plot ACF benchmark results.

Reads the long-format CSV produced by benchmark_acf.py (one row per
individual repeat) and, for each kernel (gaussian/rectangle):
  - Pastas timings are fit to   a * n^2       + overhead
  - nufftcf timings are fit to a * n*ln(n)  + overhead

Fitting is ROBUST (scipy.optimize.least_squares with loss='soft_l1'),
which down-weights the influence of one-off outlier points (e.g. a single
measurement hit by transient thermal throttling or OS scheduling noise)
without requiring you to manually identify and drop them. Both parameters
are constrained to be non-negative (overhead < 0 is not physical, and an
unconstrained fit landing there is usually a sign of a misspecified model
rather than a meaningful negative overhead -- see discussion).

Parameter uncertainties come from a bootstrap over the actual repeats
recorded at each point (not a covariance-matrix approximation, which is not
reliable for a fit at/near a boundary or with a non-quadratic robust loss):
for each of `--n-boot` iterations, the repeats at every n are resampled
with replacement and the per-point minimum is recomputed (mirroring exactly
how the point estimate itself is built), then the model is refit; the
spread of refit parameters across iterations is the reported uncertainty.

Usage: python fit_benchmark_acf.py <data.csv> [--n-boot 500] [--output-prefix benchmark_acf]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares



import matplotlib as mpl
mpl.rcParams.update({
    'font.size'         : 13,
    'axes.titlesize'    : 14,
    'axes.titleweight'  : 'bold',
    'axes.labelsize'    : 14,
    'xtick.labelsize'   : 14,
    'ytick.labelsize'   : 14,
    'legend.fontsize'   : 12,
    'figure.titlesize'  : 16,
    'figure.titleweight': 'bold',
})


# ============================================================
# 1. Models
# ============================================================
def model_pastas(n, a, ovh):
    return a * n**2 + ovh


def model_nufft(n, a, ovh):
    return a * n * np.log(n) + ovh


MODELS = {
    "pastas": (model_pastas, f"$a.n^2 + overhead$"),
    "nufft": (model_nufft, f"$a.n.\ln(n) + overhead$"),
}


# ============================================================
# 2. Robust fit (soft_l1), with an initial linear-loss pass to pick a
#    sensible f_scale (the residual magnitude beyond which soft_l1 starts
#    down-weighting points like a robust loss rather than a plain square)
# ============================================================
def robust_fit(n_arr, t_arr, model_func, p0):
    n_arr = np.asarray(n_arr, dtype=float)
    t_arr = np.asarray(t_arr, dtype=float)
    bounds = (np.array([0.0, 0.0]), np.array([np.inf, np.inf]))

    def residuals(params):
        a, ovh = params
        return model_func(n_arr, a, ovh) - t_arr

    # Pass 1: ordinary least squares, just to get a robust scale estimate
    res0 = least_squares(residuals, p0, bounds=bounds)
    mad = np.median(np.abs(res0.fun - np.median(res0.fun))) or 1e-6
    f_scale = max(mad, 1e-6)

    # Pass 2: robust refit using that scale
    res = least_squares(
        residuals, res0.x, loss="soft_l1", f_scale=f_scale, bounds=bounds
    )
    return res.x  # [a, ovh]


def r_squared(n_arr, t_arr, model_func, params):
    pred = model_func(np.asarray(n_arr, dtype=float), *params)
    t_arr = np.asarray(t_arr, dtype=float)
    ss_res = np.sum((t_arr - pred) ** 2)
    ss_tot = np.sum((t_arr - t_arr.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ============================================================
# 3. Bootstrap over the recorded repeats (resample repeats at each n,
#    recompute the min -- exactly mirroring how the point estimate is
#    built -- then refit)
# ============================================================
def bootstrap_fit(repeats_by_n, model_func, p0, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    n_values = np.array(sorted(repeats_by_n.keys()), dtype=float)
    boot_params = []
    for _ in range(n_boot):
        t_boot = np.array(
            [
                rng.choice(
                    repeats_by_n[n], size=len(repeats_by_n[n]), replace=True
                ).min()
                for n in n_values
            ]
        )
        try:
            params = robust_fit(n_values, t_boot, model_func, p0)
            boot_params.append(params)
        except Exception:
            continue
    boot_params = np.array(boot_params)
    return boot_params.std(axis=0)


# ============================================================
# 4. Per (kernel, algo) analysis
# ============================================================
def analyze_group(df, kernel, algo, n_boot, seed):
    sub = df[(df["method"] == kernel) & (df["algo"] == algo)]
    if sub.empty:
        return None

    repeats_by_n = {n: g["time_s"].to_numpy() for n, g in sub.groupby("n_points")}
    n_values = np.array(sorted(repeats_by_n.keys()), dtype=float)
    t_min = np.array([repeats_by_n[n].min() for n in n_values])

    model_func, model_label = MODELS[algo]
    p0 = [1e-7, 0.01] if algo == "pastas" else [1e-6, 0.005]

    params = robust_fit(n_values, t_min, model_func, p0)
    a, ovh = params
    r2 = r_squared(n_values, t_min, model_func, params)
    a_err, ovh_err = bootstrap_fit(
        repeats_by_n, model_func, p0, n_boot=n_boot, seed=seed
    )

    spread = np.array([repeats_by_n[n].max() - repeats_by_n[n].min() for n in n_values])

    return dict(
        kernel=kernel,
        algo=algo,
        model=model_label,
        a=a,
        a_err=a_err,
        ovh_ms=ovh * 1000,
        ovh_err_ms=ovh_err * 1000,
        r2=r2,
        n_values=n_values,
        t_min=t_min,
        spread=spread,
        model_func=model_func,
    )


# ============================================================
# 5. Plot
# ============================================================
def plot_results(results, outfile, show_fit_res=True):
    kernels = sorted(set(r["kernel"] for r in results))
    fig, axes = plt.subplots(
        1, len(kernels), figsize=(7 * len(kernels), 6), sharey=True
    )
    if len(kernels) == 1:
        axes = [axes]

    colors = {"pastas": "tab:blue", "nufft": "tab:red"}
    markers = {"pastas": "o", "nufft": "s"}
    linestyles = {"pastas": "--", "nufft": ":"}

    for ax, kernel in zip(axes, kernels):
        n_min_all, n_max_all = np.inf, 0
        for algo in ("pastas", "nufft"):
            r = next(
                (x for x in results if x["kernel"] == kernel and x["algo"] == algo),
                None,
            )
            if r is None:
                continue
            n_min_all = min(n_min_all, r["n_values"].min())
            n_max_all = max(n_max_all, r["n_values"].max())

            ax.errorbar(
                r["n_values"],
                r["t_min"],
                yerr=[np.zeros_like(r["spread"]), r["spread"]],
                fmt=markers[algo],
                color=colors[algo],
                capsize=3,
                label=f"{algo} -- data",
            )

            n_grid = np.logspace(
                np.log10(r["n_values"].min()), np.log10(n_max_all), 200
            )
            fit_curve = r["model_func"](n_grid, r["a"], r["ovh_ms"] / 1000)
            if show_fit_res:
                label_info = (
                    f"fit {r['model']} (a={r['a']:.2e}$\pm${r['a_err']:.1e}, "
                    f"ovh={r['ovh_ms']:.1f}$\pm${r['ovh_err_ms']:.1f} ms, "
                    f"$R^2$={r['r2']:.3f})"
                )
            else:
                label_info = (
                    f"fit {r['model']} ($R^2$={r['r2']:.3f})"
                )
            ax.plot(
                n_grid,
                fit_curve,
                linestyles[algo],
                color=colors[algo],
                label=label_info,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of points in series")
        ax.set_title(f"{kernel} kernel")
        fontsz = 8 if show_fit_res else mpl.rcParams['legend.fontsize']
        ax.legend(fontsize=fontsz)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("Computation time [s]")
    if show_fit_res:
        fig.suptitle(
            "ACF benchmark (irregular data): Pastas vs nufftcf -- robust fit (soft_l1)\n"
            "error bars show the min-to-max spread across repeats at each point"
        )
    else:
        fig.suptitle("ACF benchmark (irregular data): Pastas vs nufftcf")
    plt.tight_layout()
    plt.savefig(outfile, format="pdf", bbox_inches="tight", dpi=200)
    print(f"Figure saved: {outfile}")


# ============================================================
# 6. Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit and plot ACF benchmark results from a CSV produced by benchmark_acf.py."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./benchmark_acf_results.csv",
        help="Path to the long-format CSV produced by benchmark_acf.py.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=500,
        help="Number of bootstrap iterations for parameter uncertainties.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap random seed.")
    parser.add_argument(
        "--output-prefix",
        default="benchmark_acf",
        help="Prefix for output files (figure and fit summary CSV).",
    )
    parser.add_argument(
        "--no-save-fit-res",
        dest="save_fit_res",
        action="store_false",
        default=True,
        help="Disable saving the fit results (enabled by default).",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    required_cols = {"method", "n_points", "algo", "time_s"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    results = []
    for kernel in sorted(df["method"].unique()):
        for algo in ("pastas", "nufft"):
            r = analyze_group(df, kernel, algo, n_boot=args.n_boot, seed=args.seed)
            if r is not None:
                results.append(r)

    print("\n=== Fit summary ===")
    summary_rows = []
    for r in results:
        print(
            f"{r['kernel']:9s} {r['algo']:7s} {r['model']:22s} "
            f"a={r['a']:.3e}\u00b1{r['a_err']:.1e}  "
            f"ovh={r['ovh_ms']:.2f}\u00b1{r['ovh_err_ms']:.2f} ms  R2={r['r2']:.4f}"
        )
        summary_rows.append(
            {
                "kernel": r["kernel"],
                "algo": r["algo"],
                "model": r["model"],
                "a": r["a"],
                "a_err": r["a_err"],
                "overhead_ms": r["ovh_ms"],
                "overhead_err_ms": r["ovh_err_ms"],
                "R2": r["r2"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = f"{args.output_prefix}_fit_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nFit summary saved: {summary_csv}")

    plot_results(results, f"{args.output_prefix}_fit.pdf",  show_fit_res=args.save_fit_res)
