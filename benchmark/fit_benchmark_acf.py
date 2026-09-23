"""
Fit and plot ACF benchmark results.

Reads the long-format CSV produced by benchmark_acf.py (one row per
individual repeat) and, for each kernel (gaussian/rectangle):
  - Pastas timings are fit to           a * n^2                  + overhead
  - nufftcf timings are fit to  a_lin * n  +  a_nlogn * n*ln(n)  + overhead

CHANGED: the nufftcf model now
has two additive terms because `compute_*_nufft` does two things of
different asymptotic cost at a FIXED number of lags K (K=366 throughout
this benchmark, see `lags = np.arange(0.0, 366.0)` in benchmark_acf.py):
the NUFFT type-1/type-2 calls, O(n log n); and the pair-count / kernel
normalization (`kernels.py`'s two-pointer scan), O(n) per lag, i.e. O(nK)
overall since K is fixed here. For fixed K, `a_lin*n` and
`a_nlogn*n*ln(n)` are only weakly distinguishable over one to two decades
of n (ln(n) barely changes), so both coefficients are fit under the
physical constraint a_lin, a_nlogn >= 0 (already how `robust_fit` bounds
every parameter) rather than assuming either term is exactly zero. The
n*ln(n)-only model used previously implicitly assumed the pair-count term
was negligible; on these data the fit instead drives a_nlogn to ~0, i.e.
the pair-count term alone already explains the timings within noise over
the range tested (a_lin*n and a_nlogn*n*ln(n) would only become
comparable at n ~ exp(K), meaningless here). The O(n log n) label
describes the NUFFT calls alone, not the full `compute_*_nufft` pipeline
measured here -- see `dominant_term_note` below, printed with the fit
summary.

Fitting is ROBUST (scipy.optimize.least_squares with loss='soft_l1'),
which down-weights the influence of one-off outlier points (e.g. a single
measurement hit by transient thermal throttling or OS scheduling noise)
without requiring you to manually identify and drop them. All model
parameters are constrained to be non-negative (overhead < 0 is not
physical, and an unconstrained fit landing there -- or, for the nufft
model, an unconstrained n*ln(n) coefficient landing negative -- is usually
a sign of a misspecified/over-parameterized model rather than a
meaningful negative term -- see discussion above).

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

mpl.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
    }
)


# ============================================================
# 1. Models
# ============================================================
def model_pastas(n, a, ovh):
    return a * n**2 + ovh


def model_nufft(n, a_lin, a_nlogn, ovh):
    """Two additive terms: pair-count O(n) [at fixed K] + NUFFT O(n log n).
    See module docstring for why both are fit jointly instead of assuming
    the pair-count term away."""
    return a_lin * n + a_nlogn * n * np.log(n) + ovh


# Each entry: (model_func, model_label, initial_guess_p0)
MODELS = {
    "pastas": (model_pastas, r"$a.n^2 + ovh$", [1e-7, 0.01]),
    "nufft": (
        model_nufft,
        r"$a_1.n + a_2.n.\ln(n) + ovh$",
        [1e-6, 1e-9, 0.005],
    ),
}


# ============================================================
# 2. Robust fit (soft_l1), with an initial linear-loss pass to pick a
#    sensible f_scale (the residual magnitude beyond which soft_l1 starts
#    down-weighting points like a robust loss rather than a plain square)
# ============================================================
def robust_fit(n_arr, t_arr, model_func, p0):
    n_arr = np.asarray(n_arr, dtype=float)
    t_arr = np.asarray(t_arr, dtype=float)
    n_params = len(p0)
    # Bounds are now built from len(p0), so this works for both the
    # 2-parameter (pastas) and 3-parameter (nufft) models.
    bounds = (np.zeros(n_params), np.full(n_params, np.inf))

    def residuals(params):
        return model_func(n_arr, *params) - t_arr

    # Pass 1: ordinary least squares, just to get a robust scale estimate
    res0 = least_squares(residuals, p0, bounds=bounds)
    mad = np.median(np.abs(res0.fun - np.median(res0.fun))) or 1e-6
    f_scale = max(mad, 1e-6)

    # Pass 2: robust refit using that scale
    res = least_squares(
        residuals, res0.x, loss="soft_l1", f_scale=f_scale, bounds=bounds
    )
    return res.x  # [a, ovh] for pastas; [a_lin, a_nlogn, ovh] for nufft


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

    model_func, model_label, p0 = MODELS[algo]

    params = robust_fit(n_values, t_min, model_func, p0)
    r2 = r_squared(n_values, t_min, model_func, params)
    param_err = bootstrap_fit(repeats_by_n, model_func, p0, n_boot=n_boot, seed=seed)

    spread = np.array([repeats_by_n[n].max() - repeats_by_n[n].min() for n in n_values])

    return dict(
        kernel=kernel,
        algo=algo,
        model=model_label,
        params=params,
        param_err=param_err,
        r2=r2,
        n_values=n_values,
        t_min=t_min,
        spread=spread,
        model_func=model_func,
    )


# ============================================================
# 4bis. Human-readable parameter labels + dominant-term diagnostic
# ============================================================
PARAM_NAMES = {
    "pastas": ["a", "ovh"],
    "nufft": ["a_1", "a_2", "ovh"],
}


def format_params(algo, params, param_err):
    names = PARAM_NAMES[algo]
    parts = []
    for name, val, err in zip(names, params, param_err):
        if name.startswith("ovh"):
            parts.append(f"${name}$={val * 1000:.2f}\u00b1{err * 1000:.2f} ms")
        else:
            parts.append(f"${name}$={val:.2e}\u00b1{err:.1e}")
    return "  ".join(parts)


def dominant_term_note(algo, params, n_max):
    """At n = n_max, which of the two nufft terms explains more of the
    predicted time? Purely descriptive -- not used by the fit itself."""
    if algo != "nufft":
        return ""
    a_lin, a_nlogn, _ = params
    lin_part = a_lin * n_max
    nlogn_part = a_nlogn * n_max * np.log(n_max)
    total = lin_part + nlogn_part
    if total <= 0:
        return ""
    frac_lin = lin_part / total
    return (
        f"  [at n={n_max:.0f}: linear term is {frac_lin:.1%} of the fitted "
        f"nufft time, n*ln(n) term is {1 - frac_lin:.1%}]"
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
            fit_curve = r["model_func"](n_grid, *r["params"])
            if show_fit_res:
                # Generic parameter formatting (works for 2 or 3 params).
                param_str = format_params(algo, r["params"], r["param_err"])
                label_info = f"fit {r['model']} ($R^2$={r['r2']:.3f})\n {param_str}"
            else:
                label_info = f"fit {r['model']} ($R^2$={r['r2']:.3f})"
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
        fontsz = 7 if show_fit_res else mpl.rcParams["legend.fontsize"]
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
        note = dominant_term_note(r["algo"], r["params"], r["n_values"].max())
        print(
            f"{r['kernel']:9s} {r['algo']:7s} {r['model']:28s} "
            f"{format_params(r['algo'], r['params'], r['param_err'])}  "
            f"R2={r['r2']:.4f}{note}"
        )
        row = {
            "kernel": r["kernel"],
            "algo": r["algo"],
            "model": r["model"],
            "R2": r["r2"],
        }
        for name, val, err in zip(PARAM_NAMES[r["algo"]], r["params"], r["param_err"]):
            row[name] = val
            row[f"{name}_err"] = err
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = f"{args.output_prefix}_fit_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nFit summary saved: {summary_csv}")

    plot_results(
        results, f"{args.output_prefix}_fit.pdf", show_fit_res=args.save_fit_res
    )
