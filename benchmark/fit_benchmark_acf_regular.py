"""
Fit and plot the REGULAR-sampling ACF benchmark results.

Reads the long-format CSV produced by benchmark_acf_regular.py and, for
each kernel (regular/rectangle/gaussian), fits each algo present to its
expected scaling law:

  - Pastas, "gaussian"/"rectangle" bin methods : a * n^2      + overhead
  - Pastas, "regular" bin method               : a * n       + overhead
    (empirically ~O(n): a fixed-size, O(window) np.corrcoef call per lag,
    NOT the O(n^2) slotting technique used by the other two bin methods --
    confirmed by direct timing before writing this script, not assumed.)
  - nufftcf "fft" and "nufft"                : a * n*ln(n) + overhead
    (both scale the same way; "fft" has a smaller `a`, see README)

Same robust-fit / bootstrap machinery as fit_benchmark_acf.py (the
irregular-sampling counterpart) -- see that script's docstring for the
rationale. Deliberately a separate script rather than a generalization of
fit_benchmark_acf.py: the column schema, kernel set (3, not 2), and algo
set (up to 3, not 2, with a model override) all differ enough that sharing
one script would need as many special cases as just keeping the original,
already-published-numbers script untouched.

Usage: python fit_benchmark_acf_regular.py <data.csv> [--n-boot 500] [--output-prefix benchmark_acf_regular]
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
def model_quadratic(n, a, ovh):
    return a * n**2 + ovh


def model_linear(n, a, ovh):
    return a * n + ovh


def model_nlogn(n, a, ovh):
    return a * n * np.log(n) + ovh


# Default model per algo ...
ALGO_MODELS = {
    "pastas": (model_quadratic, f"$a.n^2 + overhead$"),
    "fft": (model_nlogn, f"$a.n.\ln(n) + overhead$"),
    "nufft": (model_nlogn, f"$a.n.\ln(n) + overhead$"),
}
# ... overridden for the one case that scales differently: Pastas has no
# smoothing kernel to apply in "regular" mode, just a fixed-size windowed
# np.corrcoef per lag, so it doesn't pay the O(n^2) cost the slotting
# technique incurs for "gaussian"/"rectangle".
MODEL_OVERRIDE = {
    ("regular", "pastas"): (model_linear, f"$a.n + overhead$"),
}


def get_model(kernel, algo):
    return MODEL_OVERRIDE.get((kernel, algo), ALGO_MODELS[algo])


# ============================================================
# 2. Robust fit (soft_l1) -- identical to fit_benchmark_acf.py
# ============================================================
def robust_fit(n_arr, t_arr, model_func, p0):
    n_arr = np.asarray(n_arr, dtype=float)
    t_arr = np.asarray(t_arr, dtype=float)
    bounds = (np.array([0.0, 0.0]), np.array([np.inf, np.inf]))

    def residuals(params):
        a, ovh = params
        return model_func(n_arr, a, ovh) - t_arr

    res0 = least_squares(residuals, p0, bounds=bounds)
    mad = np.median(np.abs(res0.fun - np.median(res0.fun))) or 1e-6
    f_scale = max(mad, 1e-6)
    res = least_squares(
        residuals, res0.x, loss="soft_l1", f_scale=f_scale, bounds=bounds
    )
    return res.x


def r_squared(n_arr, t_arr, model_func, params):
    pred = model_func(np.asarray(n_arr, dtype=float), *params)
    t_arr = np.asarray(t_arr, dtype=float)
    ss_res = np.sum((t_arr - pred) ** 2)
    ss_tot = np.sum((t_arr - t_arr.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


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
            boot_params.append(robust_fit(n_values, t_boot, model_func, p0))
        except Exception:
            continue
    boot_params = np.array(boot_params)
    if len(boot_params) == 0:
        return np.array([np.nan, np.nan])
    return boot_params.std(axis=0)


# ============================================================
# 3. Per (kernel, algo) analysis
# ============================================================
def analyze_group(df, kernel, algo, n_boot, seed):
    sub = df[(df["method"] == kernel) & (df["algo"] == algo)]
    if sub.empty or sub["n_points"].nunique() < 3:
        return None

    repeats_by_n = {n: g["time_s"].to_numpy() for n, g in sub.groupby("n_points")}
    n_values = np.array(sorted(repeats_by_n.keys()), dtype=float)
    t_min = np.array([np.median(repeats_by_n[n]) for n in n_values])

    model_func, model_label = get_model(kernel, algo)
    p0 = [1e-7, 0.01] if algo == "pastas" else [1e-7, 0.001]

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
# 4. Plot: one panel per kernel, one figure per kernel saved separately
#    (mirrors fit_benchmark_acf.py's plot2_results)
# ============================================================
COLORS = {"pastas": "tab:blue", "fft": "tab:red", "nufft": "tab:green"}
MARKERS = {"pastas": "o", "fft": "s", "nufft": "^"}
LINESTYLES = {"pastas": "--", "fft": ":", "nufft": "-."}


def plot_results(results, output_prefix, show_fit_res=True):
    kernels = sorted(set(r["kernel"] for r in results))
    for kernel in kernels:
        fig, ax = plt.subplots(figsize=(7, 6))
        n_max_all = max(r["n_values"].max() for r in results if r["kernel"] == kernel)
        for algo in ("pastas", "fft", "nufft"):
            r = next(
                (x for x in results if x["kernel"] == kernel and x["algo"] == algo),
                None,
            )
            if r is None:
                continue
            ax.errorbar(
                r["n_values"],
                r["t_min"],
                yerr=[np.zeros_like(r["spread"]), r["spread"]],
                fmt=MARKERS[algo],
                color=COLORS[algo],
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
                label_info = f"fit {r['model']} ($R^2$={r['r2']:.3f})"
            ax.plot(
                n_grid,
                fit_curve,
                LINESTYLES[algo],
                color=COLORS[algo],
                label=label_info,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of points in series")
        ax.set_ylabel("Computation time [s]")
        ax.set_title(f"{kernel}")
        fontsz = 8 if show_fit_res else mpl.rcParams["legend.fontsize"]
        ax.legend(fontsize=fontsz)
        ax.grid(True, which="both", alpha=0.3)
        fig.suptitle("ACF benchmark (regular data): Pastas vs nufftcf (fft / nufft)")
        plt.tight_layout()
        outfile = f"{kernel}_{output_prefix}_fit.pdf"
        fig.savefig(outfile, format="pdf", bbox_inches="tight", dpi=200)
        print(f"Figure saved: {outfile}")


# ============================================================
# 5. Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit and plot REGULAR-sampling ACF benchmark results."
    )
    parser.add_argument("--csv_path", default="./benchmark_acf_regular_results.csv")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-prefix", default="benchmark_acf_regular")
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
        for algo in ("pastas", "fft", "nufft"):
            r = analyze_group(df, kernel, algo, n_boot=args.n_boot, seed=args.seed)
            if r is not None:
                results.append(r)

    print("\n=== Fit summary ===")
    summary_rows = []
    for r in results:
        print(
            f"{r['kernel']:9s} {r['algo']:7s} {r['model']:18s} "
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

    plot_results(results, args.output_prefix, show_fit_res=args.save_fit_res)
