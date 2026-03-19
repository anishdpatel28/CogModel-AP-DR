from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel


TRUE_ALPHA = 2.3
TRUE_BETA = 4.0
TRUE_SIGMA = 2.0
SEED = 2026


def simulate_data(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=n)
    y = TRUE_ALPHA + TRUE_BETA * x + TRUE_SIGMA * rng.normal(size=n)
    return x, y


def print_diagnostics(summary_df, section: str) -> None:
    print(f"\n{section}: Convergence and Efficiency Diagnostics")
    params = ["alpha", "beta", "sigma"]
    diag_cols = ["R_hat", "ESS_bulk", "ESS_tail"]
    print(summary_df.loc[params, diag_cols].to_string(float_format=lambda v: f"{v:0.4f}"))

    warnings = []
    for p in params:
        rhat = summary_df.loc[p, "R_hat"]
        if np.isnan(rhat) or rhat >= 1.01:
            warnings.append(f"R-hat warning for {p}: {rhat}")

        ess_bulk = summary_df.loc[p, "ESS_bulk"]
        ess_tail = summary_df.loc[p, "ESS_tail"]
        if np.isnan(ess_bulk) or ess_bulk < 400:
            warnings.append(f"Low ESS bulk for {p}: {ess_bulk}")
        if np.isnan(ess_tail) or ess_tail < 400:
            warnings.append(f"Low ESS tail for {p}: {ess_tail}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")
    else:
        print("\nWarnings:\n- None")


def print_posterior_summary(fit, section: str) -> None:
    print(f"\n{section}: Posterior Summary")
    params = ["alpha", "beta", "sigma"]
    rows = []
    for p in params:
        draws = fit.stan_variable(p, inc_warmup=False)
        rows.append(
            {
                "param": p,
                "mean": float(np.mean(draws)),
                "sd": float(np.std(draws, ddof=1)),
                "ci_2.5": float(np.quantile(draws, 0.025)),
                "ci_97.5": float(np.quantile(draws, 0.975)),
            }
        )
    view = pd.DataFrame(rows).set_index("param")[["mean", "sd", "ci_2.5", "ci_97.5"]]
    print(view.to_string(float_format=lambda v: f"{v:0.4f}"))


def save_trace_plots(fit, out_path: Path) -> None:
    params = ["alpha", "beta", "sigma"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, p in enumerate(params):
        draws = fit.stan_variable(p, inc_warmup=False)
        chain_draws = draws.reshape(fit.chains, fit.num_draws_sampling).T
        for c in range(fit.chains):
            axes[i].plot(chain_draws[:, c], lw=0.8, alpha=0.85, label=f"chain {c + 1}")
        axes[i].set_ylabel(p)
        axes[i].grid(alpha=0.2)
        if i == 0:
            axes[i].legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("Post-warmup iteration")
    fig.suptitle("Trace plots")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_histograms(fit, out_path: Path) -> None:
    params = ["alpha", "beta", "sigma"]
    truths = [TRUE_ALPHA, TRUE_BETA, TRUE_SIGMA]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, p, truth in zip(axes, params, truths):
        draws = fit.stan_variable(p, inc_warmup=False)
        ax.hist(draws, bins=40, density=True, alpha=0.75, color="#4c72b0", edgecolor="white")
        ax.axvline(truth, color="red", linestyle="--", linewidth=2, label=f"true={truth:0.2f}")
        ax.set_title(p)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Posterior histograms")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_posterior_predictive_plot(fit, x: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    alpha_draws = fit.stan_variable("alpha", inc_warmup=False)
    beta_draws = fit.stan_variable("beta", inc_warmup=False)

    x_grid = np.linspace(np.min(x), np.max(x), 200)
    y_grid_draws = alpha_draws[:, None] + beta_draws[:, None] * x_grid[None, :]
    y_mean = np.mean(y_grid_draws, axis=0)
    y_low = np.quantile(y_grid_draws, 0.025, axis=0)
    y_high = np.quantile(y_grid_draws, 0.975, axis=0)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, s=16, alpha=0.5, label="observed data")
    ax.plot(x_grid, y_mean, color="black", linewidth=2, label="posterior mean line")
    ax.fill_between(x_grid, y_low, y_high, alpha=0.25, color="gray", label="95% credible band")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior predictive regression fit")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_analysis(
    n: int,
    model: CmdStanModel,
    out_dir: Path,
    rng: np.random.Generator,
    suffix: str = "",
):
    print(f"\n{'=' * 70}")
    print(f"Bayesian Linear Regression with Stan (N={n})")
    print(f"{'=' * 70}")

    x, y = simulate_data(n, rng)
    fit = model.sample(
        data={"N": n, "x": x, "y": y},
        seed=SEED + n,
        chains=4,
        parallel_chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        show_progress=False,
    )

    summary_df = fit.summary()
    section = f"N={n}"
    print_diagnostics(summary_df, section)
    print_posterior_summary(fit, section)

    trace_name = f"trace_plots{suffix}.png"
    hist_name = f"posterior_histograms{suffix}.png"
    pred_name = f"posterior_predictive{suffix}.png"
    save_trace_plots(fit, out_dir / trace_name)
    save_histograms(fit, out_dir / hist_name)
    save_posterior_predictive_plot(fit, x, y, out_dir / pred_name)

    print(f"\nSaved plots: {trace_name}, {hist_name}, {pred_name}")
    return summary_df


def compare_precision(summary_100, summary_1000) -> None:
    print("\nN=100 vs N=1000: Posterior SD Comparison")
    params = ["alpha", "beta", "sigma"]
    print("param    sd(N=100)  sd(N=1000)  ratio(N100/N1000)")
    for p in params:
        sd_100 = float(summary_100.loc[p, "StdDev"])
        sd_1000 = float(summary_1000.loc[p, "StdDev"])
        ratio = sd_100 / sd_1000 if sd_1000 > 0 else np.nan
        print(f"{p:<8}{sd_100:>10.4f}{sd_1000:>12.4f}{ratio:>16.2f}")

    print(
        "\nWith N=1000, posterior standard deviations are smaller, "
        "so parameter precision increases and uncertainty decreases compared to N=100."
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    stan_file = out_dir / "model.stan"
    model = CmdStanModel(stan_file=str(stan_file))
    rng = np.random.default_rng(SEED)

    summary_100 = run_analysis(100, model, out_dir, rng, suffix="")
    summary_1000 = run_analysis(1000, model, out_dir, rng, suffix="_N1000")
    compare_precision(summary_100, summary_1000)


if __name__ == "__main__":
    main()
