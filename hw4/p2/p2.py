from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

SEED = 2026
OUT_DIR = Path(__file__).resolve().parent


def simulate_data(rng: np.random.Generator):
    n_old, n_new = 50, 50
    hits = int(rng.binomial(n_old, 0.72))
    false_alarms = int(rng.binomial(n_new, 0.22))

    print("Simulated Recognition Memory Data")
    print(f"Hits: {hits}/{n_old},  False alarms: {false_alarms}/{n_new}")
    return {"n_old": n_old, "n_new": n_new, "hits": hits, "false_alarms": false_alarms}


def print_diagnostics(summary_df: pd.DataFrame, params: list[str], label: str):
    print(f"\n{label}: Convergence Diagnostics")
    diag_cols = ["R_hat", "ESS_bulk", "ESS_tail"]
    print(summary_df.loc[params, diag_cols].to_string(float_format=lambda v: f"{v:.4f}"))

    for p in params:
        rhat = summary_df.loc[p, "R_hat"]
        if np.isnan(rhat) or rhat >= 1.01:
            print(f"  WARNING: R-hat for {p} = {rhat:.4f}")


def print_posterior(fit, params: list[str], label: str):
    print(f"\n{label}: Posterior Summary")
    rows = []
    for p in params:
        draws = fit.stan_variable(p, inc_warmup=False)
        rows.append({
            "param": p,
            "mean": float(np.mean(draws)),
            "sd": float(np.std(draws, ddof=1)),
            "ci_2.5": float(np.quantile(draws, 0.025)),
            "ci_97.5": float(np.quantile(draws, 0.975)),
        })
    print(pd.DataFrame(rows).set_index("param").to_string(float_format=lambda v: f"{v:.4f}"))


def save_trace_plots(fit, params: list[str], out_path: Path):
    fig, axes = plt.subplots(len(params), 1, figsize=(10, 3 * len(params)), sharex=True)
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
    fig.suptitle(out_path.stem)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fit_model(stan_file: str, data_dict: dict, label: str):
    model = CmdStanModel(stan_file=str(OUT_DIR / stan_file))
    fit = model.sample(data=data_dict, seed=SEED, chains=4, parallel_chains=4, show_progress=False)
    summary_df = fit.summary()

    params = ["d", "g"]
    print_diagnostics(summary_df, params, label)
    print_posterior(fit, params, label)

    tag = stan_file.replace(".stan", "").replace("mpt_", "")
    save_trace_plots(fit, params, OUT_DIR / f"trace_{tag}.png")
    return fit


def main():
    rng = np.random.default_rng(SEED)
    data_dict = simulate_data(rng)

    print("\n")
    fit_1ht = fit_model("mpt_1ht.stan", data_dict, "1HT")
    print("\n")
    fit_2ht = fit_model("mpt_2ht.stan", data_dict, "2HT")


if __name__ == "__main__":
    main()
