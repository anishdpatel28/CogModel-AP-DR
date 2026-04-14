from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT_DIR = Path(__file__).resolve().parent


def load_data():
    df = pd.read_csv(OUT_DIR / "Speed Dating Data.csv", encoding="ISO-8859-1")
    data = df[["attr", "sinc", "intel", "dec"]].dropna()
    X = data[["attr", "sinc", "intel"]]
    y = data["dec"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=SEED
    )
    return X_train, X_test, y_train.values, y_test.values


def main():
    X_train, X_test, y_train, y_test = load_data()
    M = X_test.shape[0]

    model = CmdStanModel(stan_file=str(OUT_DIR / "speed_dating_pred.stan"))
    fit = model.sample(
        data={
            "N": X_train.shape[0],
            "K": X_train.shape[1],
            "X": X_train,
            "y": y_train,
            "M": M,
            "X_test": X_test,
        },
        seed=SEED,
        chains=4,
        parallel_chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        show_progress=False,
    )

    # p_test_draws shape: (num_draws * chains, M)
    p_test_draws = fit.stan_variable("p_test", inc_warmup=False)

    # point-estimate
    p_bar = p_test_draws.mean(axis=0)  # predictive mean per test instance
    brier = np.mean((p_bar - y_test) ** 2)
    y_pred = (p_bar >= 0.5).astype(int)
    accuracy = np.mean(y_pred == y_test)

    # baseline
    majority_class = int(np.round(y_train.mean()))
    baseline_acc = np.mean(majority_class == y_test)
    baseline_brier = np.mean((majority_class - y_test) ** 2)

    print("Point-Estimate Metrics (using predictive means)")
    print(f"Brier Score:  {brier:.4f}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"\nBaseline (always predict {majority_class}):")
    print(f"Brier Score:  {baseline_brier:.4f}")
    print(f"Accuracy:     {baseline_acc:.4f}")

    # distribution over metrics
    n_samples = p_test_draws.shape[0]
    brier_per_iter = np.array([
        np.mean((p_test_draws[s, :] - y_test) ** 2) for s in range(n_samples)
    ])
    acc_per_iter = np.array([
        np.mean(((p_test_draws[s, :] >= 0.5).astype(int)) == y_test)
        for s in range(n_samples)
    ])

    print("\nDistribution Over Metrics (per MCMC iteration)")
    print(f"Brier Score:  mean = {brier_per_iter.mean():.4f}, "
          f"95% CI = [{np.quantile(brier_per_iter, 0.025):.4f}, "
          f"{np.quantile(brier_per_iter, 0.975):.4f}]")
    print(f"Accuracy:     mean = {acc_per_iter.mean():.4f}, "
          f"95% CI = [{np.quantile(acc_per_iter, 0.025):.4f}, "
          f"{np.quantile(acc_per_iter, 0.975):.4f}]")


if __name__ == "__main__":
    main()
