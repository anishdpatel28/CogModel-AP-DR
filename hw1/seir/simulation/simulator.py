from typing import Tuple

import numpy as np


def simulate_seir(
    parameters: Tuple[float, float, float],
    init_conditions: Tuple[float, float, float, float],
    days: int = 51,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Extract parameters and initial conditions
    beta, sigma, gamma = parameters
    S0, E0, I0, R0 = init_conditions
    N = S0 + E0 + I0 + R0

    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    S[0], E[0], I[0], R[0] = S0, E0, I0, R0

    # For each day, perform SEIR update
    for t in range(1, days):
        # Compute new cases and update equations (Your code here)
        E_new = (beta * S[t - 1] * I[t - 1]) / N
        I_new = sigma * E[t - 1]
        R_new = gamma * I[t - 1]

        # Update equations
        S[t] = S[t - 1] - E_new
        E[t] = E[t - 1] + E_new - I_new
        I[t] = I[t - 1] + I_new - R_new
        R[t] = R[t - 1] + R_new

    return (S, E, I, R)
