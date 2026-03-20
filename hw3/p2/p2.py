import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# 1. Define a local RNG with a fixed seed
global_rng = np.random.default_rng(42)

def simulate_ddm(v, a, z, t0, n_trials=2000, dt=0.001, s=1.0, max_t=10.0, rng=global_rng):
        """
        Simulates the basic Drift Diffusion Model using the Euler-Maruyama method.
        """
        results = []
        
        for _ in range(n_trials):
            x = z
            t = 0.0
            
            # Loop until a boundary is hit or max time is reached
            while 0 < x < a and t < max_t:
                # Accumulate evidence with drift and Gaussian noise
                x += v * dt + s * np.sqrt(dt) * rng.normal()
                t += dt
                
            if x >= a:
                results.append({'drift_rate': v, 'boundary': 'upper', 'rt': t + t0})
            elif x <= 0:
                results.append({'drift_rate': v, 'boundary': 'lower', 'rt': t + t0})
                
        return pd.DataFrame(results)

def explore_and_plot(param_name, values, base_params):
    """Sweeps one parameter, plots the distributions, and prints summary stats."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    all_dfs = []
    for val in values:
        # Update the specific parameter being tested
        params = base_params.copy()
        params[param_name] = val
        
        df = simulate_ddm(**params, n_trials=2000)
        df['param_value'] = f"{param_name} = {val}"
        all_dfs.append(df)
        
    df_combined = pd.concat(all_dfs, ignore_index=True)
    
    # Filter for upper boundary to keep visualization clean
    df_upper = df_combined[df_combined['boundary'] == 'upper']
    
    # Summary statistics
    summary = df_upper.groupby('param_value')['rt'].agg(['mean', 'std']).reset_index()
    print(f"\n--- Effects of varying {param_name} ---")
    print(summary)
    
    # Plotting using seaborn as preferred in analyze_eeg.ipynb
    sns.kdeplot(data=df_upper, x='rt', hue='param_value', fill=True, alpha=0.3, ax=ax)
    ax.set_title(f"RT Distributions when varying {param_name}")
    ax.set_xlabel("Response Time (s)")
    plt.show()

def part1():
    # --- Run Part 1 ---
    a_fixed = 1.0
    z_fixed = 0.5
    t0_fixed = 0.3

    # 25 drift rates from 0.5 to 1.5
    v_values = np.linspace(0.5, 1.5, 25)
    all_results = []

    print("Simulating 25 configurations... this will take a moment.")
    for v in v_values:
        df_sim = simulate_ddm(v=v, a=a_fixed, z=z_fixed, t0=t0_fixed, n_trials=2000)
        all_results.append(df_sim)

    df_part1 = pd.concat(all_results, ignore_index=True)

    # Calculate empirical means 
    summary_part1 = df_part1.groupby(['drift_rate', 'boundary'])['rt'].mean().unstack()
    summary_part1['mean_difference'] = summary_part1['upper'] - summary_part1['lower']

    print("\nMean RTs and Differences by Drift Rate:")
    print(summary_part1.head())


def part2():
    # Base parameters
    base_params = {'v': 1.0, 'a': 1.0, 'z': 0.5, 't0': 0.3}

    # 1. Explore Drift Rate (v)
    explore_and_plot('v', [0.5, 1.0, 1.5], base_params)

    # 2. Explore Boundary Separation (a)
    explore_and_plot('a', [0.8, 1.0, 1.2], base_params)

    # 3. Explore Starting Point (z)
    explore_and_plot('z', [0.3, 0.5, 0.7], base_params)

    # 4. Explore Non-Decision Time (t0)
    explore_and_plot('t0', [0.2, 0.4, 0.6], base_params)

PART = "-1"
if __name__ == "__main__":
    if len(sys.argv) > 1:
        PART = sys.argv[1]
    if PART == "1":
        part1()
    elif PART == "2":
        part2()
    else:
        print("What are you doing?")
