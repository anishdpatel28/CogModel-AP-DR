import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Set up the random number generator
rng = np.random.default_rng(42)

# --- Define the Entertaining Use Case ---
# Prior: Dave claims he is exactly on time (mean = 0), with some uncertainty (std = 5)
mu_0 = 0.0      
sigma_0 = 5.0   

# Likelihood properties (Known Variance)
# Let's say Dave is actually terribly late on average (True mean = 12 minutes)
true_mu = 12.0
known_sigma = 4.0 # Known standard deviation of the data/likelihood

# --- Simulate "Conjured" Data ---
# Simulate 8 data points (e.g., 8 weekly hangouts)
n_samples = 8
data = rng.normal(loc=true_mu, scale=known_sigma, size=n_samples)
y_bar = np.mean(data)

print(f"Simulated Data (Minutes late): {np.round(data, 1)}")
print(f"Sample Mean (y_bar): {y_bar:.2f} minutes")

# --- Compute the Exact Posterior ---
# Using the conjugate update rules for Normal-Normal:
# 1. Posterior Precision = Prior Precision + Data Precision
prior_precision = 1.0 / (sigma_0**2)
data_precision = n_samples / (known_sigma**2)
posterior_precision = prior_precision + data_precision

# 2. Posterior Variance and Standard Deviation
sigma_n_sq = 1.0 / posterior_precision
sigma_n = np.sqrt(sigma_n_sq)

# 3. Posterior Mean
mu_n = sigma_n_sq * ((mu_0 / sigma_0**2) + (n_samples * y_bar / known_sigma**2))

print(f"\nPrior Distribution:     N(mu = {mu_0:.2f}, sigma = {sigma_0:.2f})")
print(f"Posterior Distribution: N(mu = {mu_n:.2f}, sigma = {sigma_n:.2f})")

# --- Visualize using Histograms ---
# Draw samples from the theoretical Prior and Posterior to create the histograms requested
n_plot_samples = 10000
prior_samples = rng.normal(loc=mu_0, scale=sigma_0, size=n_plot_samples)
posterior_samples = rng.normal(loc=mu_n, scale=sigma_n, size=n_plot_samples)

plt.figure(figsize=(10, 6))

# Plot the histograms
sns.histplot(prior_samples, bins=50, color='blue', alpha=0.3, stat='density', 
             label=f'Prior $N({mu_0}, {sigma_0}^2)$')
sns.histplot(posterior_samples, bins=50, color='red', alpha=0.6, stat='density', 
             label=f'Posterior $N({mu_n:.2f}, {sigma_n:.2f}^2)$')

# Add vertical lines for context
plt.axvline(true_mu, color='black', linestyle='--', linewidth=2, label=f'True Mean ($\mu={true_mu}$)')
plt.axvline(y_bar, color='green', linestyle=':', linewidth=2, label=f'Sample Mean ($\\bar{{y}}={y_bar:.2f}$)')

plt.title("Bayesian Updating: Dave's True Tardiness")
plt.xlabel("Minutes Late")
plt.ylabel("Density")
plt.legend()
plt.show()

