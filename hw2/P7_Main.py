import numpy as np
from scipy.stats import multivariate_normal

# --- COMPARISONS ---
x_test = [0.5, -0.2]
mu_test = [0, 0]

# 1. Spherical Gaussian (shared variance, zero covariance)
cov_spherical = [[2, 0], 
                 [0, 2]]

# 2. Diagonal Gaussian (different variance, zero covariance)
cov_diagonal = [[2, 0], 
                [0, 5]]

# 3. Full-covariance Gaussian (different variance, non-zero covariance)
cov_full = [[2, 1], 
            [1, 5]]

covariances = {
    "Spherical": cov_spherical,
    "Diagonal": cov_diagonal,
    "Full-covariance": cov_full
}

from P7_Function import gem_multivariate_normal_density

print("--- Density Comparison ---")
for name, cov in covariances.items():
    my_density = gem_multivariate_normal_density(x_test, mu_test, cov)
    scipy_density = multivariate_normal(mu_test, cov).pdf(x_test)
    print(f"{name}:")
    print(f"  Custom Function: {my_density:.8f}")
    print(f"  SciPy Function:  {scipy_density:.8f}\n")


from P7_Class import MultivariateNormal

# Define test parameters
x_test = [0.5, -0.2]
mu_test = [0, 0]
cov_full = [[2, 1], 
            [1, 5]]

# Initialize both classes
my_mvn = MultivariateNormal(mu_test, cov_full)
scipy_mvn = multivariate_normal(mu_test, cov_full)

# Compare the Log PDFs
print("--- Class Method Comparison (Log PDF) ---")
print(f"Custom Class: {my_mvn.logpdf(x_test):.8f}")
print(f"SciPy Class:  {scipy_mvn.logpdf(x_test):.8f}")

# Compare random variable generation shape
print("\n--- Random Variable Generation Shape ---")
print(f"Custom rvs shape: {my_mvn.rvs(size=5).shape}")
print(f"SciPy rvs shape:  {scipy_mvn.rvs(size=5).shape}")