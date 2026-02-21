import numpy as np

class MultivariateNormal:
    """
    A custom implementation of a Multivariate Normal Distribution 
    matching SciPy's basic semantics.
    """
    def __init__(self, mu, cov):
        self.mu = np.array(mu)
        self.cov = np.array(cov)
        self.D = len(self.mu)
        self.inv_cov = np.linalg.inv(self.cov)
        self.det_cov = np.linalg.det(self.cov)
        
    def logpdf(self, x):
        """Returns the log probability density of vector x."""
        x = np.array(x)
        diff = x - self.mu
        
        # Log of the normalization constant
        log_norm_const = -0.5 * self.D * np.log(2 * np.pi) - 0.5 * np.log(self.det_cov)
        
        # Exponent term (using einsum to handle both 1D and 2D arrays like SciPy)
        if x.ndim == 1:
            exponent = -0.5 * np.dot(diff.T, np.dot(self.inv_cov, diff))
        else:
            exponent = -0.5 * np.einsum('...i,ij,...j->...', diff, self.inv_cov, diff)
            
        return log_norm_const + exponent
        
    def rvs(self, size=1):
        """Generates random variables."""
        return np.random.multivariate_normal(self.mu, self.cov, size=size)