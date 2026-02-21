import numpy as np

def gem_multivariate_normal_density(x, mu, Sigma):
    """Returns the density of a D-dimensional vector x."""
    x = np.array(x)
    mu = np.array(mu)
    Sigma = np.array(Sigma)
    
    D = len(mu)
    det_Sigma = np.linalg.det(Sigma)
    inv_Sigma = np.linalg.inv(Sigma)
    
    # Calculate the normalization constant
    norm_const = 1.0 / (np.power(2 * np.pi, D / 2.0) * np.sqrt(det_Sigma))
    
    # Calculate the exponent term
    diff = x - mu
    exponent = -0.5 * np.dot(np.dot(diff.T, inv_Sigma), diff)
    
    return norm_const * np.exp(exponent)