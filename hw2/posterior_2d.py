import numpy as np
import matplotlib.pyplot as plt

def posterior(p, s, c):
  return (s * p) / (s * p + (1 - c) * (1 - p))

# fixed val baseline values
prior = 0.01
sensitivity = 0.95
specificity = 0.90

# posterior vs prior
priors = np.linspace(0.001, 0.2, 500)
post_prior = posterior(priors, sensitivity, specificity)

plt.figure()
plt.plot(priors, post_prior)
plt.xlabel("Prior Probability P(D)")
plt.ylabel("Posterior Probability P(D | T+)")
plt.title("Posterior vs Prior Probability")
plt.grid(True)
plt.show()

# posterior vs sensitivity
sensitivities = np.linspace(0.5, 1.0, 500)
post_sens = posterior(prior, sensitivities, specificity)

plt.figure()
plt.plot(sensitivities, post_sens)
plt.xlabel("Sensitivity")
plt.ylabel("Posterior Probability P(D | T+)")
plt.title("Posterior vs Sensitivity")
plt.grid(True)
plt.show()

# posterior vs specificity
specificities = np.linspace(0.5, 1.0, 500)
post_spec = posterior(prior, sensitivity, specificities)

plt.figure()
plt.plot(specificities, post_spec)
plt.xlabel("Specificity")
plt.ylabel("Posterior Probability P(D | T+)")
plt.title("Posterior vs Specificity")
plt.grid(True)
plt.show()