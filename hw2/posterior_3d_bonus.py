import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from posterior_2d import *

priors = np.linspace(0.001, 0.2, 100)
sensitivities = np.linspace(0.5, 1.0, 100)
specificities = np.linspace(0.5, 1.0, 100)

# prior vs Sensitivity
P, S = np.meshgrid(priors, sensitivities)
Z1 = posterior(P, S, specificity)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(P, S, Z1)
ax.set_xlabel("Prior")
ax.set_ylabel("Sensitivity")
ax.set_zlabel("Posterior P(D | T+)")
ax.set_title("Posterior vs Prior and Sensitivity")
plt.show()

# prior vs specificity
P, C = np.meshgrid(priors, specificities)
Z2 = posterior(P, sensitivity, C)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(P, C, Z2)
ax.set_xlabel("Prior")
ax.set_ylabel("Specificity")
ax.set_zlabel("Posterior P(D | T+)")
ax.set_title("Posterior vs Prior and Specificity")
plt.show()

# sensitivity vs specificity
S, C = np.meshgrid(sensitivities, specificities)
Z3 = posterior(prior, S, C)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S, C, Z3)
ax.set_xlabel("Sensitivity")
ax.set_ylabel("Specificity")
ax.set_zlabel("Posterior P(D | T+)")
ax.set_title("Posterior vs Sensitivity and Specificity")
plt.show()