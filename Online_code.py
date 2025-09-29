import numpy as np
import matplotlib.pyplot as plt
import scipy 
from scipy import stats
from matplotlib.animation import FuncAnimation
import pandas as pd
import seaborn as sns

# seed the random number generator
rng = np.random.default_rng(42)

# determine the number of paths and points per path
points = 1000
paths = 50

# creating the initial set of random normal draws
mu, sigma = 0.0, 1.0
Z = rng.normal(mu, sigma, (paths, points))

# defining the time step and size and t-axis
interval = [0.0, 1.0]
dt = (interval[1] - interval[0]/(points - 1))
t_axis = np.linspace(interval[0], interval[1], points)

W = np.zeros((paths, points))
for idx in range(points - 1) :
    real_idx = idx + 1
    W[:, real_idx] = W[:, real_idx - 1] + np.sqrt(dt) * Z[:, idx]

# Plotting the paths
fig, ax = plt.subplots(1, 1, figsize=(12,8))
for path in range(paths):
    ax.plot(t_axis, W[path, :])
ax.set_title("Standard brownian motion - sample paths")
ax.set_xlabel("Time")
ax.set_ylabel("Asset value")
plt.show()

# Set of final path values
final_values = pd.DataFrame({'final_values' : W[:, -1]})

# Estimate and plot the distribution of the final values with seaborn
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
sns.kdeplot(data=final_values, x='final_values', fill=True, ax=ax)
ax.set_title("kernel density estimate of asset path final value distribution")
ax.set_ylim(0.0, 0.325)
ax.set_xlabel('Final Values of Asset paths')
plt.show()

# mean and standard deviation of the final values
print(final_values.mean(), final_values.std())

# Creating a non_zero mean and standard deviation
mu_c, sigma_c = 5.0, 2.0

# Sampling 50 brownian motion samples
X = np.zeros((paths, points))
for idx in range(points - 1):
    real_idx = idx + 1
    X[:, real_idx] = X[:, real_idx - 1] + mu_c * dt + sigma_c * np.sqrt(dt) * Z[:, idx]

# plotting the paths returns
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
for path in range(paths):
    ax.plot(t_axis, X[path, :])
    ax.set_title("constant mean and standard deviation for sample paths")
    ax.set_xlabel("Time")
    ax.set_ylabel("Ass et value")
    plt.show()   