"""
Shock process utilities for the trucking dynamic programming model.

This module discretizes an AR(1) process using Tauchen's method:

    x_{t+1} = mu + rho * (x_t - mu) + epsilon_t
    epsilon_t ~ N(0, sigma^2)

Returns:
    - z_grid : discrete shock values (numpy array)
    - P      : transition probability matrix where P[i,j] = Prob(z_{t+1}=j | z_t=i)

This gives you a Markov chain approximation of the continuous AR(1) spot rate.
"""

import numpy as np
from math import sqrt, erf


# ------------------------------------------------------------
# Helper: Normal CDF
# ------------------------------------------------------------
def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# ------------------------------------------------------------
# Tauchen discretization
# ------------------------------------------------------------
def tauchen(n, mu, rho, sigma, m=3):
    """
    Tauchen method for AR(1) approximation.

    Parameters:
        n     : number of grid points
        mu    : unconditional mean
        rho   : persistence coefficient
        sigma : standard deviation of the innovation
        m     : width of grid in std deviations (default 3)

    Returns:
        z     : state grid (numpy array of length n)
        P     : n x n transition probability matrix
    """
    # Standard deviation of stationary distribution
    std_z = sigma / sqrt(1 - rho**2)

    # Grid boundaries
    z_max = mu + m * std_z
    z_min = mu - m * std_z

    # Even spacing over the interval
    z = np.linspace(z_min, z_max, n)

    # Distance between grid points
    step = (z_max - z_min) / (n - 1)

    # Transition matrix
    P = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if j == 0:
                # lower tail
                upper = (z[0] - mu - rho * (z[i] - mu) + step / 2) / sigma
                P[i, j] = norm_cdf(upper)
            elif j == n - 1:
                # upper tail
                lower = (z[-1] - mu - rho * (z[i] - mu) - step / 2) / sigma
                P[i, j] = 1 - norm_cdf(lower)
            else:
                # middle cells
                lower = (z[j] - mu - rho * (z[i] - mu) - step / 2) / sigma
                upper = (z[j] - mu - rho * (z[i] - mu) + step / 2) / sigma
                P[i, j] = norm_cdf(upper) - norm_cdf(lower)

    return z, P
