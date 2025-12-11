"""
Grid constructors for the trucking dynamic programming model.

Functions included:
    - integer_grid(k_min, k_max)
        Creates a grid of integers from k_min to k_max inclusive.
        This is the grid for truck fleet size.

    - uniform_grid(a, b, n)
        Creates n evenly spaced points on [a, b].
        Included only for potential simple continuous variables.
"""

import numpy as np


# ------------------------------------------------------------
# 1. INTEGER GRID (MAIN GRID FOR TRUCK COUNTS)
# ------------------------------------------------------------
def integer_grid(k_min: int, k_max: int) -> np.ndarray:
    """
    Returns integer values from k_min to k_max inclusive.

    Example:
        integer_grid(0, 10) -> array([0, 1, 2, ..., 10])
    """
    k_min_i = int(np.floor(k_min))
    k_max_i = int(np.ceil(k_max))

    if k_max_i < k_min_i:
        raise ValueError("k_max must be >= k_min")

    return np.arange(k_min_i, k_max_i + 1, dtype=int)


# ------------------------------------------------------------
# 2. SIMPLE CONTINUOUS GRID
# ------------------------------------------------------------
def uniform_grid(a: float, b: float, n: int) -> np.ndarray:
    """
    n evenly spaced points on [a, b].
    If n <= 1, returns [a].

    Example:
        uniform_grid(0, 1, 5) -> array([0., 0.25, 0.5, 0.75, 1.])
    """
    if n <= 1:
        return np.array([float(a)])
    return np.linspace(float(a), float(b), int(n))
