# modules/tauchen.py
"""
Tauchen discretization of AR(1) process for log productivity.

This module provides:
 - tauchen(rho, sigma_eps, m=3, n=7, mu=0.0) -> (z_grid, P) as numpy arrays
 - discretize_A_from_params(params, save_path=None, m=3.0) -> (A_grid, P_A) as numpy arrays

Behavior:
 - Always returns numpy arrays (A_grid: ndarray, P_A: ndarray).
 - Ensures rows of P_A sum to 1 (numeric normalization).
 - JSON saving converts arrays to lists for serializability.
"""
from pathlib import Path
import json
import math
from typing import Tuple, Optional, Dict, Any
import numpy as np

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def tauchen(rho: float,
            sigma_eps: float,
            m: float = 3.0,
            n: int = 7,
            mu: float = 0.0
            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tauchen (1986) discretization for AR(1): z_t = mu + rho * z_{t-1} + eps_t, eps ~ N(0, sigma_eps^2).

    Returns:
      z_grid : ndarray (n,)  -- grid for z (log A), ascending
      P      : ndarray (n, n) -- transition matrix where P[i,j] = Pr(z' = z_j | z = z_i)

    Notes:
      - If n < 2, raises ValueError.
      - Uses one-sided tails for first/last interval.
    """
    if n < 2:
        raise ValueError("n (number of grid points) must be >= 2")

    # stationary sd of z
    if abs(rho) < 1.0:
        sigma_z = sigma_eps / math.sqrt(1.0 - rho * rho)
    else:
        # degenerate but avoid division by zero
        sigma_z = sigma_eps

    z_max = mu + m * sigma_z
    z_min = mu - m * sigma_z

    step = (z_max - z_min) / (n - 1)
    z_grid = np.array([z_min + i * step for i in range(n)], dtype=float)

    P = np.zeros((n, n), dtype=float)

    for i, z_i in enumerate(z_grid):
        for j in range(n):
            if j == 0:
                z_low = -math.inf
            else:
                z_low = (z_grid[j] - step / 2.0 - rho * z_i - mu) / sigma_eps

            if j == n - 1:
                z_high = math.inf
            else:
                z_high = (z_grid[j] + step / 2.0 - rho * z_i - mu) / sigma_eps

            P[i, j] = _norm_cdf(z_high) - _norm_cdf(z_low)

        # normalize row so numeric issues don't break stochasticity
        row_sum = P[i].sum()
        if row_sum <= 0.0:
            # fallback: put mass on nearest grid point to unconditional mean rho*z_i + mu
            target = mu + rho * z_i
            nearest_j = int(np.argmin(np.abs(z_grid - target)))
            P[i, :] = 0.0
            P[i, nearest_j] = 1.0
        else:
            P[i, :] = P[i, :] / row_sum

    return z_grid, P

def discretize_A_from_params(params: Dict[str, Any],
                             save_path: Optional[str] = None,
                             m: float = 3.0
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper that reads params['rho'], params['sigma'] (innovation std in logs),
    params['N_A'] and returns:
      A_grid (ndarray, levels) and P_A (ndarray, transition matrix).

    Parameters
    ----------
    params : dict
      Must contain 'rho' and 'sigma'. 'N_A' optional (default 7). 'mu_logA' optional (default 0).
    save_path : optional relative path to save JSON (A_grid and P_A). If None, no saving.
    m : float, Tauchen width parameter

    Returns
    -------
    A_grid : ndarray (N_A,)
    P_A : ndarray (N_A, N_A)
    """
    rho = params.get("rho")
    sigma = params.get("sigma")
    n = int(params.get("N_A", 7))
    mu = float(params.get("mu_logA", 0.0))

    if rho is None or sigma is None:
        raise KeyError("params must include 'rho' and 'sigma' for log(A) AR(1)")

    # compute discretization for log(A)
    z_grid, P = tauchen(rho=float(rho), sigma_eps=float(sigma), m=float(m), n=n, mu=mu)

    # convert to levels (positive A)
    A_grid = np.exp(z_grid)

    # ensure numpy arrays and normalize rows robustly
    A_grid = np.asarray(A_grid, dtype=float)
    P = np.asarray(P, dtype=float)
    # protect against tiny negative entries (numeric)
    P = np.clip(P, 0.0, None)
    row_sums = P.sum(axis=1)
    # avoid divide-by-zero
    row_sums_safe = np.where(row_sums == 0.0, 1.0, row_sums)
    P = (P.T / row_sums_safe).T

    # optionally save JSON for reproducibility
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "A_grid": A_grid.tolist(),
            "P_A": P.tolist(),
            "z_grid": z_grid.tolist(),
            "metadata": {
                "rho": float(rho),
                "sigma": float(sigma),
                "N_A": n,
                "m": float(m),
                "mu_logA": float(mu)
            }
        }
        with p.open("w", encoding="utf8") as f:
            json.dump(out, f, indent=2)

    return A_grid, P

# Quick self-test when run standalone
if __name__ == "__main__":
    params_example = {"rho": 0.0976, "sigma": 0.8932, "N_A": 7}
    A_grid, P_A = discretize_A_from_params(params_example, save_path="./data/A_grid.json", m=3.0)
    print("A_grid (first 5):", A_grid[:5])
    print("P_A shape:", P_A.shape)
    print("Row sums (first 5):", np.round(P_A.sum(axis=1)[:5], 8))
