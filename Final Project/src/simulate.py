"""
Simulation helpers for the trucking dynamic program.

Functions:
  - simulate(policy, k_grid, z_grid, P, T, k0=None, z0_idx=None, payoff_fn=None, params=None, seed=None)
      Simulate T periods following `policy`. Returns dict with time series.

  - simulate_from_value_iteration(...):
      Convenience wrapper that runs value_iteration first (if you prefer),
      then simulates under the computed policy.
"""

from typing import Callable, Dict, Any, Optional
import numpy as np


def default_payoff_fn(k: int, z: float, params: Optional[dict] = None) -> float:
    """
    Default payoff: revenue - costs (all in thousands).
    revenue = z * k
    cost = operation_cost * k + fixed_cost
    depreciation uses dynamic price(z) = price_base * (z / spot_mean)
    If params provided, use operation_cost / fixed_cost / depreciation from it.
    Otherwise assume zeros.
    """
    if params is None:
        # fallback: no costs
        return float(z) * int(k)

    op = float(params.get("operation_cost", 0.0))
    fix = float(params.get("fixed_cost", 0.0))
    price_base = float(params.get("price_per_fleet_unit", 0.0))
    spot_mean = float(params.get("spot_rate", 1.0))
    depreciation = float(params.get("depreciation", 0.0))

    # state-dependent price
    price_z = price_base * (float(z) / float(spot_mean))

    revenue = float(z) * int(k)
    operating = op * int(k) + fix
    depreciation_cost = depreciation * price_z * int(k)

    return revenue - operating - depreciation_cost


def simulate(
    policy: np.ndarray,
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    P: np.ndarray,
    T: int,
    k0: Optional[int] = None,
    z0_idx: Optional[int] = None,
    payoff_fn: Optional[Callable[[int, float, Optional[dict]], float]] = None,
    params: Optional[dict] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Simulate T periods following `policy`.

    Parameters
    ----------
    policy : 2D array, shape (Nk, Nz). policy[i_k, i_z] = chosen next k' value (not index)
    k_grid : 1D array of possible k values (integers)
    z_grid : 1D array of shock values (floats)
    P      : transition matrix (Nz x Nz)
    T      : integer, number of periods to simulate
    k0     : initial k value (if None, uses k_grid[0])
    z0_idx : initial shock index (if None, uses middle index len(z_grid)//2)
    payoff_fn : function(k, z, params) -> immediate payoff (float). If None uses default_payoff_fn
    params : params dict (optional), forwarded to payoff_fn if used
    seed   : RNG seed for reproducibility

    Returns
    -------
    dict with keys:
      'k_path'    : array length T+1 of k values
      'z_idx'     : array length T+1 of z indices
      'z_vals'    : array length T+1 of z values
      'payoffs'   : array length T of immediate payoffs (period 0..T-1)
      'policy'    : the policy array passed in (for convenience)
    """
    rng = np.random.RandomState(seed)

    Nk = len(k_grid)
    Nz = len(z_grid)

    # map k value -> index in k_grid
    k_to_idx = {int(k): idx for idx, k in enumerate(k_grid)}

    # initial states
    if k0 is None:
        k0 = int(k_grid[0])
    if z0_idx is None:
        z0_idx = Nz // 2

    # allocate paths
    k_path = np.empty(T + 1, dtype=int)
    z_idx_path = np.empty(T + 1, dtype=int)
    z_vals = np.empty(T + 1, dtype=float)
    payoffs = np.empty(T, dtype=float)

    # initialize
    k_path[0] = int(k0)
    z_idx_path[0] = int(z0_idx)
    z_vals[0] = float(z_grid[z_idx_path[0]])

    # payoff function
    if payoff_fn is None:
        def payoff_fn_local(k_val, z_val, params_local):
            return default_payoff_fn(k_val, z_val, params_local)
        payoff_fn_use = payoff_fn_local
    else:
        # user-supplied expects (k,z,params) or (k,z); try to wrap if needed.
        def payoff_fn_use(k_val, z_val, params_local):
            try:
                return payoff_fn(k_val, z_val, params_local)
            except TypeError:
                return payoff_fn(k_val, z_val)

    # simulate
    for t in range(T):
        k_curr = int(k_path[t])
        z_curr_idx = int(z_idx_path[t])

        # find indices
        k_idx = k_to_idx[k_curr]

        # policy gives NEXT period's k value
        k_next = int(policy[k_idx, z_curr_idx])
        k_path[t + 1] = k_next

        # immediate payoff at time t uses current k and current z
        payoffs[t] = float(payoff_fn_use(k_curr, float(z_grid[z_curr_idx]), params))

        # draw next shock index using transition probabilities P[z_curr_idx]
        probs = P[z_curr_idx]
        probs = np.array(probs, dtype=float)
        s = probs.sum()
        if s <= 0:
            next_z_idx = z_curr_idx
        else:
            probs = probs / s
            next_z_idx = rng.choice(np.arange(Nz), p=probs)

        z_idx_path[t + 1] = int(next_z_idx)
        z_vals[t + 1] = float(z_grid[next_z_idx])

    return {
        "k_path": k_path,
        "z_idx": z_idx_path,
        "z_vals": z_vals,
        "payoffs": payoffs,
        "policy": policy,
    }


def simulate_from_value_iteration(
    value_iteration_fn,
    k_grid,
    z_grid,
    P,
    beta,
    payoff_fn,
    params,
    T=50,
    tol=1e-6,
    max_iter=1000,
    k0=None,
    z0_idx=None,
    seed=None,
):
    """
    Convenience helper: run VFI to obtain policy and then simulate T periods.

    Expects value_iteration_fn(..., payoff_fn, params=...) signature.
    """
    V, pol = value_iteration_fn(
        k_grid=k_grid,
        z_grid=z_grid,
        P=P,
        beta=beta,
        payoff_fn=payoff_fn,
        params=params,
        tol=tol,
        max_iter=max_iter,
    )

    # pol is in k values; pass directly to simulate
    return simulate(
        policy=pol,
        k_grid=k_grid,
        z_grid=z_grid,
        P=P,
        T=T,
        k0=k0,
        z0_idx=z0_idx,
        payoff_fn=payoff_fn,
        params=params,
        seed=seed,
    )

