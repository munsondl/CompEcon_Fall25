"""
Dynamic programming solver for the trucking model.

States:
    k : number of trucks (integer)
    z : discretized spot-rate shock

Actions:
    k' ∈ {0, 1, ..., k, k+1}  (selling any number of trucks, staying the same, or buying 1)

Adjustment economics (updated):
    - Buying 1 truck costs: price(z) + fixed_cost
    - Selling trucks yields: (k - k') * price(z)
    where price(z) = price_per_fleet_unit * (z / params["spot_rate"])
"""

import numpy as np


def feasible_actions(k, k_max):
    """
    Feasible next-period truck counts from current k.
    """
    actions = list(range(0, k + 1))
    if k < k_max:
        actions.append(k + 1)
    return actions


def value_iteration(
    k_grid,
    z_grid,
    P,
    beta,
    payoff_fn,
    params,
    tol=1e-6,
    max_iter=1000,
):
    """
    Perform value function iteration.

    Expects:
        payoff_fn(k, z, params)
        params: dict (so solver can compute price(z) for buy/sell)

    Returns:
        V   : value function array (Nk x Nz)
        pol : policy array storing optimal k' choices (Nk x Nz)
    """

    Nk = len(k_grid)
    Nz = len(z_grid)

    V = np.zeros((Nk, Nz))
    V_new = np.zeros_like(V)
    pol = np.zeros((Nk, Nz), dtype=int)

    # mapping from k value -> index
    k_to_idx = {k: i for i, k in enumerate(k_grid)}

    price_base = float(params["price_per_fleet_unit"])
    spot_mean = float(params["spot_rate"])
    fixed_cost = float(params["fixed_cost"])
    beta = float(beta)

    for it in range(max_iter):
        diff = 0.0

        for i_k, k in enumerate(k_grid):
            actions = feasible_actions(k, k_grid[-1])

            for i_z, z in enumerate(z_grid):
                # immediate operating profit depends on current z
                pi = payoff_fn(k, z, params)

                best_val = -1e18
                best_kp = k

                # compute price at this z for adjustment mechanics
                price_z = price_base * (float(z) / spot_mean)

                for k_prime in actions:

                    # Adjustment term:
                    if k_prime == k + 1:
                        # Buying one truck: cost = price(z) + fixed_cost
                        adj = -(price_z + fixed_cost)
                    elif k_prime < k:
                        # Selling trucks: revenue = (k - k') * price(z)
                        adj = (k - k_prime) * price_z
                    else:
                        adj = 0.0

                    # Continuation value: expectation over next-period shock
                    EV = np.dot(P[i_z], V[k_to_idx[k_prime]])

                    value = pi + adj + beta * EV

                    if value > best_val:
                        best_val = value
                        best_kp = k_prime

                V_new[i_k, i_z] = best_val
                pol[i_k, i_z] = best_kp

                diff = max(diff, abs(best_val - V[i_k, i_z]))

        if diff < tol:
            return V_new, pol

        V[:] = V_new

    return V_new, pol
