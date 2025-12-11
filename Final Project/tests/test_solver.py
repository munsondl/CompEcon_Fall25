# tests/test_solver.py

import numpy as np
from src.solver import value_iteration, feasible_actions
from src.config import load_params
from src.payoffs import profit
from src.shocks import tauchen
from src.grids import integer_grid


def test_feasible_actions_basic():
    acts = feasible_actions(5, 10)
    # for k = 5, actions are: sell to 0..5 AND buy to 6
    assert acts == [0, 1, 2, 3, 4, 5, 6]


def test_value_iteration_runs():
    params = load_params()

    # grids
    k_grid = integer_grid(params["grid_k_min"], params["grid_k_max"])

    z_grid, P = tauchen(
        n=5,
        mu=params["spot_rate"],
        rho=params["shock_rho"],
        sigma=params["shock_sigma"],
    )

    # wrap payoff to match solver signature (now requires params)
    def payoff_wrapper(k, z, params_local):
        return profit(k, z, params_local)

    V, pol = value_iteration(
        k_grid=k_grid,
        z_grid=z_grid,
        P=P,
        beta=params["beta"],
        payoff_fn=payoff_wrapper,
        params=params,
        tol=1e-4,
        max_iter=200,
    )

    # shape tests
    assert V.shape == (len(k_grid), len(z_grid))
    assert pol.shape == V.shape

    # each chosen k' must be feasible
    for i, k in enumerate(k_grid):
        for j in range(len(z_grid)):
            assert pol[i, j] in feasible_actions(k, k_grid[-1])
