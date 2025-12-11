# tests/test_simulate.py
import numpy as np
from src.simulate import simulate
from src.solver import value_iteration
from src.shocks import tauchen
from src.grids import integer_grid
from src.config import load_params
from src.payoffs import profit


def test_simulate_basic_deterministic():
    params = load_params()

    # small grids for test
    k_grid = integer_grid(params["grid_k_min"], params["grid_k_max"])
    # construct trivial shock grid and identity transitions for determinism
    z_grid = np.array([params["spot_rate"]])
    P = np.ones((1, 1))  # always stay in the single state

    # policy that always keeps the same k (policy[i_k, i_z] = k_grid[i_k])
    pol = np.zeros((len(k_grid), len(z_grid)), dtype=int)
    for i, k in enumerate(k_grid):
        pol[i, 0] = int(k)

    # simulate T=5 starting from k0 = 2 and z0_idx = 0
    out = simulate(policy=pol, k_grid=k_grid, z_grid=z_grid, P=P, T=5, k0=2, z0_idx=0, seed=0, params=params)

    # Check lengths
    assert len(out["k_path"]) == 6
    assert len(out["z_idx"]) == 6
    assert len(out["payoffs"]) == 5

    # policy is to keep same k, so k_path should be constant at 2
    assert all(out["k_path"] == 2)

    # payoffs: use profit(k, z, params) with z = params["spot_rate"]
    expected_pay = profit(2, params["spot_rate"], params)
    assert np.allclose(out["payoffs"], expected_pay)


def test_simulate_follows_policy_from_vfi():
    params = load_params()
    k_grid = integer_grid(params["grid_k_min"], params["grid_k_max"])

    # small shock discretization
    z_grid, P = tauchen(n=3, mu=params["spot_rate"], rho=params["shock_rho"], sigma=params["shock_sigma"])

    # build VFI policy quickly (use small tol and max_iter)
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

    # Simulate deterministically by making P identity for this check
    P_det = np.eye(len(z_grid))
    out = simulate(policy=pol, k_grid=k_grid, z_grid=z_grid, P=P_det, T=5, k0=0, z0_idx=1, seed=0, params=params)

    # Check that each transition followed the policy
    for t in range(5):
        k_curr = out["k_path"][t]
        z_idx = out["z_idx"][t]
        expected_k_next = pol[int(np.where(k_grid == k_curr)[0][0]), z_idx]
        assert out["k_path"][t + 1] == expected_k_next
