# tests/test_bellman.py
import json
from pathlib import Path
import numpy as np
import pytest

from modules.parameters import init_params
from modules.grids import build_capital_grid
from modules.bellman import solve_bellman

def test_bellman_tiny_grid_runs_and_basic_checks(tmp_path):
    params = init_params()
    # tiny grid for speed
    params_local = params.copy()
    params_local["N_K"] = 12
    K = build_capital_grid({"N_K": params_local["N_K"],
                            "K_min": params.get("K_min", 1e-6),
                            "K_max": params.get("K_max", 50.0)},
                           spacing="linear", save_path=str(tmp_path/"K_grid.json"))

    # load A grid from data folder (assume tauchen created it)
    a_path = Path("./data/A_grid.json")
    assert a_path.exists(), "A_grid.json missing; run tauchen first"
    a_obj = json.loads(a_path.read_text())
    A = np.array(a_obj["A_grid"], dtype=float)
    P_A = np.array(a_obj["P_A"], dtype=float)
    # shrink A for test speed
    if A.size > 3:
        A = A[:3]; P_A = P_A[:3,:3]

    V, policy_idx, ext_flag, q = solve_bellman(params_local, K, A, P_A,
                                               enforce_internal_constraint=False,
                                               tol=1e-6, maxiter=800,
                                               compute_q=True, save_path=str(tmp_path/"bellman_out.json"))

    # shape checks
    assert V.shape == (K.size, A.size)
    assert policy_idx.shape == (K.size, A.size)
    assert ext_flag.shape == (K.size, A.size)
    assert q.shape == (K.size, A.size)

    # values finite
    assert np.isfinite(V).all()
    assert np.isfinite(q).all()
    # policy index bounds
    assert policy_idx.min() >= 0
    assert policy_idx.max() < K.size

    # saved file exists and contains keys
    outf = tmp_path/"bellman_out.json"
    assert outf.exists()
    j = json.loads(outf.read_text())
    for key in ("metadata","K_grid","A_grid","policy_k_idx","external_flag","V"):
        assert key in j
