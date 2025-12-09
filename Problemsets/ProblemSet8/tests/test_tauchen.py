# tests/test_tauchen.py
"""
Unit tests for the Tauchen-produced A_grid JSON.

Run from project root:
  pytest -q

The tests:
 - confirm ./data/A_grid.json exists
 - confirm A_grid is numeric, positive, strictly increasing
 - confirm A_grid length matches N_A in params (if available) otherwise checks >= 2
 - confirm P_A is square with same dimension and that each row sums to 1 (within tolerance)
 - if z_grid exists in the JSON, check that A_grid ~= exp(z_grid)
"""
from pathlib import Path
import json
import math
import os
import pytest

PROJECT_ROOT = Path(".").resolve()
DATA_JSON = PROJECT_ROOT / "data" / "A_grid.json"
PARAMS_JSON = PROJECT_ROOT / "data" / "params_default.json"

def load_json(path: Path):
    with path.open("r", encoding="utf8") as f:
        return json.load(f)

def approx_equal(a, b, tol=1e-8):
    return abs(a - b) <= tol

def test_a_grid_json_exists():
    assert DATA_JSON.exists(), f"Expected A_grid JSON at {DATA_JSON.resolve()} but file not found."

def test_a_grid_structure_and_bounds():
    data = load_json(DATA_JSON)
    assert "A_grid" in data, "A_grid key missing from JSON."
    A = data["A_grid"]
    assert isinstance(A, list), "A_grid must be a list."
    assert len(A) >= 2, f"A_grid must contain at least 2 points, found {len(A)}."

    # load expected N_A from params if available
    expected_n = None
    if PARAMS_JSON.exists():
        try:
            p = load_json(PARAMS_JSON)
            expected_n = int(p.get("N_A", p.get("N_states_A", None))) if p else None
        except Exception:
            expected_n = None

    if expected_n is not None:
        assert len(A) == expected_n, f"A_grid length {len(A)} != N_A from params ({expected_n})."

    # positivity and monotonicity
    assert all(isinstance(x, (int, float)) for x in A), "A_grid must contain numeric values only."
    assert all(x > 0 for x in A), "All A_grid entries must be positive."
    assert all(A[i] < A[i+1] for i in range(len(A)-1)), "A_grid must be strictly increasing."

    # plausible bounds (tunable): reject wildly tiny or huge values that usually indicate bad params
    min_allowed = 0.001   # anything below 0.001 is suspicious for this model
    max_allowed = 1e4     # anything above 10000 is suspicious; tune if you expect huge shocks
    amin = min(A)
    amax = max(A)
    assert amin >= min_allowed, f"A_grid minimum {amin:.6g} smaller than allowed {min_allowed}."
    assert amax <= max_allowed, f"A_grid maximum {amax:.6g} larger than allowed {max_allowed}."

def test_transition_matrix_stochastic():
    data = load_json(DATA_JSON)
    assert "P_A" in data, "P_A key missing from JSON."
    P = data["P_A"]
    A = data["A_grid"]
    n = len(A)
    assert isinstance(P, list) and len(P) == n, "P_A must be a list of length equal to A_grid."
    for i, row in enumerate(P):
        assert isinstance(row, list), f"P_A row {i} is not a list."
        assert len(row) == n, f"P_A row {i} length {len(row)} != {n}."
        # convert row entries to floats and sum
        s = sum(float(x) for x in row)
        assert math.isfinite(s), f"P_A row {i} sums to non-finite value {s}."
        assert abs(s - 1.0) < 1e-8, f"P_A row {i} sums to {s} (not 1) within tolerance."

def test_z_grid_consistency_if_present():
    data = load_json(DATA_JSON)
    if "z_grid" in data:
        z = data["z_grid"]
        A = data["A_grid"]
        assert len(z) == len(A), "z_grid and A_grid length mismatch."
        # compare elementwise exp(z) to A
        for zi, ai in zip(z, A):
            # allow relative tolerance because exp can amplify tiny differences
            ai_from_z = math.exp(float(zi))
            # use relative tolerance
            rel_err = abs(ai_from_z - float(ai)) / max(abs(ai_from_z), 1e-12)
            assert rel_err < 1e-9, f"exp(z) and A_grid differ: exp({zi})={ai_from_z}, A={ai}, rel_err={rel_err}"

