# tests/test_grids.py
from pathlib import Path
import json
import numpy as np
import math
import pytest

PROJECT_ROOT = Path(".").resolve()
DATA_JSON = PROJECT_ROOT / "data" / "K_grid.json"

def load_json(path):
    with path.open("r", encoding="utf8") as f:
        return json.load(f)

def test_build_and_save_grid_exists():
    assert DATA_JSON.exists(), f"Expected K_grid JSON at {DATA_JSON.resolve()} but file not found."

def test_k_grid_structure():
    data = load_json(DATA_JSON)
    assert "K_grid" in data, "K_grid key missing from JSON."
    K = data["K_grid"]
    assert isinstance(K, list), "K_grid must be a list."
    assert len(K) >= 2, f"K_grid must have at least 2 points, found {len(K)}"
    assert all(isinstance(x, (int,float)) for x in K)
    assert all(K[i] < K[i+1] for i in range(len(K)-1)), "K_grid must be strictly increasing."

def test_metadata_consistency_with_params():
    data = load_json(DATA_JSON)
    meta = data.get("metadata", {})
    # If parameters file exists in data, check N_K
    params_path = PROJECT_ROOT / "data" / "params_default.json"
    if params_path.exists():
        p = json.load(params_path.open("r", encoding="utf8"))
        expected = int(p.get("N_K", p.get("N_K", 0)))
        if expected > 0:
            assert meta.get("N_K") == expected, f"metadata N_K {meta.get('N_K')} != params N_K {expected}"
