# tests/test_rewards.py
"""
Unit tests for modules.rewards.

This test:
 - calls save_rewards_snapshot with small known inputs (arrays)
 - reads the JSON file written to ./data/rewards_snapshot.json
 - checks arrays (pi, I, C, D) are present and match in-memory calculations
"""

from pathlib import Path
import json
import math
import numpy as np
import os
import shutil

import pytest

from modules.rewards import (
    profit,
    adjustment_cost,
    net_investment,
    net_return,
    save_rewards_snapshot
)

DATA_PATH = Path("./data")
SNAP_PATH = DATA_PATH / "rewards_snapshot.json"

def remove_snapshot_if_exists():
    if SNAP_PATH.exists():
        SNAP_PATH.unlink()

def load_snapshot(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

def test_save_rewards_snapshot_and_values():
    # ensure clean start
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    remove_snapshot_if_exists()

    # small known arrays
    K = np.array([10.0, 5.0, 2.0])
    Kp = np.array([10.0, 6.0, 2.5])
    A = np.array([1.0, 2.0, 0.5])

    alpha = 0.7
    gamma = 0.1
    delta = 0.2
    p = 1.0

    # call saver (defaults to ./data/rewards_snapshot.json)
    saved_path = save_rewards_snapshot(Kp, K, A, alpha, gamma, delta, p=p, path=str(SNAP_PATH))
    assert Path(saved_path).exists(), f"Snapshot file not created at {saved_path}"

    # load JSON
    data = load_snapshot(saved_path)
    assert "arrays" in data and "metadata" in data

    arrays = data["arrays"]
    # check arrays exist
    for key in ["K", "Kp", "A", "pi", "I", "C", "D"]:
        assert key in arrays, f"{key} missing in saved arrays"

    # convert saved arrays to numpy arrays for comparison
    saved_K = np.array(arrays["K"], dtype=float)
    saved_Kp = np.array(arrays["Kp"], dtype=float)
    saved_A = np.array(arrays["A"], dtype=float)
    saved_pi = np.array(arrays["pi"], dtype=float)
    saved_I = np.array(arrays["I"], dtype=float)
    saved_C = np.array(arrays["C"], dtype=float)
    saved_D = np.array(arrays["D"], dtype=float)

    # compute in-memory expected values
    expected_pi = profit(saved_K, saved_A, alpha)
    expected_I = net_investment(saved_Kp, saved_K, delta)
    expected_C = adjustment_cost(saved_Kp, saved_K, gamma, delta)
    expected_D = net_return(saved_Kp, saved_K, saved_A, alpha, gamma, delta, p)

    # elementwise checks (allow small numerical tolerance)
    assert np.allclose(saved_pi, expected_pi, rtol=1e-12, atol=1e-12)
    assert np.allclose(saved_I, expected_I, rtol=1e-12, atol=1e-12)
    assert np.allclose(saved_C, expected_C, rtol=1e-12, atol=1e-12)
    assert np.allclose(saved_D, expected_D, rtol=1e-12, atol=1e-12)

    # cleanup (optional)
    remove_snapshot_if_exists()
