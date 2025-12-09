# modules/grids.py
"""
Module C — Capital grid construction for ProblemSet8.

This file provides `build_capital_grid(params, spacing="linear", power=3.0, save_path="./data/K_grid.json")`
which builds a capital grid K with N_K points between K_min and K_max.

DEFAULT BEHAVIOR:
  - function returns a numpy array of length N_K and optionally writes a JSON record
    to save_path containing "K_grid" and "metadata".

Notes:
 - The function performs sanity checks (N_K >= 2, K_max > K_min) and ensures no duplicate points.
"""
from pathlib import Path
import json
from typing import Dict, Any, Optional
import numpy as np

def build_capital_grid(params: Dict[str, Any],
                       spacing: str = "linear",
                       power: float = 3.0,
                       save_path: Optional[str] = "./data/K_grid.json"):
    """
    Build and optionally save the capital grid.

    Parameters
    ----------
    params : dict
        Must contain keys 'N_K', 'K_min', 'K_max'. Other keys are ignored.
    spacing : str, optional
        One of {"linear", "power", "log"}. Default "linear".
    power : float, optional
        Parameter used only if spacing == "power". Kept for compatibility.
    save_path : str or None, optional
        Relative path to write JSON snapshot. If None, do not save.

    Returns
    -------
    K_grid : numpy.ndarray
        1-D array of length N_K (dtype float64), strictly increasing.
    """
    # read required params
    try:
        N_K = int(params["N_K"])
        K_min = float(params["K_min"])
        K_max = float(params["K_max"])
    except KeyError as e:
        raise KeyError(f"Missing required grid parameter: {e}")

    if N_K < 2:
        raise ValueError("N_K must be >= 2")
    if K_max <= K_min:
        raise ValueError("K_max must be strictly greater than K_min")

    spacing = spacing.lower()
    if spacing not in {"linear", "power", "log"}:
        raise ValueError(f"Unknown spacing '{spacing}'. Choose 'linear','power' or 'log'.")

    # build grid
    if spacing == "linear":
        K_grid = np.linspace(K_min, K_max, N_K, dtype=float)
    elif spacing == "power":
        # kept for compatibility but not default
        u = np.linspace(0.0, 1.0, N_K)
        K_grid = K_min + (u ** power) * (K_max - K_min)
    else:  # spacing == "log"
        # logspace between K_min and K_max (requires K_min>0 ideally)
        if K_min <= 0:
            # shift by a tiny epsilon to create a meaningful log grid
            eps = 1e-12
            K_min_eff = max(K_min + eps, eps)
        else:
            K_min_eff = K_min
        # generate on log scale then set first to original K_min if it was zero
        log_min = np.log(K_min_eff)
        log_max = np.log(K_max)
        K_grid = np.exp(np.linspace(log_min, log_max, N_K, dtype=float))
        if K_min == 0:
            K_grid[0] = 0.0

    # final safety checks
    # ensure strictly increasing (allow tiny numerical eps)
    diffs = np.diff(K_grid)
    if np.any(diffs <= 0):
        raise RuntimeError("Generated K_grid is not strictly increasing. Check parameters.")
    # check for duplicates (within numerical tolerance)
    if np.unique(K_grid).size != K_grid.size:
        raise RuntimeError("K_grid has duplicate entries; adjust N_K, K_min/K_max, or spacing.")

    # optionally save human-readable JSON
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "K_grid": K_grid.tolist(),
            "metadata": {
                "N_K": N_K,
                "K_min": K_min,
                "K_max": K_max,
                "spacing": spacing,
                "power": power
            }
        }
        with p.open("w", encoding="utf8") as f:
            json.dump(out, f, indent=2)

    return K_grid

# If run directly, regenerate and save to default path
if __name__ == "__main__":
    import sys
    # attempt to find params from the common data file
    params_path = Path("./data/params_default.json")
    if params_path.exists():
        with params_path.open("r", encoding="utf8") as f:
            try:
                params = json.load(f)
            except Exception:
                params = {}
    else:
        # fallback minimal defaults
        params = {"N_K": 600, "K_min": 1e-6, "K_max": 50.0}

    K = build_capital_grid(params, spacing="linear", power=3.0, save_path="./data/K_grid.json")
    print(f"Constructed K grid length: {len(K)}")
    print("K_grid (first 8):", K[:8].tolist())
    print("K_grid (last 8):", K[-8:].tolist())
