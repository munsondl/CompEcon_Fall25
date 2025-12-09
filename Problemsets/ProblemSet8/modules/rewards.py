# modules/rewards.py
"""
Module D — profit & adjustment cost utilities.

Functions:
  - profit(K, A, alpha)
  - adjustment_cost(Kp, K, gamma, delta)
  - net_investment(Kp, K, delta)
  - net_return(Kp, K, A, alpha, gamma, delta, p=1.0)

Convenience saver:
  - save_array_with_metadata(arrays_dict, metadata, path="./data/rewards_snapshot.json")
  - save_rewards_snapshot(Kp, K, A, alpha, gamma, delta, p=1.0, path=None)

Behavior:
  - The economic functions behave purely (no I/O).
  - Call save_rewards_snapshot(...) to compute arrays and save them to a human-readable JSON
    stored in the project's ./data/ folder by default. The function returns the absolute path.
"""
from pathlib import Path
import json
from typing import Dict, Any, Optional, Tuple

import numpy as np

# -------------------------
# Economic functions
# -------------------------
def profit(K, A, alpha):
    """
    Profit: pi(K,A) = A * K^alpha
    K, A may be scalars or numpy arrays (broadcasting supported).
    Returns numpy array.
    """
    K_arr = np.asarray(K, dtype=float)
    A_arr = np.asarray(A, dtype=float)
    return A_arr * (K_arr ** float(alpha))

def net_investment(Kp, K, delta):
    """
    Net investment: I = K' - (1-delta)*K
    Supports array broadcasting.
    """
    Kp_arr = np.asarray(Kp, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    return Kp_arr - (1.0 - float(delta)) * K_arr

def adjustment_cost(Kp, K, gamma, delta):
    """
    Quadratic adjustment cost:
      C(K',K) = gamma/2 * ( (K' - (1-delta)K) / K )^2 * K
    If any K entry is zero, treat adjustment expression safely (result is defined by limit).
    """
    Kp_arr = np.asarray(Kp, dtype=float)
    K_arr = np.asarray(K, dtype=float)

    # compute net investment I
    I = net_investment(Kp_arr, K_arr, delta)

    # avoid division by zero; define cost = (gamma/2) * (I^2 / K) when K>0
    C = np.zeros_like(I, dtype=float)

    positive_mask = K_arr > 0
    if np.any(positive_mask):
        C[positive_mask] = 0.5 * float(gamma) * ( (I[positive_mask] ** 2) / K_arr[positive_mask] )

    # For K == 0, we use the continuous-limit form: C = 0.5 * gamma * (I^2 / eps)
    # but to avoid numerical blowup, set C = 0.5*gamma*(I^2)/(1e-12) if needed.
    zero_mask = ~positive_mask
    if np.any(zero_mask):
        eps = 1e-12
        C[zero_mask] = 0.5 * float(gamma) * ( (I[zero_mask] ** 2) / eps )

    return C

def net_return(Kp, K, A, alpha, gamma, delta, p=1.0):
    """
    Net return (dividends) D = pi(K,A) - p * I - C(K',K).
    Returns numpy array (same shape as broadcast of inputs).
    """
    pi = profit(K, A, alpha)
    I = net_investment(Kp, K, delta)
    C = adjustment_cost(Kp, K, gamma, delta)
    return pi - float(p) * I - C

# -------------------------
# I/O helpers: save arrays + metadata (default folder ./data)
# -------------------------
def save_array_with_metadata(arrays: Dict[str, Any],
                             metadata: Dict[str, Any],
                             path: Optional[str] = "./data/rewards_snapshot.json") -> str:
    """
    Save arrays (numpy arrays or lists) and metadata into a human-readable JSON.
    Default path is './data/rewards_snapshot.json' relative to project root.

    Returns the absolute path string of the saved file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    serial = {}
    for key, val in arrays.items():
        # convert numpy arrays to python lists
        if isinstance(val, np.ndarray):
            serial[key] = val.tolist()
        else:
            # try basic conversion
            try:
                serial[key] = list(val)
            except Exception:
                serial[key] = val

    out = {
        "arrays": serial,
        "metadata": metadata
    }
    with p.open("w", encoding="utf8") as f:
        json.dump(out, f, indent=2)

    return str(p.resolve())

def save_rewards_snapshot(Kp, K, A, alpha, gamma, delta, p=1.0,
                          path: Optional[str] = None) -> str:
    """
    Convenience function: computes pi, I, C, D for given inputs and saves them to JSON.

    Parameters:
      - Kp, K, A: arrays or scalars (broadcasted)
      - alpha, gamma, delta, p: model parameters
      - path: optional relative path. If None, defaults to './data/rewards_snapshot.json'

    Returns:
      - absolute path to the saved JSON file
    """
    if path is None:
        path = "./data/rewards_snapshot.json"

    # ensure numpy arrays
    Kp_arr = np.asarray(Kp, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    A_arr = np.asarray(A, dtype=float)

    pi_vals = profit(K_arr, A_arr, alpha)
    I_vals = net_investment(Kp_arr, K_arr, delta)
    C_vals = adjustment_cost(Kp_arr, K_arr, gamma, delta)
    D_vals = net_return(Kp_arr, K_arr, A_arr, alpha, gamma, delta, p)

    arrays = {
        "K": K_arr,
        "Kp": Kp_arr,
        "A": A_arr,
        "pi": pi_vals,
        "I": I_vals,
        "C": C_vals,
        "D": D_vals
    }
    metadata = {
        "alpha": float(alpha),
        "gamma": float(gamma),
        "delta": float(delta),
        "p": float(p),
        "len": int(K_arr.size)
    }

    abs_path = save_array_with_metadata(arrays, metadata, path=path)
    return abs_path

# -------------------------
# Demo / script behavior
# -------------------------
if __name__ == "__main__":
    # If module run directly, create a small default snapshot and save to ./data/
    import numpy as _np

    alpha = 0.6956
    gamma = 0.1331
    delta = 0.15
    p = 1.0

    # small example grid for demonstration
    K = _np.linspace(1.0, 10.0, 10)
    Kp = (1.0 - delta) * K + 0.5  # simple positive investment
    A = _np.ones_like(K)

    saved = save_rewards_snapshot(Kp, K, A, alpha, gamma, delta, p=p, path="./data/rewards_snapshot.json")
    print("Saved rewards snapshot to:", saved)
