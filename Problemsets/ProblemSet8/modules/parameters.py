# modules/parameters.py
"""
Module A — parameter initialization utilities for ProblemSet8.

This file:
 - provides init_params() to build the default params dict
 - provides save_params() to write a readable JSON at a relative path (default ./data/params_default.json)
 - provides load_params() to read it back
 - provides dump_and_show() which saves the JSON and prints the absolute filesystem path
   (useful to confirm where the file was written in Windows Explorer)

All paths are relative to the working directory when you run Python. Make sure you `cd`
to the project root (the folder that contains `modules/`) before running the test commands.
"""
from pathlib import Path
import json
from typing import Dict, Any

def init_params(**overrides) -> Dict[str, Any]:
    """
    Return default parameters (cooper & ejarque, costly finance case).
    Pass keyword overrides to change any default.
    """
    params: Dict[str, Any] = {
        "alpha": 0.6956,
        "gamma": 0.1331,
        "rho": 0.0976,
        "sigma": 0.8932,
        "phi_tilde_0": 0.0,
        "delta": 0.15,
        "beta": 0.95,
        "p": 1.0,
        "phi1": 0.0,
        "N_K": 600,
        "K_min": 1e-6,
        "K_max": 50.0,
        "N_A": 7,
        "N_firms": 1000,
        "T": 50,
        "burn_in": 50,
        "moment_targets": {
            "a1": 0.03,
            "a2": 0.24,
            "sc_IK": 0.4,
            "std_pi_K": 0.25,
            "mean_q": 3.0,
            "ext_frac": 0.25
        },
        "vfi_tol": 1e-6,
        "vfi_maxiter": 2000,
        "random_seed": 2025
    }

    # apply overrides
    for k, v in overrides.items():
        if k in params:
            params[k] = v
        elif k.startswith("moment_"):
            # allow e.g. moment_a1=0.04
            mkey = k.replace("moment_", "")
            params.setdefault("moment_targets", {})[mkey] = v
        else:
            # accept arbitrary additional keys
            params[k] = v

    return params

# -------------------------
# Save / load helpers
# -------------------------
def save_params(params: Dict[str, Any], path: str = "./data/params_default.json") -> str:
    """
    Save params to a JSON file (relative path). Returns the absolute path string written.

    This function creates the parent folder if it doesn't exist and ensures the JSON is
    human-readable (indentation).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # make sure everything is JSON serializable (convert simple numpy scalars if present)
    def _serial(x):
        try:
            json.dumps(x)
            return x
        except TypeError:
            # fallback: convert via str or .item() if present
            if hasattr(x, "item"):
                return x.item()
            return str(x)

    serial = {}
    for k, v in params.items():
        if isinstance(v, dict):
            serial[k] = {kk: _serial(vv) for kk, vv in v.items()}
        else:
            serial[k] = _serial(v)

    with p.open("w", encoding="utf8") as f:
        json.dump(serial, f, indent=2)

    # return the absolute path so user can inspect it in Explorer easily
    return str(p.resolve())

def load_params(path: str = "./data/params_default.json") -> Dict[str, Any]:
    """
    Load params from a JSON file (relative path). Raises FileNotFoundError if missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parameter file not found: {p}")
    with p.open("r", encoding="utf8") as f:
        return json.load(f)

def dump_and_show(path: str = "./data/params_default.json", **overrides) -> str:
    """
    Convenience: build params (optionally with overrides), write JSON and print the absolute path.
    Returns the absolute path string.
    Example (PowerShell):
      cd path\to\ProblemSet8
      python -c "from modules.parameters import dump_and_show; print(dump_and_show())"
    """
    params = init_params(**overrides)
    abs_path = save_params(params, path=path)
    print(f"Parameters saved to: {abs_path}")
    return abs_path

# script usage for quick test
if __name__ == "__main__":
    abs_path = dump_and_show()
    print("Wrote params JSON to:", abs_path)
