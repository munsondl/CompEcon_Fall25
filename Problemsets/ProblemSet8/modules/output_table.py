# modules/output_table.py
"""
Module L (data-only): Output tables for Table 3 replication.

This module DOES NOT produce LaTeX. It:
 - loads optimizer output and (optional) std-error files from relative paths
 - constructs two pandas DataFrames: param_df and moments_df (Table 3 content)
 - saves those DataFrames to ./results/table3_params.csv and ./results/table3_moments.csv
 - also saves a compact JSON with both objects at ./results/table3.json
 - returns the DataFrames for further use.

Usage (from project root):
>>> from modules.output_table import make_table3_data
>>> out = make_table3_data(opt_json_path="./results/opt_result.json",
                           moments_json_path=None,
                           std_errors_json_path="./results/smm_std_errors.json",
                           out_dir="./results/")
>>> out['param_df']   # pandas DataFrame
>>> out['moments_df'] # pandas DataFrame
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import numpy as np
import pandas as pd

# --------------------
# Helpers
# --------------------
def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with p.open("r", encoding="utf8") as f:
        return json.load(f)

def _try_load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return _load_json(path)

# --------------------
# Build tables
# --------------------
def build_param_df(local_result: Dict[str, Any], std_errs: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Build parameter DataFrame from local_result dict (must contain 'x' and mapping).
    Returns DataFrame with columns ['parameter','estimate','std_error'].
    """
    if "x" in local_result:
        theta_vals = np.asarray(local_result["x"], dtype=float)
    elif "theta" in local_result:
        theta_vals = np.asarray(local_result["theta"], dtype=float)
    else:
        raise KeyError("local_result must contain 'x' or 'theta' with parameter vector.")

    mapping = local_result.get("mapping", None)
    if mapping is None:
        # try top-level mapping if present
        mapping = local_result.get("_mapping", None)
    if mapping is None:
        raise KeyError("local_result must include 'mapping' (list of parameter names).")

    if len(mapping) != theta_vals.size:
        raise ValueError("Length of mapping does not match parameter vector length.")

    # align std errors if provided
    se_list = []
    if std_errs is not None:
        for name in mapping:
            se_list.append(float(std_errs.get(name, np.nan)))
    else:
        se_list = [np.nan] * len(mapping)

    df = pd.DataFrame({
        "parameter": list(mapping),
        "estimate": theta_vals,
        "std_error": se_list
    })
    return df

def build_moments_df(moments_sim: Dict[str, float]) -> pd.DataFrame:
    """
    Build moments DataFrame with two rows: target and model(simulated).
    Columns order: a1, a2, sc_IK, std_pi_K, mean_q, ext_frac
    """
    order = ["a1","a2","sc_IK","std_pi_K","mean_q","ext_frac"]
    # default paper targets (costly external finance case)
    targets = {"a1":0.03, "a2":0.24, "sc_IK":0.4, "std_pi_K":0.25, "mean_q":3.0, "ext_frac":0.25}
    target_row = [targets[k] for k in order]
    sim_row = [float(moments_sim.get(k, np.nan)) for k in order]
    df = pd.DataFrame([target_row, sim_row], index=["Moments (target)","Model (simulated)"], columns=order)
    return df

# --------------------
# Main convenience function
# --------------------
def make_table3_data(opt_json_path: str = "./results/opt_result.json",
                     moments_json_path: Optional[str] = None,
                     std_errors_json_path: Optional[str] = "./results/smm_std_errors.json",
                     out_dir: str = "./results/") -> Dict[str, Any]:
    """
    Load results and produce CSV/JSON with Table 3 data.

    Parameters:
      opt_json_path : relative path to optimizer output JSON (optimize_smm output). Required.
      moments_json_path : optional path to JSON containing simulated moments; if None, looks inside opt JSON.
      std_errors_json_path : optional path to JSON containing std errors (mapping + list or dict)
      out_dir : relative directory where CSV/JSON outputs are written (default ./results/)

    Returns:
      dict containing 'param_df' (DataFrame), 'moments_df' (DataFrame), 'json_path', 'csv_paths'
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Load opt result
    opt = _load_json(opt_json_path)
    local_result = opt.get("local_result", opt)  # allow top-level storage
    # ensure mapping is available in local_result
    if "mapping" not in local_result and "mapping" in opt:
        local_result["mapping"] = opt["mapping"]

    # Load moments: prefer explicit file, else inside opt JSON
    moments = None
    if moments_json_path is not None:
        mm = _try_load_json(moments_json_path)
        if mm is None:
            raise FileNotFoundError(f"moments_json_path provided but file not found: {moments_json_path}")
        moments = mm.get("moments_sim", mm)
    else:
        moments = opt.get("moments_sim", None)
        if moments is None:
            moments = local_result.get("moments_sim", None)
        if moments is None:
            raise KeyError("Simulated moments not found in either moments_json_path or opt JSON (opt_result).")

    # Load std errors if any
    stds = None
    std_json = _try_load_json(std_errors_json_path)
    if std_json is not None:
        # several possible formats: {'std_err': [list], 'mapping': [...]}, or {'param': se, ...}
        if "std_err" in std_json and isinstance(std_json["std_err"], list):
            map_from_std = std_json.get("mapping", local_result.get("mapping", None))
            if map_from_std is None:
                # cannot align list to names -> skip
                stds = None
            else:
                stds = {name: float(val) for name, val in zip(map_from_std, std_json["std_err"])}
        else:
            # try treat file as name->value mapping
            if all(isinstance(v, (int, float)) for v in std_json.values()):
                stds = {k: float(v) for k, v in std_json.items()}

    # Build DataFrames
    param_df = build_param_df(local_result, std_errs=stds)
    moments_df = build_moments_df(moments)

    # Save CSVs
    params_csv = out_dir_p / "table3_params.csv"
    moments_csv = out_dir_p / "table3_moments.csv"
    param_df.to_csv(params_csv, index=False)
    moments_df.to_csv(moments_csv, index=True)

    # Save combined JSON for easy programmatic consumption later
    combined = {
        "parameters": {
            "names": param_df["parameter"].tolist(),
            "estimates": param_df["estimate"].tolist(),
            "std_errors": param_df["std_error"].tolist()
        },
        "moments": {
            "columns": list(moments_df.columns),
            "rows": {
                str(idx): moments_df.loc[idx].tolist() for idx in moments_df.index
            }
        }
    }
    json_path = out_dir_p / "table3.json"
    with json_path.open("w", encoding="utf8") as f:
        json.dump(combined, f, indent=2)

    return {
        "param_df": param_df,
        "moments_df": moments_df,
        "csv_paths": {"params": str(params_csv), "moments": str(moments_csv)},
        "json_path": str(json_path)
    }
