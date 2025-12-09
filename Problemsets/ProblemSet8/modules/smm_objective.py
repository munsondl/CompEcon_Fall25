# modules/smm_objective.py
"""
SMM objective and moment utilities (Module I)

Provides:
 - moment_targets() : target moments from Cooper & Ejarque (costly external finance case)
 - compute_moments_from_panel(df, params) : compute the six simulated moments from a panel
 - compute_moments_from_policy(...) : wrapper that either uses stationary-dist evaluation (fast)
   or runs a panel simulation and computes moments from the simulated panel
 - smm_objective(theta, mapping, params_base, W=None, use_panel=False, ...) :
   computes J(theta) = (Psi_d - Psi_s(theta))' W (Psi_d - Psi_s(theta))

Notes
 - Default target moments and ordering: [a1, a2, sc_IK, std_pi_K, mean_q, ext_frac].
 - The ordering of theta is provided by `mapping` (list of strings). Example mapping:
       ["alpha","gamma","rho","sigma","phi_tilde_0"]
 - Relative paths only for any saved outputs.
 - Citation: Cooper & Ejarque (2003) and the PS8 assignment (targets & procedure). 
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd

# imports from your modules (assumed available in your project)
from modules.parameters import init_params
from modules.tauchen import discretize_A_from_params
from modules.grids import build_capital_grid
from modules.bellman import solve_bellman
from modules.simulate_panel import simulate_panel
from modules.policy_eval import evaluate_policy_states

# ---------------------------------------------------------------------
# Target moments (from paper / PS8)
# ordering: [a1, a2, sc_IK, std_pi_K, mean_q, ext_frac]
# ---------------------------------------------------------------------
def moment_targets() -> np.ndarray:
    """
    Return the target moments for the costly external finance estimation (Table 3).
    Order: [a1, a2, sc(I/K), std(pi/K), mean_q, ext_frac]
    Source: Cooper & Ejarque (2003) Table 3 and the Problem Set instructions. 
    """
    # paper targets for the costly case: (a1=0.03, a2=0.24, sc(I/K)=0.4, std(pi/K)=0.25, mean_q=3.0, ext_frac=0.25)
    targets = np.array([0.03, 0.24, 0.4, 0.25, 3.0, 0.25], dtype=float)
    return targets

# ---------------------------------------------------------------------
# Compute moments from simulated panel (pandas DataFrame)
# ---------------------------------------------------------------------
def compute_moments_from_panel(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute the 6 moments from a simulated panel (DataFrame format created by simulate_panel)
    Expects columns: ['firm_id','t','K','Kp','I','pi','q','E','ext_flag']
    Returns dict with keys: 'a1','a2','sc_IK','std_pi_K','mean_q','ext_frac'
    """
    # Defensive checks
    required = {"firm_id","t","K","Kp","I","pi","q","E","ext_flag"}
    if not required.issubset(set(df.columns)):
        raise KeyError(f"Input DataFrame missing required columns: {required - set(df.columns)}")

    # compute IK and piK, drop rows where K==0 (or use small epsilon)
    eps = 1e-12
    df = df.copy()
    df["IK"] = df["I"] / df["K"].replace(0, eps)
    df["piK"] = df["pi"] / df["K"].replace(0, eps)

    # a1 and a2 regression: IK_t = a0 + a1 * q_{t+1} + a2 * piK_t + error
    # need q_{t+1} for each observation: merge shifted q by firm
    df_sorted = df.sort_values(["firm_id","t"])
    df_sorted["q_next"] = df_sorted.groupby("firm_id")["q"].shift(-1)
    # drop the last observation per firm where q_next is NaN
    reg_df = df_sorted.dropna(subset=["q_next", "IK", "piK"])
    if reg_df.shape[0] < 10:
        # fallback: use stationary evaluation / raise
        raise RuntimeError("Not enough observations to estimate Q-regression from panel")

    # Weighted OLS: here we use simple OLS (unweighted) as in many panel studies,
    # but the paper uses cross-section/regression exactly on panel — this is a standard approach.
    X = np.column_stack([np.ones(len(reg_df)), reg_df["q_next"].values, reg_df["piK"].values])
    y = reg_df["IK"].values
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    a0, a1, a2 = float(beta_hat[0]), float(beta_hat[1]), float(beta_hat[2])

    # sc_IK: serial correlation of IK computed across all firm-time (lag 1 within firm)
    reg_df2 = df_sorted.copy()
    reg_df2["IK_next"] = reg_df2.groupby("firm_id")["IK"].shift(-1)
    sc_df = reg_df2.dropna(subset=["IK","IK_next"])
    if sc_df.shape[0] < 10:
        sc_IK = 0.0
    else:
        sc_IK = float(np.corrcoef(sc_df["IK"].values, sc_df["IK_next"].values)[0,1])

    # std_pi_K: standard deviation of pi/K across all observations (use cross-sectional panel)
    std_pi_K = float(df["piK"].std(ddof=0))

    # mean_q: mean of q across panel
    mean_q = float(df["q"].mean())

    # ext_frac: fraction of total investment financed externally:
    # ext_frac = sum(E) / sum(I_positive) (use positive investments only)
    E_total = float(df["E"].sum())
    Ipos = df["I"].copy()
    Ipos = Ipos.where(Ipos > 1e-12, 0.0)
    Ipos_total = float(Ipos.sum())
    ext_frac = float(E_total / Ipos_total) if Ipos_total > 0 else 0.0

    moments = {
        "a1": a1,
        "a2": a2,
        "sc_IK": sc_IK,
        "std_pi_K": std_pi_K,
        "mean_q": mean_q,
        "ext_frac": ext_frac
    }
    return moments

# ---------------------------------------------------------------------
# Wrapper: compute moments given current policy / V or by simulating a panel
# ---------------------------------------------------------------------
def compute_moments_from_policy(policy_idx: np.ndarray,
                                V: np.ndarray,
                                ext_flag: np.ndarray,
                                K_grid: np.ndarray,
                                A_grid: np.ndarray,
                                P_A: np.ndarray,
                                params: Dict[str, Any],
                                use_panel: bool = False,
                                panel_args: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute simulated moments either from stationary policy evaluation (fast)
    or by running a panel simulation and computing moments from the panel (slower).

    Returns:
      - moments_vec: np.array ordered as [a1,a2,sc_IK,std_pi_K,mean_q,ext_frac]
      - moments_dict: dict keyed by names (same as compute_moments_from_panel)
    """
    if not use_panel:
        # Fast stationary evaluation using evaluate_policy_states (Module F)
        state_dict, moments = evaluate_policy_states(policy_idx, V, ext_flag,
                                                     K_grid, A_grid, P_A, params,
                                                     save_path=None)
        # evaluate_policy_states already returns the same moment names (a0,a1,a2,...). Map accordingly
        # moments has keys: 'a0','a1','a2','sc_IK','std_pi_K','mean_q','ext_frac'
        mvec = np.array([moments["a1"], moments["a2"], moments["sc_IK"],
                         moments["std_pi_K"], moments["mean_q"], moments["ext_frac"]], dtype=float)
        return mvec, {k: moments[k] for k in ["a1","a2","sc_IK","std_pi_K","mean_q","ext_frac"]}
    else:
        # Run panel simulation (Module G) then compute moments from panel (slower, exact)
        panel_args = panel_args or {}
        # ensure defaults
        pf = {
            "N_firms": int(params.get("N_firms", 1000)),
            "T": int(params.get("T", 50)),
            "burn_in": int(params.get("burn_in", 50)),
            "seed": panel_args.get("seed", 2025),
            "init_method": panel_args.get("init_method", "stationary"),
            "save_path_csv": panel_args.get("save_path_csv", "./data/sim_panel.csv"),
            "save_path_npz": panel_args.get("save_path_npz", "./data/sim_panel.npz")
        }
        # call simulate_panel: returns pandas DataFrame
        df = simulate_panel(policy_idx, V, ext_flag, K_grid, A_grid, P_A, params,
                            N_firms=pf["N_firms"], T=pf["T"], burn_in=pf["burn_in"],
                            init_method=pf["init_method"], seed=pf["seed"],
                            save_path_csv=pf["save_path_csv"], save_path_npz=pf["save_path_npz"])
        # compute moments from df
        moments_dict = compute_moments_from_panel(df, params)
        mvec = np.array([moments_dict["a1"], moments_dict["a2"], moments_dict["sc_IK"],
                         moments_dict["std_pi_K"], moments_dict["mean_q"], moments_dict["ext_frac"]], dtype=float)
        return mvec, moments_dict

# ---------------------------------------------------------------------
# Objective: J(theta)
# ---------------------------------------------------------------------
def smm_objective(theta: Union[np.ndarray, List[float]],
                  mapping: List[str],
                  params_base: Dict[str, Any],
                  W: Optional[Union[np.ndarray, str]] = None,
                  use_panel: bool = False,
                  panel_args: Optional[Dict[str, Any]] = None,
                  solver_opts: Optional[Dict[str, Any]] = None,
                  save_results: Optional[str] = None
                  ) -> Tuple[float, Dict[str, Any]]:
    """
    Compute SMM objective J(theta) for given parameter vector theta.

    Parameters
    ----------
    theta : array-like
        Values for parameters in the order specified by mapping.
    mapping : list[str]
        Names of parameters to override in params_base (e.g. ["alpha","gamma","rho","sigma","phi_tilde_0"])
    params_base : dict
        Base params dict (from modules.parameters.init_params()). theta entries override these keys.
    W : None, ndarray (6x6) or str (path)
        Weighting matrix. If None, identity is used. If str, attempts to load as JSON or .npy file.
    use_panel : bool
        If True, run panel simulation to compute moments; else use fast stationary evaluation.
    panel_args : dict
        Passed into simulate_panel if use_panel True.
    solver_opts : dict
        Options for the bellman solver (tol, maxiter, enforce_internal_constraint, etc.)
    save_results : str or None
        If provided, save a JSON with theta, moments_sim, J, moment_diff at this relative path.

    Returns
    -------
    J : float
    info : dict containing {
        "theta_dict", "moments_sim" (dict), "moment_diff" (numpy), "J": float
    }
    """
    # convert theta -> params
    theta = np.asarray(theta, dtype=float)
    if theta.size != len(mapping):
        raise ValueError("theta length must match mapping length")

    params = params_base.copy()
    theta_dict = {}
    for name, val in zip(mapping, theta):
        params[name] = float(val)
        theta_dict[name] = float(val)

    # Build grids and solve DP (use existing modules)
    # Use solver_opts or defaults from params
    solver_opts = solver_opts or {}
    tol = solver_opts.get("tol", params.get("vfi_tol", 1e-6))
    maxiter = solver_opts.get("maxiter", params.get("vfi_maxiter", 2000))
    enforce_internal = solver_opts.get("enforce_internal_constraint", False)

    # Discretize A
    A_grid, P_A = discretize_A_from_params(params, save_path="./data/A_grid.json", m=params.get("tauchen_m", 3.0))
    # Build K grid
    K_grid = build_capital_grid(params, spacing=solver_opts.get("K_spacing","power"),
                                power=solver_opts.get("K_power", 3.0), save_path="./data/K_grid.json")
    # Solve Bellman
    V, policy_idx, ext_flag, q = solve_bellman(params, K_grid, A_grid, np.array(P_A),
                                               enforce_internal_constraint=enforce_internal,
                                               tol=tol, maxiter=maxiter, compute_q=True,
                                               save_path=None)

    # Compute simulated moments (fast stationary or panel)
    moments_vec, moments_dict = compute_moments_from_policy(policy_idx, V, ext_flag,
                                                            K_grid, A_grid, np.array(P_A), params,
                                                            use_panel=use_panel, panel_args=panel_args)

    # Load or build weighting matrix W (6x6)
    if W is None:
        Wmat = np.eye(len(moments_vec))
    elif isinstance(W, str):
        # attempt to load as json or numpy
        p = Path(W)
        if p.suffix == ".json":
            with p.open("r", encoding="utf8") as f:
                Wmat = np.array(json.load(f), dtype=float)
        else:
            Wmat = np.load(W)
    else:
        Wmat = np.asarray(W, dtype=float)

    # ensure Wmat shape
    qn = moments_vec.size
    if Wmat.shape != (qn, qn):
        raise ValueError(f"Weight matrix W must be shape ({qn},{qn}), got {Wmat.shape}")

    # Target moments
    target = moment_targets()
    if target.size != qn:
        raise ValueError("Target moment vector size mismatch")

    diff = target - moments_vec
    # Quadratic objective
    J = float(diff.T @ Wmat @ diff)

    info = {
        "theta_dict": theta_dict,
        "moments_sim": {k: float(v) for k, v in moments_dict.items()},
        "moment_diff": diff,
        "J": J
    }

    # optional save
    if save_results is not None:
        outp = {
            "theta": theta_dict,
            "targets": target.tolist(),
            "moments_sim": info["moments_sim"],
            "moment_diff": diff.tolist(),
            "J": J
        }
        p = Path(save_results)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf8") as f:
            json.dump(outp, f, indent=2)

    return J, info

# ---------------------------------------------------------------------
# Convenience: small helper to compute numeric jacobian of moments
# ---------------------------------------------------------------------
def numeric_moment_jacobian(theta: Union[np.ndarray, List[float]],
                            mapping: List[str],
                            params_base: Dict[str, Any],
                            eps: float = 1e-4,
                            **smm_kwargs) -> Tuple[np.ndarray, List[str]]:
    """
    Numerically approximate ∂Ψ_s(θ)/∂θ using central differences.
    Returns jac (q x p) and mapping list.
    Note: expensive since each column requires two objective evaluations (or moment evaluations).
    """
    theta = np.asarray(theta, dtype=float)
    p = theta.size
    # base evaluation to get q
    _, base_info = smm_objective(theta, mapping, params_base, **smm_kwargs)
    base_moments = np.array([base_info["moments_sim"][k] for k in ["a1","a2","sc_IK","std_pi_K","mean_q","ext_frac"]])
    q = base_moments.size
    jac = np.zeros((q, p), dtype=float)

    for j in range(p):
        d = np.zeros_like(theta)
        d[j] = eps
        # plus
        thp = theta + d
        _, info_p = smm_objective(thp, mapping, params_base, **smm_kwargs)
        m_p = np.array([info_p["moments_sim"][k] for k in ["a1","a2","sc_IK","std_pi_K","mean_q","ext_frac"]])
        # minus
        thm = theta - d
        _, info_m = smm_objective(thm, mapping, params_base, **smm_kwargs)
        m_m = np.array([info_m["moments_sim"][k] for k in ["a1","a2","sc_IK","std_pi_K","mean_q","ext_frac"]])
        jac[:, j] = (m_p - m_m) / (2.0 * eps)

    return jac, mapping
