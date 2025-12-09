# modules/std_errors.py
"""
Module K: Standard errors for SMM estimates (optional)

Provides:
 - estimate_moment_covariance(theta, mapping, params_base, S_panels, panel_args)
     -> estimate covariance matrix of simulated moments (q x q) by simulating S independent panels
 - compute_jacobian(theta, mapping, params_base, eps, **smm_kwargs)
     -> numerical Jacobian ∂Ψ_s(θ)/∂θ (q x p) using central differences (wraps smm_objective.numeric_moment_jacobian)
 - compute_smm_standard_errors(theta, mapping, params_base, W, S_panels, panel_args,
                               eps=1e-4, use_general_sandwich=True, save_path=None, **smm_kwargs)
     -> returns cov_theta (p x p), std_err (p,), and diagnostic objects

Notes / Implementation choices
 - The simulated-moment covariance is computed by simulating S independent panels at θ and computing
   the sample covariance of the resulting moment vectors (shape q). Use S large enough (e.g. 200-1000).
 - The Jacobian is computed with central finite differences. This is computationally expensive since each
   column requires two moment evaluations (and each moment evaluation may re-solve the DP and simulate).
 - Two formulas are provided:
    * General sandwich (recommended if W is arbitrary):
         Var(θ_hat) ≈ (1 + 1/S_data) * (G' W G)^{-1} (G' W S_psi W G) (G' W G)^{-1}
      where S_psi is the (q x q) covariance of simulated moments (from S simulated panels),
      G is the (q x p) Jacobian, and S_data is number of independent panels used to compute data moments.
      In our assignment the "data" moments come from the paper (treated as exact), but the formula
      in the problem set includes a factor (1 + 1/S) — we include it as multiply_factor = (1 + 1/S_panels).
    * Efficient two-step shortcut (when W = S_psi^{-1}): 
         Var(θ_hat) ≈ multiply_factor * (G' W G)^{-1}
      which simplifies the algebra and is what Cooper & Ejarque use after they set W = S_psi^{-1}.
 - You can pass W=None, in which case the code will set W = I (identity) for the general sandwich,
   or automatically use the efficient choice W = S_psi^{-1} if use_general_sandwich=False.
 - All saved files use relative paths (e.g., ./results/ or ./data/).

Dependencies (make sure these are importable from your project root):
  - numpy, pandas, pathlib
  - modules.smm_objective (for smm_objective and numeric_moment_jacobian)
  - modules.simulate_panel (for panel sim if needed)
  - modules.smm_objective.compute_moments_from_policy or compute_moments_from_policy wrapper
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import time
import numpy as np

# imports from your project
from modules.smm_objective import compute_moments_from_policy, smm_objective, numeric_moment_jacobian
from modules.simulate_panel import simulate_panel

# ---------------------------------------------------------------------
# 1) Estimate covariance matrix of simulated moments by drawing S panels
# ---------------------------------------------------------------------
def estimate_moment_covariance(theta: np.ndarray,
                               mapping: list,
                               params_base: Dict[str, Any],
                               S_panels: int = 500,
                               panel_args: Optional[Dict[str, Any]] = None,
                               use_panel_simulator: bool = True,
                               save_path: Optional[str] = "./data/moment_covariance.npz"
                               ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Simulate S_panels independent panels at parameter theta and compute the covariance of the q-moment vectors.

    Parameters
    ----------
    theta, mapping, params_base : as in smm_objective
    S_panels : int
        number of independent simulated panels used to compute covariance
    panel_args : dict
        forwarded to simulate_panel (e.g., {"N_firms":1000,"T":50,"burn_in":50,"seed":2025})
    use_panel_simulator : bool
        If True, compute moments by running simulate_panel + compute_moments_from_panel (exact).
        If False, call compute_moments_from_policy with use_panel=False (fast stationary evaluation) for each draw.
        Note: using simulate_panel with different RNG seeds is slower but is the correct way to estimate panel sampling variance.
    save_path : relative path to save the covariance matrix and simulated moments (optional)

    Returns
    -------
    S_psi : (q,q) sample covariance matrix of simulated moments
    info : dict with 'moments_matrix' (S x q), 'mean_moments' (q,), 'S_panels' etc.
    """
    panel_args = panel_args or {}
    rng_seed = int(panel_args.get("seed", 2025))

    # Evaluate once to determine q dimension
    # use compute_moments_from_policy (fast) with use_panel = use_panel_simulator
    # This call will solve DP etc. — but we only need dimension q here.
    # We call smm_objective with small tweaks: use_panel=use_panel_simulator so that moments are computed the right way.
    J0, info0 = smm_objective(theta, mapping, params_base, use_panel=use_panel_simulator,
                              panel_args=panel_args, solver_opts=panel_args.get("solver_opts", None))
    # extract q order used in smm_objective: [a1,a2,sc_IK,std_pi_K,mean_q,ext_frac]
    moment_names = ["a1","a2","sc_IK","std_pi_K","mean_q","ext_frac"]
    q = len(moment_names)

    moments_matrix = np.zeros((S_panels, q), dtype=float)

    # Use different seeds for panel draws to ensure independence
    base_seed = rng_seed
    for s in range(S_panels):
        seed_s = base_seed + s + 1
        # If using full panel sim: run simulate_panel then compute moments from panel
        if use_panel_simulator:
            # Build panel via simulate_panel (it saves to disk by default; we set save paths to None to avoid I/O thrashing)
            pa = panel_args.copy()
            pa["seed"] = seed_s
            pa["save_path_csv"] = None
            pa["save_path_npz"] = None
            # We need policy and V at current theta to feed simulate_panel — compute them via smm_objective helper?
            # But compute_moments_from_policy with use_panel=True handles it internally by solving DP and simulating.
            mvec, mdict = compute_moments_from_policy_wrapper(theta, mapping, params_base, use_panel=True, panel_args=pa)
        else:
            # fast stationary evaluation (no panel) - still need unique randomness? not needed
            mvec, mdict = compute_moments_from_policy_wrapper(theta, mapping, params_base, use_panel=False)
        moments_matrix[s, :] = mvec

    mean_moments = moments_matrix.mean(axis=0)
    # unbiased sample covariance (rowvar=False, ddof=1)
    S_psi = np.cov(moments_matrix, rowvar=False, ddof=1)

    info = {
        "moments_matrix": moments_matrix,
        "mean_moments": mean_moments,
        "S_panels": S_panels,
        "moment_names": moment_names
    }

    # save if requested
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, moments_matrix=moments_matrix, mean_moments=mean_moments, S_psi=S_psi)

    return S_psi, info

# Helper wrapper to compute moments vector in consistent order
def compute_moments_from_policy_wrapper(theta: np.ndarray,
                                        mapping: list,
                                        params_base: Dict[str, Any],
                                        use_panel: bool = False,
                                        panel_args: Optional[Dict[str, Any]] = None
                                        ) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Solve model at theta and return moment vector mvec and dict mdict.
    This wrapper ensures we return moments in order: [a1,a2,sc_IK,std_pi_K,mean_q,ext_frac]
    """
    # update params_base with theta via smm_objective's pattern
    theta = np.asarray(theta, dtype=float)
    params = params_base.copy()
    for name, val in zip(mapping, theta):
        params[name] = float(val)

    # Build grids and solve DP inside compute_moments_from_policy called from smm_objective.compute_moments_from_policy
    # Use the compute_moments_from_policy function exported by modules.smm_objective (it was defined earlier)
    # We call the function through smm_objective.compute_moments_from_policy (imported at module top)
    mvec, mdict = compute_moments_from_policy_wrapper_inner(params, use_panel=use_panel, panel_args=panel_args)
    # mvec is np.array ordered as desired
    return mvec, mdict

# Because the earlier code in your repo puts compute_moments_from_policy inside smm_objective module,
# we must import it dynamically to avoid circular imports. Let's define a small inner dispatcher:
def compute_moments_from_policy_wrapper_inner(params: Dict[str, Any],
                                              use_panel: bool = False,
                                              panel_args: Optional[Dict[str, Any]] = None
                                              ) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Helper to call modules.smm_objective.compute_moments_from_policy by constructing
    the required policy/V inside that function. We call smm_objective.smm_objective with a 'quiet' flag
    or directly call compute_moments_from_policy imported at the top. To avoid import cycles,
    smm_objective.compute_moments_from_policy should be imported directly from your smm_objective module.
    """
    # Import here to avoid circular imports at module import time
    from modules.smm_objective import compute_moments_from_policy as _cmp
    # We need policy_idx, V, ext_flag, K_grid, A_grid, P_A; but compute_moments_from_policy in smm_objective
    # expects those objects. Fortunately the smm_objective module exposes compute_moments_from_policy wrapper
    # that accepts theta and builds them internally. But if not, we can call smm_objective.smm_objective with a hack:
    # call smm_objective to compute J and info (which also returns moments). We'll reuse that info.
    # Use use_panel flag and panel_args forwarded
    theta_list = [params[name] for name in params.get("_mapping_order_", [])] if "_mapping_order_" in params else None
    # Fallback approach: call smm_objective with mapping provided via panel_args (user must supply mapping earlier)
    # To keep this function general, we expect panel_args to contain 'mapping' and original 'theta'.
    if panel_args is not None and "mapping" in panel_args and "theta" in panel_args:
        mapping = panel_args["mapping"]
        theta = panel_args["theta"]
        J, info = smm_objective(theta, mapping, params, use_panel=use_panel, panel_args=panel_args)
        # info["moments_sim"] is dict with keys used earlier
        mdict = info["moments_sim"]
        mvec = np.array([mdict[k] for k in ["a1","a2","sc_IK","std_pi_K","mean_q","ext_frac"]], dtype=float)
        return mvec, mdict
    else:
        # last resort: raise informative error
        raise RuntimeError("compute_moments_from_policy_wrapper_inner needs panel_args with 'mapping' and 'theta' so it can call smm_objective. "
                           "Please call estimate_moment_covariance with panel_args that include 'mapping' and 'theta'.")

# ---------------------------------------------------------------------
# 2) Compute numerical Jacobian of simulated moments (q x p)
# ---------------------------------------------------------------------
def compute_jacobian(theta: np.ndarray,
                     mapping: list,
                     params_base: Dict[str, Any],
                     eps: float = 1e-4,
                     use_panel: bool = False,
                     panel_args: Optional[Dict[str, Any]] = None,
                     save_path: Optional[str] = "./data/moment_jacobian.npz"
                     ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Numerical Jacobian using central differences. Wraps smm_objective.numeric_moment_jacobian.

    Returns:
      G : (q x p) Jacobian (rows correspond to moments in canonical order)
      info : dict with details and timestamps
    """
    # numeric_moment_jacobian exists in modules.smm_objective and expects to call smm_objective internally.
    # We forward the same kwargs via smm_kwargs
    smm_kwargs = {"use_panel": use_panel, "panel_args": panel_args}
    jac, mapping_out = numeric_moment_jacobian(theta, mapping, params_base, eps=eps, **{"smm_kwargs": smm_kwargs})
    # numeric_moment_jacobian returns jac (q x p)
    info = {"eps": eps, "computed_at": time.time()}
    if save_path is not None:
        p = Path(save_path); p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, jac=jac)
    return jac, info

# ---------------------------------------------------------------------
# 3) Compute standard errors via sandwich or efficient formula
# ---------------------------------------------------------------------
def compute_smm_standard_errors(theta: np.ndarray,
                                mapping: list,
                                params_base: Dict[str, Any],
                                S_panels: int = 500,
                                panel_args: Optional[Dict[str, Any]] = None,
                                eps: float = 1e-4,
                                W: Optional[np.ndarray] = None,
                                use_general_sandwich: bool = True,
                                save_path: Optional[str] = "./results/smm_std_errors.json"
                                ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute covariance matrix and standard errors for theta_hat using S simulated panels.

    Returns:
      cov_theta : (p x p) covariance matrix
      std_err : (p,) standard errors (sqrt of diag of cov_theta)
      diagnostics : dict with G, S_psi, W, moment_names, etc.

    Notes:
      - S_panels: number of simulated panels used to estimate S_psi (covariance of moments)
      - If W is None:
          * if use_general_sandwich: use identity W = I
          * else (efficient): set W = S_psi^{-1}
      - multiply_factor = (1 + 1/S_panels) as in the PS description
    """
    panel_args = panel_args or {}
    # 1) compute S_psi
    S_psi, info_cov = estimate_moment_covariance(theta, mapping, params_base,
                                                S_panels=S_panels, panel_args=panel_args,
                                                use_panel_simulator=True, save_path="./data/moment_covariance.npz")
    # 2) compute Jacobian G (q x p)
    G, info_jac = compute_jacobian(theta, mapping, params_base, eps=eps, use_panel=True, panel_args={"mapping": mapping, "theta": theta, **(panel_args or {})}, save_path="./data/moment_jacobian.npz")

    q, p = G.shape
    # default W handling
    if W is None:
        if use_general_sandwich:
            Wmat = np.eye(q)
        else:
            # efficient: W = S_psi^{-1}
            Wmat = np.linalg.pinv(S_psi)
    else:
        Wmat = np.asarray(W, dtype=float)

    # Multiply factor from PS: (1 + 1/S)
    multiply_factor = 1.0 + 1.0 / float(S_panels)

    # Compute (G' W G)
    GWG = G.T @ Wmat @ G
    # invert GWG robustly (pinv)
    try:
        GWG_inv = np.linalg.inv(GWG)
    except np.linalg.LinAlgError:
        GWG_inv = np.linalg.pinv(GWG)

    if use_general_sandwich:
        # sandwich: (G' W G)^{-1} (G' W S_psi W G) (G' W G)^{-1}
        middle = G.T @ Wmat @ S_psi @ Wmat @ G
        cov_theta = multiply_factor * (GWG_inv @ middle @ GWG_inv)
    else:
        # efficient: W = S_psi^{-1} so cov = multiply_factor * (G' W G)^{-1}
        cov_theta = multiply_factor * GWG_inv

    std_err = np.sqrt(np.real(np.diag(cov_theta)))

    diagnostics = {
        "G": G,
        "S_psi": S_psi,
        "W": Wmat,
        "GWG": GWG,
        "GWG_inv": GWG_inv,
        "multiply_factor": multiply_factor,
        "S_panels": S_panels,
        "moment_names": ["a1","a2","sc_IK","std_pi_K","mean_q","ext_frac"],
        "mapping": mapping
    }

    # save results
    if save_path is not None:
        pth = Path(save_path); pth.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "theta": {name: float(val) for name, val in zip(mapping, theta)},
            "std_err": std_err.tolist(),
            "cov_theta": cov_theta.tolist(),
            "diagnostics": {
                "S_panels": S_panels,
                "multiply_factor": multiply_factor
            }
        }
        with pth.open("w", encoding="utf8") as f:
            json.dump(out, f, indent=2)

    return cov_theta, std_err, diagnostics