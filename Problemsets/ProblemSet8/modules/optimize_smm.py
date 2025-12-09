# modules/optimize_smm.py
"""
Module J -- Optimization Routine for SMM estimation

Provides a two-stage optimizer that:
  1) runs a global search (Differential Evolution or Basin-hopping)
  2) refines the best global result using a local optimizer (L-BFGS-B)

Primary functions:
 - run_global_search(...) -> dict with best 'x' and 'fun' and diagnostics
 - run_local_refinement(...) -> dict with refined 'x' and 'fun'
 - optimize_smm(...) -> coordinates the whole process, saves results to relative path

Requirements:
 - scipy (for optimizers)
 - numpy
 - uses modules.smm_objective.smm_objective to evaluate J(theta)
 - uses only relative paths for saving results (./results/...)

Notes:
 - The mapping argument is the list of parameter names (strings) in the order theta is supplied.
 - Bounds should be provided as a dict {param_name: (lower, upper)}. If None, defaults are used.
 - This wrapper keeps the optimization interface simple and flexible for experiments.
"""
from pathlib import Path
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from scipy.optimize import differential_evolution, minimize, basinhopping

# import the SMM objective wrapper you already have
from modules.smm_objective import smm_objective

# -----------------------
# Utilities
# -----------------------
def _params_from_theta(theta: np.ndarray, mapping: List[str]) -> Dict[str, float]:
    """Return dict mapping parameter names to values."""
    return {name: float(val) for name, val in zip(mapping, theta)}

def _default_bounds_for_mapping(mapping: List[str]) -> List[Tuple[float, float]]:
    """
    Provide conservative default bounds for common parameters:
    - alpha in (0.2, 1.2)
    - gamma in (1e-4, 10)
    - rho in (-0.99, 0.99)
    - sigma in (1e-4, 3.0)
    - phi_tilde_0 in (0.0, 0.05)  (fixed cost relative to mean K; paper finds 0)
    Others: fallback ( -10, 10 ).
    """
    defaults = []
    for name in mapping:
        if name.lower() in ("alpha",):
            defaults.append((0.2, 1.2))
        elif name.lower() in ("gamma", "gamma_", "gam"):
            defaults.append((1e-4, 10.0))
        elif name.lower() in ("rho",):
            defaults.append((-0.99, 0.99))
        elif name.lower() in ("sigma", "sig"):
            defaults.append((1e-4, 3.0))
        elif name.lower() in ("phi_tilde_0", "phitilde0", "phi0", "phi_tilde"):
            defaults.append((0.0, 0.05))
        else:
            defaults.append((-10.0, 10.0))
    return defaults

# -----------------------
# Progress-wrapper utility for objective evaluation
# -----------------------
# How often to print progress (change this to 1 for very chatty output)
_PRINT_EVERY = 10

def make_progress_wrapper(obj_func, print_every: int = _PRINT_EVERY):
    """
    Wrap an objective function so that each call is counted and progress is printed every
    `print_every` evaluations. If the wrapped objective raises an exception, a large
    penalty value is returned so the global optimizer can continue.
    """

    state = {"count": 0, "best_val": float("inf"), "start_time": time.time()}

    def wrapped(x):
        state["count"] += 1
        c = state["count"]

        try:
            val = obj_func(x)
            # obj_func expected to return scalar (float)
            fv = float(val)
        except Exception as e:
            # Print a short message and return a large penalty instead of crashing
            print(f"[opt] eval={c}: objective raised ({e}); returning large penalty", flush=True)
            fv = 1e20

        # update best
        if fv < state["best_val"]:
            state["best_val"] = fv

        # periodic print
        if (c % print_every) == 0:
            elapsed = time.time() - state["start_time"]
            print(f"[opt] eval={c}  best={state['best_val']:.6g}  last={fv:.6g}  elapsed={elapsed:.1f}s", flush=True)

        return fv

    return wrapped

# -----------------------
# Global search routines
# -----------------------
def run_global_search(theta0: Optional[np.ndarray],
                      mapping: List[str],
                      params_base: Dict[str, Any],
                      bounds: Optional[List[Tuple[float,float]]] = None,
                      global_method: str = "diffev",
                      de_popsize: int = 15,
                      de_maxiter: int = 40,
                      basinhopping_niter: int = 50,
                      seed: Optional[int] = 12345,
                      smm_kwargs: Optional[Dict[str, Any]] = None
                      ) -> Dict[str, Any]:
    """
    Run a global search over theta.

    Parameters
    ----------
    theta0 : initial guess (1D array) or None
    mapping : list of parameter names (order for theta)
    params_base : base params dict (passed to smm_objective)
    bounds : list of (low,high) tuples in same order as mapping. If None, use defaults.
    global_method : 'diffev' (DifferentialEvolution) or 'basinhopping'
    de_popsize, de_maxiter : DE tuning
    basinhopping_niter : basin-hopping steps if chosen
    seed : RNG seed for reproducible DE or BH
    smm_kwargs : extra kwargs passed to smm_objective as (use_panel, solver_opts, etc.)

    Returns
    -------
    dict with keys: 'x' (best theta), 'fun' (objective value), 'method', 'time', 'details'
    """
    smm_kwargs = smm_kwargs or {}
    rng = np.random.default_rng(seed)

    p = len(mapping)
    if bounds is None:
        bounds = _default_bounds_for_mapping(mapping)
    bounds_arr = np.array(bounds, dtype=float)
    assert bounds_arr.shape == (p, 2), "bounds must match mapping length"

    # objective wrapper for optimizer (minimize)
    def obj_wrapped(x):
        # x -> float list
        try:
            J, _info = smm_objective(x, mapping, params_base, **smm_kwargs)
            return float(J)
        except Exception as e:
            # In global search, prefer returning large penalty rather than crashing
            print("Warning: smm_objective raised during global search:", e)
            return 1e12

    start_time = time.time()

    if global_method.lower() in ("diffev", "differential_evolution"):
        # Wrap with progress-tracking
        obj_for_de = make_progress_wrapper(obj_wrapped, print_every=_PRINT_EVERY)

        # Differential evolution
        result = differential_evolution(obj_for_de, bounds, strategy='best1bin',
                                        maxiter=de_maxiter, popsize=de_popsize,
                                        seed=int(seed) if seed is not None else None,
                                        polish=False, disp=False)
        x_best = result.x
        fun_best = result.fun
        details = {"nit": result.nit, "popsize": de_popsize, "message": result.message}
        method = "differential_evolution"

    elif global_method.lower() in ("basinhopping", "bh"):
        # Need a local minimizer for the inner loop
        if theta0 is None:
            # start from center of bounds
            theta0 = 0.5 * (bounds_arr[:,0] + bounds_arr[:,1])
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
        bh = basinhopping(obj_wrapped, theta0, niter=basinhopping_niter, minimizer_kwargs=minimizer_kwargs,
                          seed=int(seed) if seed is not None else None, disp=False)
        x_best = bh.x
        fun_best = bh.fun
        details = {"niter": basinhopping_niter, "message": getattr(bh, "message", "")}
        method = "basinhopping"
    else:
        raise ValueError("unknown global_method: choose 'diffev' or 'basinhopping'")

    elapsed = time.time() - start_time
    return {"x": np.array(x_best, dtype=float), "fun": float(fun_best),
            "method": method, "time": elapsed, "details": details}


# -----------------------
# Local refinement
# -----------------------
def run_local_refinement(x0: np.ndarray,
                         mapping: List[str],
                         params_base: Dict[str, Any],
                         bounds: Optional[List[Tuple[float,float]]] = None,
                         local_method: str = "L-BFGS-B",
                         options: Optional[Dict[str,Any]] = None,
                         smm_kwargs: Optional[Dict[str,Any]] = None
                         ) -> Dict[str, Any]:
    """
    Run a local optimization starting from x0 using scipy.minimize.

    Returns dict with keys: 'x', 'fun', 'success', 'message', 'nit', 'time'
    """
    smm_kwargs = smm_kwargs or {}
    p = len(mapping)
    if bounds is None:
        bounds = _default_bounds_for_mapping(mapping)
    bounds_tuple = tuple(bounds)

    def obj_wrapped(x):
        J, _info = smm_objective(x, mapping, params_base, **smm_kwargs)
        return float(J)

    start_time = time.time()
    res = minimize(obj_wrapped, x0, method=local_method, bounds=bounds_tuple, options=options or {"maxiter":100})
    elapsed = time.time() - start_time

    return {"x": np.array(res.x, dtype=float), "fun": float(res.fun),
            "success": bool(res.success), "message": res.message, "nit": getattr(res, "nit", None),
            "time": elapsed, "full_result": res}


# -----------------------
# Full optimization driver
# -----------------------
def optimize_smm(mapping: List[str],
                 params_base: Dict[str, Any],
                 theta0: Optional[List[float]] = None,
                 bounds: Optional[Dict[str, Tuple[float,float]]] = None,
                 global_opts: Optional[Dict[str, Any]] = None,
                 local_opts: Optional[Dict[str, Any]] = None,
                 smm_opts: Optional[Dict[str, Any]] = None,
                 save_path: Optional[str] = "./results/opt_result.json",
                 verbose: bool = True
                 ) -> Dict[str, Any]:
    """
    Coordinate global + local optimization to minimize J(theta).

    Parameters
    ----------
    mapping : list of parameter names (strings)
    params_base : base params dict (Module A)
    theta0 : optional starting guess list (same order as mapping)
    bounds : optional dict {param_name: (low,high)} ; if None defaults used
    global_opts : dict configuring global search (see run_global_search)
    local_opts : dict configuring local refinement (see run_local_refinement)
    smm_opts : dict forwarded into smm_objective (use_panel, solver_opts etc.)
    save_path : relative path where to save final results (JSON)
    verbose : whether to print progress

    Returns
    -------
    info : dict with final solution, histories, and saved filename
    """
    # defaults / unpack
    global_opts = global_opts or {}
    local_opts = local_opts or {}
    smm_opts = smm_opts or {}

    p = len(mapping)

    # build bounds list in order of mapping
    if bounds is None:
        bounds_list = _default_bounds_for_mapping(mapping)
    else:
        # bounds passed as dict: transform to list
        bounds_list = []
        for name in mapping:
            if name not in bounds:
                raise KeyError(f"Bounds must include parameter '{name}'")
            bounds_list.append(tuple(bounds[name]))
    # initial guess
    if theta0 is None:
        # choose midpoint of bounds
        theta0_arr = np.array([(b[0] + b[1]) / 2.0 for b in bounds_list], dtype=float)
    else:
        theta0_arr = np.asarray(theta0, dtype=float)
        if theta0_arr.size != p:
            raise ValueError("theta0 length must match mapping length")

    if verbose:
        print("Starting optimization with mapping =", mapping)
        print("Initial theta0 =", theta0_arr)

    # Stage 1: Global search
    if verbose:
        print("Running global search...")
    g_opts = {"global_method": global_opts.get("global_method", "diffev"),
              "de_popsize": global_opts.get("de_popsize", 15),
              "de_maxiter": global_opts.get("de_maxiter", 30),
              "basinhopping_niter": global_opts.get("basinhopping_niter", 50),
              "seed": global_opts.get("seed", 12345),
              "smm_kwargs": smm_opts}
    gres = run_global_search(theta0_arr, mapping, params_base,
                             bounds=bounds_list,
                             global_method=g_opts["global_method"],
                             de_popsize=g_opts["de_popsize"],
                             de_maxiter=g_opts["de_maxiter"],
                             basinhopping_niter=g_opts["basinhopping_niter"],
                             seed=g_opts["seed"],
                             smm_kwargs=g_opts["smm_kwargs"])
    if verbose:
        print(f"Global search finished (method={gres['method']}, time={gres['time']:.1f}s). Best J = {gres['fun']:.6g}")

    # Stage 2: Local refinement
    if verbose:
        print("Running local refinement (L-BFGS-B)...")
    l_opts = {"local_method": local_opts.get("local_method", "L-BFGS-B"),
              "options": local_opts.get("options", {"maxiter": 200}),
              "smm_kwargs": smm_opts}
    lres = run_local_refinement(gres["x"], mapping, params_base, bounds=bounds_list,
                                local_method=l_opts["local_method"], options=l_opts["options"],
                                smm_kwargs=l_opts["smm_kwargs"])
    if verbose:
        print(f"Local refinement finished. Final J = {lres['fun']:.6g}, success = {lres['success']}")

    # Package results
    final_theta = lres["x"]
    final_fun = lres["fun"]
    info = {
        "mapping": mapping,
        "theta_init": theta0_arr.tolist(),
        "bounds": [list(b) for b in bounds_list],
        "global_result": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in gres.items()},
        "local_result": {"x": final_theta.tolist(), "fun": final_fun,
                         "success": lres["success"], "message": lres["message"], "nit": lres["nit"]},
        "smm_opts": smm_opts,
        "timestamp": time.time()
    }

    # save to relative path
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf8") as f:
            json.dump(info, f, indent=2)
        if verbose:
            print(f"Saved optimization info to {save_path}")

    return info
