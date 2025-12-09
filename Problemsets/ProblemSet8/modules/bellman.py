# modules/bellman.py
"""
Bellman solver for the firm DP.

This module provides:
 - a numba-accelerated VFI implementation (vfi_numba)
 - a pure-Python fallback (_vfi_python)
 - a wrapper solve_bellman(...) that calls the numba VFI directly.

CRITICAL: profits are computed using the *current* capital K (Ki) when forming the
one-period payoff D.  That is, pi = A * Ki**alpha.  Investment and adjustment costs
enter through I = K' - (1-delta)*K and C(I, K).
"""
from typing import Tuple, Optional
import numpy as np
import sys, traceback

# IMPORT NUMBA DIRECTLY (no runtime detection/branch)
from numba import njit

# --------------------------
# Numba-friendly helpers
# --------------------------
@njit
def _net_investment_nb(Kp_arr, Ki, delta):
    n = Kp_arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = Kp_arr[i] - (1.0 - delta) * Ki
    return out

@njit
def _adjustment_cost_nb(I_arr, Ki, gamma):
    n = I_arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    denom = Ki if Ki > 0.0 else 1e-12
    for i in range(n):
        out[i] = 0.5 * gamma * (I_arr[i] * I_arr[i]) / denom
    return out

@njit
def _build_D_base_nb(Kp_vec, Ki, Ai, alpha, gamma, delta, p):
    """
    Build vector D_base[k'] = pi(Ki,Ai) - p * I - C(I,Ki),
    where I = K' - (1-delta) Ki.
    Profit evaluated at current Ki (pi_i = Ai * Ki**alpha).
    """
    n = Kp_vec.shape[0]
    D = np.empty(n, dtype=np.float64)
    pi_i = 0.0
    if Ki > 0.0:
        pi_i = Ai * (Ki ** alpha)
    else:
        pi_i = 0.0
    # compute I and C for each candidate K'
    for idx in range(n):
        Kp = Kp_vec[idx]
        Ival = Kp - (1.0 - delta) * Ki
        denom = Ki if Ki > 0.0 else 1e-12
        Cval = 0.5 * gamma * (Ival * Ival) / denom
        D[idx] = pi_i - p * Ival - Cval
    return D

# --------------------------
# Numba-accelerated VFI
# --------------------------
@njit
def vfi_numba(K_grid: np.ndarray,
              A_grid: np.ndarray,
              P_A: np.ndarray,
              alpha: float,
              gamma: float,
              delta: float,
              p: float,
              phi0: float,
              phi1: float,
              beta: float,
              tol: float,
              maxiter: int,
              enforce_internal_constraint: bool):
    """
    Value function iteration (numba compiled).

    Returns:
        V (N_K, N_A), policy_k_idx (N_K, N_A), external_flag (N_K, N_A), q (N_K, N_A)
    """
    N_K = K_grid.shape[0]
    N_A = A_grid.shape[0]
    tiny = 1e-12

    V = np.zeros((N_K, N_A), dtype=np.float64)
    policy_k_idx = np.zeros((N_K, N_A), dtype=np.int64)
    external_flag = np.zeros((N_K, N_A), dtype=np.boolean)

    Kp_vec = K_grid.copy()

    # iterate
    for it in range(maxiter):
        V_old = V.copy()

        # precompute continuation values cont[kprime, a] = beta * sum_{a'} P_A[a, a'] * V_old[kprime, a']
        cont = np.zeros((N_K, N_A), dtype=np.float64)
        for a_idx in range(N_A):
            for kprime in range(N_K):
                s = 0.0
                for aprime in range(N_A):
                    s += P_A[a_idx, aprime] * V_old[kprime, aprime]
                cont[kprime, a_idx] = beta * s

        # iterate over states
        for a_idx in range(N_A):
            Ai = A_grid[a_idx]
            for i_k in range(N_K):
                Ki = K_grid[i_k]

                # build one-period payoff for all candidate K' using profit at current Ki
                D_base = _build_D_base_nb(Kp_vec, Ki, Ai, alpha, gamma, delta, p)

                # internal feasibility choices
                if enforce_internal_constraint:
                    avail = Ai * (Ki ** alpha) if Ki > 0.0 else 0.0
                    feasible_any = False
                    feasible_mask = np.empty(N_K, dtype=np.boolean)
                    for kp_idx in range(N_K):
                        Ival = Kp_vec[kp_idx] - (1.0 - delta) * Ki
                        feasible_mask[kp_idx] = (Ival <= avail + tiny)
                        if feasible_mask[kp_idx]:
                            feasible_any = True
                    if not feasible_any:
                        V_i_choices = np.empty(N_K, dtype=np.float64)
                        for kp_idx in range(N_K):
                            V_i_choices[kp_idx] = -1e300
                    else:
                        V_i_choices = np.empty(N_K, dtype=np.float64)
                        for kp_idx in range(N_K):
                            if feasible_mask[kp_idx]:
                                V_i_choices[kp_idx] = D_base[kp_idx]
                            else:
                                V_i_choices[kp_idx] = -1e300
                else:
                    V_i_choices = D_base.copy()

                # external financing choices (subtract fixed and variable external financing costs)
                V_e_choices = np.empty(N_K, dtype=np.float64)
                for kp_idx in range(N_K):
                    Ival = Kp_vec[kp_idx] - (1.0 - delta) * Ki
                    avail = Ai * (Ki ** alpha) if Ki > 0.0 else 0.0
                    Epos = Ival - avail
                    if Epos < 0.0:
                        Epos = 0.0
                    var_cost = phi1 * Epos * p
                    V_e_choices[kp_idx] = D_base[kp_idx] - phi0 - var_cost

                # choose best (add continuation)
                cont_col = cont[:, a_idx]  # cont for kprime across aprime probabilities
                best_val = -1e300
                best_idx = 0
                best_ext = False
                for kp_idx in range(N_K):
                    v_i = V_i_choices[kp_idx] + cont_col[kp_idx]
                    v_e = V_e_choices[kp_idx] + cont_col[kp_idx]
                    if v_e > v_i:
                        v_best = v_e
                        ext_choice = True
                    else:
                        v_best = v_i
                        ext_choice = False
                    if v_best > best_val:
                        best_val = v_best
                        best_idx = kp_idx
                        best_ext = ext_choice

                V[i_k, a_idx] = best_val
                policy_k_idx[i_k, a_idx] = best_idx
                external_flag[i_k, a_idx] = best_ext

        # convergence check
        diff = 0.0
        for i in range(N_K):
            for j in range(N_A):
                d = V[i, j] - V_old[i, j]
                if d < 0.0:
                    d = -d
                if d > diff:
                    diff = d

        if diff < tol:
            break

    # compute q = V / K (avoid divide by 0)
    q = np.empty((N_K, N_A), dtype=np.float64)
    for i in range(N_K):
        denom = K_grid[i] if K_grid[i] > 0.0 else tiny
        for j in range(N_A):
            q[i, j] = V[i, j] / denom

    return V, policy_k_idx, external_flag, q

# --------------------------
# Pure-Python fallback VFI (identical algorithm, easier debugging)
# --------------------------
def _vfi_python(K_grid: np.ndarray,
                A_grid: np.ndarray,
                P_A: np.ndarray,
                alpha: float,
                gamma: float,
                delta: float,
                p: float,
                phi0: float,
                phi1: float,
                beta: float,
                tol: float,
                maxiter: int,
                enforce_internal_constraint: bool):
    N_K = K_grid.shape[0]
    N_A = A_grid.shape[0]
    tiny = 1e-12

    V = np.zeros((N_K, N_A), dtype=float)
    policy_k_idx = np.zeros((N_K, N_A), dtype=int)
    external_flag = np.zeros((N_K, N_A), dtype=bool)

    Kp_vec = K_grid.copy()

    for it in range(maxiter):
        V_old = V.copy()

        # continuation
        cont = np.zeros((N_K, N_A), dtype=float)
        for a_idx in range(N_A):
            for kprime in range(N_K):
                s = 0.0
                for aprime in range(N_A):
                    s += P_A[a_idx, aprime] * V_old[kprime, aprime]
                cont[kprime, a_idx] = beta * s

        for a_idx in range(N_A):
            Ai = A_grid[a_idx]
            for i_k in range(N_K):
                Ki = K_grid[i_k]

                # compute pi at current Ki
                if Ki > 0.0:
                    pi_i = Ai * (Ki ** alpha)
                else:
                    pi_i = 0.0

                # build D_base for all K' candidates
                D_base = np.empty(N_K, dtype=float)
                for kp_idx in range(N_K):
                    Kp = Kp_vec[kp_idx]
                    Ival = Kp - (1.0 - delta) * Ki
                    if Ki > 0.0:
                        Cval = 0.5 * gamma * (Ival * Ival) / Ki
                    else:
                        Cval = 0.5 * gamma * (Ival * Ival) / 1e-12
                    D_base[kp_idx] = pi_i - p * Ival - Cval

                # internal choices (feasibility constraint)
                if enforce_internal_constraint:
                    avail = pi_i
                    feasible_any = False
                    feasible_mask = np.zeros(N_K, dtype=bool)
                    for kp_idx in range(N_K):
                        Ival = Kp_vec[kp_idx] - (1.0 - delta) * Ki
                        feasible_mask[kp_idx] = (Ival <= avail + tiny)
                        if feasible_mask[kp_idx]:
                            feasible_any = True
                    if not feasible_any:
                        V_i_choices = np.full(N_K, -1e300, dtype=float)
                    else:
                        V_i_choices = np.empty(N_K, dtype=float)
                        for kp_idx in range(N_K):
                            V_i_choices[kp_idx] = D_base[kp_idx] if feasible_mask[kp_idx] else -1e300
                else:
                    V_i_choices = D_base.copy()

                # external financing choices
                V_e_choices = np.empty(N_K, dtype=float)
                for kp_idx in range(N_K):
                    Ival = Kp_vec[kp_idx] - (1.0 - delta) * Ki
                    avail = pi_i
                    Epos = Ival - avail
                    if Epos < 0.0:
                        Epos = 0.0
                    var_cost = phi1 * Epos * p
                    V_e_choices[kp_idx] = D_base[kp_idx] - phi0 - var_cost

                # pick best including continuation
                cont_col = cont[:, a_idx]
                best_val = -1e300
                best_idx = 0
                best_ext = False
                for kp_idx in range(N_K):
                    v_i = V_i_choices[kp_idx] + cont_col[kp_idx]
                    v_e = V_e_choices[kp_idx] + cont_col[kp_idx]
                    if v_e > v_i:
                        v_best = v_e
                        ext_choice = True
                    else:
                        v_best = v_i
                        ext_choice = False
                    if v_best > best_val:
                        best_val = v_best
                        best_idx = kp_idx
                        best_ext = ext_choice

                V[i_k, a_idx] = best_val
                policy_k_idx[i_k, a_idx] = best_idx
                external_flag[i_k, a_idx] = best_ext

        diff = np.max(np.abs(V - V_old))
        if diff < tol:
            break

    # compute q
    q = np.empty((N_K, N_A), dtype=float)
    for i in range(N_K):
        denom = K_grid[i] if K_grid[i] > 0.0 else tiny
        for j in range(N_A):
            q[i, j] = V[i, j] / denom

    return V, policy_k_idx, external_flag, q

# --------------------------
# Wrapper solve_bellman
# --------------------------
def solve_bellman(params: dict,
                  K_grid,
                  A_grid,
                  P_A,
                  enforce_internal_constraint: bool = False,
                  tol: float = 1e-6,
                  maxiter: int = 2000,
                  compute_q: bool = False,
                  save_path: Optional[str] = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Solve Bellman using the numba-accelerated vfi_numba function (direct call).

    Returns:
        V (N_K, N_A), policy_k_idx (N_K, N_A), external_flag (N_K, N_A), q (or None)
    """
    try:
        # Convert to numpy arrays
        K_grid = np.asarray(K_grid, dtype=np.float64)
        A_grid = np.asarray(A_grid, dtype=np.float64)
        P_A = np.asarray(P_A, dtype=np.float64)

        # Unpack parameters
        alpha = float(params.get("alpha", 0.6956))
        gamma = float(params.get("gamma", 0.1331))
        delta = float(params.get("delta", 0.15))
        beta = float(params.get("beta", 0.95))
        p = float(params.get("p", 1.0))
        phi0 = float(params.get("phi_tilde_0", 0.0))
        phi1 = float(params.get("phi1", 0.0))

        # Inform user
        print("[bellman] Calling numba-accelerated VFI.", file=sys.stderr)

        # Call the numba VFI directly
        V, policy_k_idx, external_flag, q = vfi_numba(K_grid, A_grid, P_A,
                                                      alpha, gamma, delta,
                                                      p, phi0, phi1,
                                                      beta, tol, int(maxiter),
                                                      bool(enforce_internal_constraint))

        if not compute_q:
            q = None

        # ensure dtypes and shapes
        V = np.asarray(V, dtype=np.float64)
        policy_k_idx = np.asarray(policy_k_idx, dtype=np.int64)
        external_flag = np.asarray(external_flag, dtype=bool)
        if q is not None:
            q = np.asarray(q, dtype=np.float64)

        # basic shape checks
        N_K = K_grid.size
        N_A = A_grid.size
        if V.shape != (N_K, N_A):
            raise RuntimeError(f"Returned V has wrong shape {V.shape}, expected ({N_K},{N_A})")

        # safety: clip policy indices into valid range
        policy_k_idx = np.clip(policy_k_idx, 0, N_K - 1).astype(np.int64)

        return V, policy_k_idx, external_flag, q

    except Exception:
        print("[bellman] Exception inside solve_bellman:", file=sys.stderr)
        traceback.print_exc()
        # As a fallback, attempt to run the pure Python solver for debugging
        try:
            print("[bellman] Attempting Python fallback _vfi_python.", file=sys.stderr)
            V, policy_k_idx, external_flag, q = _vfi_python(K_grid, A_grid, P_A,
                                                            alpha, gamma, delta,
                                                            p, phi0, phi1,
                                                            beta, tol, int(maxiter),
                                                            bool(enforce_internal_constraint))
            if not compute_q:
                q = None
            policy_k_idx = np.clip(policy_k_idx, 0, K_grid.size - 1).astype(np.int64)
            return V, policy_k_idx, external_flag, q
        except Exception:
            print("[bellman] Python fallback also failed.", file=sys.stderr)
            traceback.print_exc()
            raise
