"""
modules/policy_eval.py

Module F: Compute policy-implied variables and model-implied moments.

Functions
---------
evaluate_policy_states(policy_k_idx, V, external_flag, K_grid, A_grid, P_A, params, save_path=None)
    Compute per-state objects and aggregated moments using the stationary distribution induced
    by the deterministic policy and the exogenous A transition matrix.

Returns a tuple (state_dict, moments_dict) where:
 - state_dict contains arrays indexed by state (i,a) flattened to length S = N_K * N_A
 - moments_dict contains the model-implied moments used in SMM:
     'a1','a2','sc_IK','std_pi_K','mean_q','ext_frac'
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json
import numpy as np

# helpers
def _flatten_idx(i: int, a: int, N_A: int) -> int:
    return i * N_A + a

def _unflatten_idx(s: int, N_A: int) -> Tuple[int,int]:
    return divmod(s, N_A)   # returns (i, a)

def _stationary_dist_from_policy(policy_k_idx: np.ndarray, P_A: np.ndarray,
                                 tol: float = 1e-12, maxiter: int = 10000) -> np.ndarray:
    """
    Compute stationary distribution over joint states s = (i,a) induced by deterministic policy:
      - from state (i,a) the next state's K index is k' = policy_k_idx[i,a]
      - A transitions according to P_A[a, a']
    Build transition matrix T of size SxS where S=N_K*N_A:
      T[s, s'] = P_A[a, a'] if s' = (k', a') where k' = policy_k_idx[i,a], else 0
    Then find stationary dist pi solving pi = pi T via power iteration.
    """
    N_K, N_A = policy_k_idx.shape
    S = N_K * N_A

    # Build sparse-like transition (we'll build dense because sizes are moderate: e.g., 600*7=4200)
    T = np.zeros((S, S), dtype=float)

    for i in range(N_K):
        for a in range(N_A):
            s = _flatten_idx(i, a, N_A)
            kprime = int(policy_k_idx[i, a])
            # transitions to states (kprime, a') with prob P_A[a, a']
            for a2 in range(N_A):
                sp = _flatten_idx(kprime, a2, N_A)
                T[s, sp] = P_A[a, a2]

    # power method: start uniform
    pi = np.ones(S, dtype=float) / S
    for it in range(maxiter):
        pi_next = pi @ T
        diff = np.max(np.abs(pi_next - pi))
        pi = pi_next
        if diff < tol:
            break
    else:
        raise RuntimeError("Stationary distribution did not converge in _stationary_dist_from_policy")

    return pi  # length S, sums to 1


def evaluate_policy_states(policy_k_idx: np.ndarray,
                           V: np.ndarray,
                           external_flag: np.ndarray,
                           K_grid: np.ndarray,
                           A_grid: np.ndarray,
                           P_A: np.ndarray,
                           params: dict,
                           save_path: Optional[str] = "./data/policy_eval.json"
                           ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Evaluate policy to obtain per-state variables and aggregate moments.

    Parameters
    ----------
    policy_k_idx : ndarray int (N_K, N_A)
        Policy mapping from (i,a) -> index k' in K_grid.
    V : ndarray (N_K, N_A)
        Value function at each state.
    external_flag : ndarray bool (N_K, N_A)
        Indicator whether external finance was chosen in the DP solution.
    K_grid : ndarray (N_K,)
    A_grid : ndarray (N_A,)
    P_A : ndarray (N_A, N_A)
    params : dict
        Must contain delta, alpha, p (if not present defaults used).
    save_path : str or None
        If provided, save state arrays and moments to JSON (relative path).

    Returns
    -------
    state_dict : dict of 1-D arrays (length S = N_K*N_A)
      keys: 'K', 'A', 'Kp', 'I', 'pi', 'E', 'q', 'IK', 'piK'
    moments_dict : dict with keys 'a1','a2','sc_IK','std_pi_K','mean_q','ext_frac'
    """
    delta = float(params.get("delta", 0.15))
    alpha = float(params.get("alpha", 0.6956))
    p = float(params.get("p", 1.0))

    N_K = K_grid.size
    N_A = A_grid.size
    S = N_K * N_A

    # Pre-allocate flattened arrays
    K = np.zeros(S, dtype=float)
    A = np.zeros(S, dtype=float)
    Kp = np.zeros(S, dtype=float)
    I = np.zeros(S, dtype=float)
    pi_vals = np.zeros(S, dtype=float)
    E = np.zeros(S, dtype=float)
    q = np.zeros(S, dtype=float)
    IK = np.zeros(S, dtype=float)
    piK = np.zeros(S, dtype=float)

    # Fill per-state arrays
    for i in range(N_K):
        Ki = K_grid[i]
        for a in range(N_A):
            s = _flatten_idx(i, a, N_A)
            Ai = A_grid[a]
            kprime_idx = int(policy_k_idx[i, a])
            Kp_s = K_grid[kprime_idx]

            # investment I = K' - (1-delta)*K
            I_s = Kp_s - (1.0 - delta) * Ki
            # profit pi = A * K^alpha
            pi_s = Ai * (Ki ** alpha) if Ki > 0 else 0.0
            # external financing E = max(I - pi, 0)
            E_s = max(I_s - pi_s, 0.0)

            K[s] = Ki
            A[s] = Ai
            Kp[s] = Kp_s
            I[s] = I_s
            pi_vals[s] = pi_s
            E[s] = E_s
            q[s] = V[i, a] / max(Ki, 1e-12)
            IK[s] = I_s / max(Ki, 1e-12)
            piK[s] = pi_s / max(Ki, 1e-12)

    # compute stationary distribution under policy
    pi_state = _stationary_dist_from_policy(policy_k_idx, P_A)

    # Moments:
    # 1) expected next-period q (Eq expectation): for state s=(i,a), expected q_{t+1} =
    #    sum_{a'} P_A[a,a'] * q( k'(i,a), a' )
    # We'll compute expected q_next for each s
    expected_q_next = np.zeros(S, dtype=float)
    for i in range(N_K):
        for a in range(N_A):
            s = _flatten_idx(i, a, N_A)
            kprime_idx = int(policy_k_idx[i, a])
            # q at (kprime_idx, a')
            for a2 in range(N_A):
                sp = _flatten_idx(kprime_idx, a2, N_A)
                expected_q_next[s] += P_A[a, a2] * q[sp]

    # Run the Q-regression: IK = intercept + a1 * expected_q_next + a2 * piK + error
    # We will estimate via OLS across the S states, weighting by stationary distribution.
    X = np.column_stack([np.ones(S), expected_q_next, piK])  # (S, 3)
    y = IK  # (S,)
    # Weighted least squares with weights = stationary distribution (gives distributional average)
    W = np.sqrt(pi_state)  # using sqrt weights to rewrite weighted least squares as OLS on scaled data
    Xw = X * W[:, None]
    yw = y * W
    beta_hat, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    intercept, a1, a2 = float(beta_hat[0]), float(beta_hat[1]), float(beta_hat[2])

    # Serial correlation of I/K: compute cov(I_t, I_{t+1}) / var(I)
    # joint distribution of (s, s') is joint_prob[s, s'] = pi_state[s] * T[s, s'] where T is built from policy
    # Rather than building full T again, reuse logic: from s=(i,a) transitions to s'=(kprime(i,a), a')
    # We'll compute mean_I and var_I first
    mean_IK = np.sum(pi_state * IK)
    var_IK = np.sum(pi_state * (IK - mean_IK) ** 2)
    if var_IK <= 0:
        sc_IK = 0.0
    else:
        cov = 0.0
        for i in range(N_K):
            for a in range(N_A):
                s = _flatten_idx(i, a, N_A)
                kprime = int(policy_k_idx[i, a])
                for a2 in range(N_A):
                    sp = _flatten_idx(kprime, a2, N_A)
                    prob = pi_state[s] * P_A[a, a2]
                    cov += prob * (IK[s] - mean_IK) * (IK[sp] - mean_IK)
        sc_IK = float(cov / var_IK)

    std_pi_K = float(np.sqrt(np.sum(pi_state * (piK - np.sum(pi_state * piK))**2)))
    mean_q = float(np.sum(pi_state * q))

    # external finance fraction: fraction of total investment financed externally
    # ext_frac = E_total / I_positive_total  (use positive investment only in denominator)
    E_total = float(np.sum(pi_state * E))
    Ipos_mask = I > 1e-12
    Ipos_total = float(np.sum(pi_state * I * Ipos_mask))
    ext_frac = float(E_total / Ipos_total) if Ipos_total > 0 else 0.0

    moments = {
        "a0": intercept,
        "a1": a1,
        "a2": a2,
        "sc_IK": sc_IK,
        "std_pi_K": std_pi_K,
        "mean_q": mean_q,
        "ext_frac": ext_frac
    }

    state_dict = {
        "K": K,
        "A": A,
        "Kp": Kp,
        "I": I,
        "pi": pi_vals,
        "E": E,
        "q": q,
        "IK": IK,
        "piK": piK,
        "expected_q_next": expected_q_next,
        "stationary_pi": pi_state
    }

    # Optionally save results to relative path
    if save_path is not None:
        out = {
            "metadata": {
                "N_K": int(N_K),
                "N_A": int(N_A)
            },
            # convert arrays to lists for JSON
            "state": {k: v.tolist() for k, v in state_dict.items()},
            "moments": moments
        }
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf8") as f:
            json.dump(out, f, indent=2)

    return state_dict, moments
