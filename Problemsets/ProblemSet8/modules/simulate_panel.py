# modules/simulate_panel.py
"""
Panel simulation module (Module G).

Simulate a panel of firms using deterministic policy functions and the Markov
process for A. Returns a pandas.DataFrame and optionally saves CSV / NPZ files
under relative paths (./data/...).

API:
    simulate_panel(policy_k_idx, V, external_flag, K_grid, A_grid, P_A, params,
                   N_firms=None, T=None, burn_in=None, init_method='stationary',
                   seed=12345, save_path_csv='./data/sim_panel.csv',
                   save_path_npz='./data/sim_panel.npz')
"""
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

# Helper: draw next A index for many firms given P_A row probabilities
def _draw_next_A_indices(current_a_indices: np.ndarray, P_A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    For each firm with current A index a_i, draw next A index a' using P_A[a_i, :].
    Returns array of next indices (shape = current_a_indices.shape).
    """
    N = current_a_indices.size
    N_A = P_A.shape[1]
    next_a = np.empty(N, dtype=np.int64)
    # sample in batches for each unique current a
    unique_a, inverse = np.unique(current_a_indices, return_inverse=True)
    for j, a_val in enumerate(unique_a):
        mask = (inverse == j)
        probs = P_A[a_val]
        draws = rng.choice(np.arange(N_A), size=mask.sum(), p=probs)
        next_a[mask] = draws
    return next_a


def simulate_panel(policy_k_idx: np.ndarray,
                   V: Optional[np.ndarray],
                   external_flag: Optional[np.ndarray],
                   K_grid: np.ndarray,
                   A_grid: np.ndarray,
                   P_A: np.ndarray,
                   params: Dict[str, Any],
                   N_firms: Optional[int] = None,
                   T: Optional[int] = None,
                   burn_in: Optional[int] = None,
                   init_method: str = "stationary",
                   seed: Optional[int] = 12345,
                   save_path_csv: Optional[str] = "./data/sim_panel.csv",
                   save_path_npz: Optional[str] = "./data/sim_panel.npz"
                   ) -> pd.DataFrame:
    """
    Simulate panel data for N_firms x T (post burn-in) using deterministic policy.

    Returns
    -------
    df : pandas.DataFrame with columns:
         ['firm_id','t','K','Kp','I','pi','q','E','ext_flag','A','a_idx','k_idx','kprime_idx']
    """
    # use defaults from params if not explicitly provided
    if N_firms is None:
        N_firms = int(params.get("N_firms", 1000))
    if T is None:
        T = int(params.get("T", 50))
    if burn_in is None:
        burn_in = int(params.get("burn_in", 50))

    # defensive cast to ints
    N_firms = int(N_firms)
    T = int(T)
    burn_in = int(burn_in)

    if N_firms <= 0 or T <= 0:
        raise ValueError("N_firms and T must be positive integers.")

    delta = float(params.get("delta", 0.15))
    alpha = float(params.get("alpha", 0.6956))
    p = float(params.get("p", 1.0))

    rng = np.random.default_rng(seed)

    K_grid = np.asarray(K_grid, dtype=float)
    A_grid = np.asarray(A_grid, dtype=float)
    P_A = np.asarray(P_A, dtype=float)
    policy_k_idx = np.asarray(policy_k_idx, dtype=np.int64)

    N_K = K_grid.size
    N_A = A_grid.size

    # small helper to flatten state index if needed
    def s_from_ia(i_idx: int, a_idx: int) -> int:
        return int(i_idx) * N_A + int(a_idx)

    total_periods = burn_in + T
    rows = N_firms * T  # guaranteed int

    # Defensive pre-allocation of arrays for post-burn-in storage
    firm_col = np.empty(rows, dtype=np.int32)
    t_col = np.empty(rows, dtype=np.int32)
    K_col = np.empty(rows, dtype=float)
    Kp_col = np.empty(rows, dtype=float)
    I_col = np.empty(rows, dtype=float)
    pi_col = np.empty(rows, dtype=float)
    q_col = np.empty(rows, dtype=float)
    E_col = np.empty(rows, dtype=float)
    ext_flag_col = np.empty(rows, dtype=np.int8)
    A_col = np.empty(rows, dtype=float)
    a_idx_col = np.empty(rows, dtype=np.int32)
    k_idx_col = np.empty(rows, dtype=np.int32)
    kprime_idx_col = np.empty(rows, dtype=np.int32)

    save_idx = 0

    # Initialize k_idx and a_idx for each firm
    if init_method == "stationary":
        # Build joint transition T of size SxS (S = N_K * N_A)
        S = N_K * N_A
        Tmat = np.zeros((S, S), dtype=float)
        for i_idx in range(N_K):
            for a_idx in range(N_A):
                s = s_from_ia(i_idx, a_idx)
                kprime = int(policy_k_idx[i_idx, a_idx])
                for a2 in range(N_A):
                    sp = s_from_ia(kprime, a2)
                    Tmat[s, sp] = P_A[a_idx, a2]
        # power method
        pi_state = np.ones(S, dtype=float) / S
        for _ in range(20000):
            pi_next = pi_state @ Tmat
            if np.max(np.abs(pi_next - pi_state)) < 1e-12:
                pi_state = pi_next
                break
            pi_state = pi_next
        # sample initial states
        s_indices = rng.choice(np.arange(S), size=N_firms, p=pi_state)
        k_indices = (s_indices // N_A).astype(np.int64)
        a_indices = (s_indices % N_A).astype(np.int64)

    elif init_method == "random":
        k_indices = rng.integers(0, N_K, size=N_firms, dtype=np.int64)
        a_indices = rng.integers(0, N_A, size=N_firms, dtype=np.int64)
    elif init_method == "zeros":
        k_indices = np.zeros(N_firms, dtype=np.int64)
        a_indices = np.full(N_firms, N_A // 2, dtype=np.int64)
    else:
        raise ValueError("init_method must be one of 'stationary','random','zeros'")

    # Simulate over periods
    for t in range(total_periods):
        # compute chosen k' for each firm at current state
        kprime_indices = policy_k_idx[k_indices, a_indices]  # shape (N_firms,)

        # compute per-firm values
        K_vals = K_grid[k_indices]
        Kp_vals = K_grid[kprime_indices]
        I_vals = Kp_vals - (1.0 - delta) * K_vals
        pi_vals_arr = A_grid[a_indices] * (K_vals ** alpha)
        E_vals = np.maximum(I_vals - pi_vals_arr, 0.0)

        if V is not None:
            V_vals = V[k_indices, a_indices]
            q_vals = V_vals / np.maximum(K_vals, 1e-12)
        else:
            q_vals = np.full_like(K_vals, np.nan)

        if external_flag is not None:
            ext_flags = external_flag[k_indices, a_indices].astype(np.int8)
        else:
            ext_flags = (E_vals > 1e-12).astype(np.int8)

        # Save post burn-in observations
        if t >= burn_in:
            for f in range(N_firms):
                firm_col[save_idx] = f
                t_col[save_idx] = t - burn_in
                K_col[save_idx] = float(K_vals[f])
                Kp_col[save_idx] = float(Kp_vals[f])
                I_col[save_idx] = float(I_vals[f])
                pi_col[save_idx] = float(pi_vals_arr[f])
                q_col[save_idx] = float(q_vals[f]) if not np.isnan(q_vals[f]) else np.nan
                E_col[save_idx] = float(E_vals[f])
                ext_flag_col[save_idx] = int(ext_flags[f])
                A_col[save_idx] = float(A_grid[a_indices[f]])
                a_idx_col[save_idx] = int(a_indices[f])
                k_idx_col[save_idx] = int(k_indices[f])
                kprime_idx_col[save_idx] = int(kprime_indices[f])
                save_idx += 1

        # advance A (stochastic) and K index (deterministic)
        a_indices = _draw_next_A_indices(a_indices, P_A, rng)
        k_indices = kprime_indices.copy()

    # Build DataFrame
    df = pd.DataFrame({
        "firm_id": firm_col,
        "t": t_col,
        "K": K_col,
        "Kp": Kp_col,
        "I": I_col,
        "pi": pi_col,
        "q": q_col,
        "E": E_col,
        "ext_flag": ext_flag_col,
        "A": A_col,
        "a_idx": a_idx_col,
        "k_idx": k_idx_col,
        "kprime_idx": kprime_idx_col
    })

    # Save outputs to relative paths
    out_dir = Path(save_path_csv).parent if save_path_csv is not None else Path("./data")
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_path_csv is not None:
        df.to_csv(save_path_csv, index=False)
    if save_path_npz is not None:
        # save numeric arrays in an npz for quick reload
        np.savez_compressed(save_path_npz,
                            firm_id=firm_col, t=t_col, K=K_col, Kp=Kp_col, I=I_col,
                            pi=pi_col, q=q_col, E=E_col, ext_flag=ext_flag_col,
                            A=A_col, a_idx=a_idx_col, k_idx=k_idx_col, kprime_idx=kprime_idx_col)

    return df
