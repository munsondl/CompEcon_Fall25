# diagnose.py
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import load_params
from src.grids import integer_grid
from src.shocks import tauchen
from src.payoffs import profit  # NOW uses profit(k, z, params)

project_root = Path.cwd()
params = load_params(project_root=project_root)

# Build grids and shocks
k_grid = integer_grid(params["grid_k_min"], params["grid_k_max"])
z_grid, P = tauchen(
    n=5,
    mu=params["spot_rate"],
    rho=params["shock_rho"],
    sigma=params["shock_sigma"],
)

# Load policy.csv produced by run_all.sh
policy_csv = project_root / params["results_dir"] / "policy.csv"
if not policy_csv.exists():
    raise SystemExit(f"ERROR: {policy_csv} does not exist. Run ./run_all.sh first.")

policy = pd.read_csv(policy_csv, index_col=0)
policy = policy.values.astype(int)

Nk = len(k_grid)
Nz = len(z_grid)

# -------------------------------------------------------------------
# Threshold diagnostics: for each k, when does buying happen?
# -------------------------------------------------------------------
rows = []
for i, k in enumerate(k_grid):
    # buy threshold: smallest z where policy == k+1
    buy_idxs = [j for j in range(Nz) if (policy[i, j] == k + 1)]
    buy_threshold = float(z_grid[buy_idxs[0]]) if buy_idxs else None

    # sell region: anywhere policy < k
    sell_idxs = [j for j in range(Nz) if policy[i, j] < k]
    sell_max = float(z_grid[sell_idxs[-1]]) if sell_idxs else None

    rows.append({"k": int(k), "buy_threshold": buy_threshold, "sell_max_z": sell_max})

diag_df = pd.DataFrame(rows)
diag_path = project_root / params["results_dir"] / "diagnostics_thresholds.csv"
diag_df.to_csv(diag_path, index=False)
print("Saved thresholds to:", diag_path)
print(diag_df)

# -------------------------------------------------------------------
# Load value function for continuation comparisons
# -------------------------------------------------------------------
value_csv = project_root / params["results_dir"] / "value.csv"
if not value_csv.exists():
    raise SystemExit("ERROR: value.csv missing; run run_all.sh first.")

V = pd.read_csv(value_csv, index_col=0).values.astype(float)

# helper for E[V(k', z')]
def EV_of(k_prime, z_idx):
    k_prime_idx = np.where(k_grid == k_prime)[0][0]
    return float(np.dot(P[z_idx], V[k_prime_idx, :]))

# -------------------------------------------------------------------
# Sample state diagnostics: compare hold vs buy at selected (k,z)
# -------------------------------------------------------------------
print("\nSample state diagnostics (hold vs buy)")
print("k,z,z_val, hold_pi, buy_pi, EV_hold, EV_buy, buy_minus_hold")

sample_rows = []
for k in [0, 2, 5, 8]:
    for j, z in enumerate(z_grid):
        hold_pi = profit(k, z, params)
        buy_pi = profit(k, z, params) - (
            # Buying cost = price(z) + fixed_cost
            params["price_per_fleet_unit"] * (z / params["spot_rate"]) +
            params["fixed_cost"]
        )
        EV_hold = EV_of(k, j)
        k_next = k + 1 if k < k_grid[-1] else k
        EV_buy  = EV_of(k_next, j)

        total_hold = hold_pi + params["beta"] * EV_hold
        total_buy  = buy_pi + params["beta"] * EV_buy

        sample_rows.append([
            k, j, float(z),
            float(hold_pi),
            float(buy_pi),
            float(EV_hold),
            float(EV_buy),
            float(total_buy - total_hold)
        ])

sample_df = pd.DataFrame(sample_rows, columns=[
    "k","z_idx","z_val",
    "hold_pi","buy_pi","EV_hold","EV_buy","buy_minus_hold"
])

# print top 10 states where buy is most attractive
print(sample_df.sort_values("buy_minus_hold", ascending=False).head(10))

# -------------------------------------------------------------------
# Print simulation summary
# -------------------------------------------------------------------
sim_summary = project_root / params["results_dir"] / "sim_summary.csv"
if sim_summary.exists():
    print("\nSimulation summary CSV:")
    print(pd.read_csv(sim_summary, index_col=0).to_string())
else:
    print("\nNo sim_summary.csv found. Run run_all.sh first.")
