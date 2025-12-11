"""
Analysis helpers for the trucking dynamic program.

Produces outputs (CSV + figures) into results_dir:
  1) policy CSV + policy heatmap
  2) value CSV + value heatmap
  3) simulation time series + simulation summary CSV + TXT + histogram

This file expects:
  - simulate(...) returns 'payoffs' already computed via profit(k, z, params)
  - price(z) = price_base * (z / spot_mean) for computing ROC
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Helpers to ensure output directories exist
# -----------------------------------------------------------------------------
def ensure_results_dir(params):
    root = Path(params.get("_project_root", Path.cwd()))
    results_dir = root / params.get("results_dir", "results")
    figures_dir = root / params.get("figures_dir", "results/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, figures_dir


# -----------------------------------------------------------------------------
# Policy saving / plotting
# -----------------------------------------------------------------------------
def save_policy(policy, k_grid, z_grid, params, fname="policy"):
    """
    Save policy (k' choices) to CSV and create a heatmap figure.
    """
    results_dir, figures_dir = ensure_results_dir(params)

    df = pd.DataFrame(policy, index=k_grid, columns=[f"{z:.3f}" for z in z_grid])
    csv_path = results_dir / f"{fname}.csv"
    df.to_csv(csv_path, index_label="k")

    fig_path = figures_dir / f"{fname}_heatmap.png"
    plt.figure(figsize=(8, 6))
    plt.imshow(policy, aspect="auto", origin="lower")
    plt.colorbar(label="k' (next period fleet size)")
    plt.xlabel("z index")
    plt.ylabel("k (fleet size)")
    plt.title("Policy function: chosen k' (rows=k, cols=z-index)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return csv_path, fig_path


# -----------------------------------------------------------------------------
# Value saving / plotting
# -----------------------------------------------------------------------------
def save_value(V, k_grid, z_grid, params, fname="value"):
    """
    Save value function to CSV and create a heatmap figure.
    """
    results_dir, figures_dir = ensure_results_dir(params)

    df = pd.DataFrame(V, index=k_grid, columns=[f"{z:.3f}" for z in z_grid])
    csv_path = results_dir / f"{fname}.csv"
    df.to_csv(csv_path, index_label="k")

    fig_path = figures_dir / f"{fname}_heatmap.png"
    plt.figure(figsize=(8, 6))
    plt.imshow(V, aspect="auto", origin="lower")
    plt.colorbar(label="Value (thousands)")
    plt.xlabel("z index")
    plt.ylabel("k (fleet size)")
    plt.title("Value function V(k, z)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return csv_path, fig_path


# -----------------------------------------------------------------------------
# Simulation & summary
# -----------------------------------------------------------------------------
def simulate_and_summarize(policy, k_grid, z_grid, P, params, T=200, seed=0):
    """
    Run a forward simulation under `policy` and save:
      - timeseries plot (fleet and z)
      - summary CSV with statistics
      - extra stats: median profit, percent time at k=0
      - histogram of fleet distribution
      - printable short summary text file

    Returns:
      out (simulation dict), summary_csv path, timeseries fig path, histogram fig path, summary text path
    """
    from src.simulate import simulate

    results_dir, figures_dir = ensure_results_dir(params)

    # Simulate T periods
    out = simulate(
        policy=policy,
        k_grid=k_grid,
        z_grid=z_grid,
        P=P,
        T=T,
        k0=int(k_grid[0]),
        z0_idx=len(z_grid) // 2,
        params=params,
        seed=seed,
    )

    k_path = out["k_path"]            # length T+1
    z_vals = out["z_vals"]            # length T+1
    payoffs = out["payoffs"]          # length T, already profit(k,z,params)

    # Use period t values (0..T-1)
    z_for_period = z_vals[:-1]
    k_for_period = k_path[:-1]

    # revenue per period (thousands): z * k
    revenue = z_for_period * k_for_period

    # price(z) = price_base * (z / spot_mean)
    price_base = float(params["price_per_fleet_unit"])
    spot_mean = float(params["spot_rate"])
    # price_z is length T
    price_z = price_base * (z_for_period / spot_mean)

    profit_k = payoffs  # already computed by simulate using profit(k,z,params)

    # Return on capital per period (profit divided by state-dependent asset value)
    # Avoid division by zero: replace zeros (if any) in price_z with a tiny positive number
    price_z_safe = np.where(price_z <= 0, 1e-8, price_z)
    roc = profit_k / price_z_safe

    # Summary statistics (original)
    summary = {
        "mean_fleet": float(np.mean(k_for_period)),
        "median_fleet": float(np.median(k_for_period)),
        "std_fleet": float(np.std(k_for_period)),
        "mean_profit_k": float(np.mean(profit_k)),
        "mean_revenue_k": float(np.mean(revenue)),
        "mean_return_on_capital": float(np.mean(roc)),
    }

    # Additional statistics requested earlier
    summary["median_profit_k"] = float(np.median(profit_k))
    summary["pct_time_at_k0"] = float(100.0 * np.mean(k_for_period == 0))

    # Save summary CSV
    summary_df = pd.DataFrame.from_dict(summary, orient="index", columns=["value"])
    summary_csv = results_dir / "sim_summary.csv"
    summary_df.to_csv(summary_csv, header=True)

    # Save a short printable summary text file
    summary_txt = results_dir / "sim_summary.txt"
    with open(summary_txt, "w") as f:
        f.write("Simulation summary\n")
        f.write("==================\n")
        for key, val in summary.items():
            f.write(f"{key}: {val}\n")
        f.write("\nNotes: profits and revenues are in thousands. ROC uses price(z)=price_base*(z/spot_mean).\n")

    # Time series plot: fleet and spot rate
    fig_path = figures_dir / "sim_timeseries.png"
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax1.plot(k_path, label="fleet size (k)")
    ax1.set_xlabel("time")
    ax1.set_ylabel("fleet size (k)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(z_vals, color="0.2", linestyle="--", label="spot rate (z)")
    ax2.set_ylabel("spot rate (thousands per truck)")

    # legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Simulation under computed policy: fleet and spot rate")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # Fleet distribution histogram
    hist_fig = figures_dir / "sim_histogram.png"
    plt.figure(figsize=(8, 4))
    data = k_for_period
    bins = np.arange(int(data.min()), int(data.max()) + 2) - 0.5
    plt.hist(data, bins=bins, edgecolor="black")
    plt.xlabel("fleet size (k)")
    plt.ylabel("count")
    plt.title("Distribution of fleet size over simulation")
    plt.tight_layout()
    plt.savefig(hist_fig, dpi=150)
    plt.close()

    return out, summary_csv, fig_path, hist_fig, summary_txt


# -----------------------------------------------------------------------------
# Top-level runner
# -----------------------------------------------------------------------------
def run_analysis(V, policy, k_grid, z_grid, P, params, simulate_T=200, seed=0):
    """
    Run the analysis pipeline and save outputs to results/ directory.
    Returns a dict with file locations and the simulation output.
    """
    results = {}

    p_csv, p_fig = save_policy(policy, k_grid, z_grid, params, fname="policy")
    results["policy_csv"] = str(p_csv)
    results["policy_fig"] = str(p_fig)

    v_csv, v_fig = save_value(V, k_grid, z_grid, params, fname="value")
    results["value_csv"] = str(v_csv)
    results["value_fig"] = str(v_fig)

    sim_out, sim_csv, sim_fig, hist_fig, summary_txt = simulate_and_summarize(
        policy, k_grid, z_grid, P, params, T=simulate_T, seed=seed
    )

    results["sim_summary_csv"] = str(sim_csv)
    results["sim_timeseries_fig"] = str(sim_fig)
    results["sim_histogram_fig"] = str(hist_fig)
    results["sim_summary_txt"] = str(summary_txt)
    results["sim_output"] = sim_out

    return results
