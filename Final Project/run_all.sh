#!/usr/bin/env bash
# Run the full pipeline for the trucking project.
# Usage: bash run_all.sh
# Make sure to run from the project root (the folder containing params.json).

set -euo pipefail

echo "===== RUN ALL: trucking model pipeline ====="
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project root: $ROOT"
cd "$ROOT"

# --- optional: run tests first (uncomment if you want) ---
# echo "Running pytest..."
# pytest -q

echo "Loading params and running value function iteration..."

python - <<'PY'
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path().resolve()
sys.path.insert(0, str(project_root / "src"))

# Import modules
from src.config import load_params
from src.grids import integer_grid
from src.shocks import tauchen
from src.solver import value_iteration
from src.payoffs import profit
from src.analysis import run_analysis

# Load params
params = load_params(project_root=project_root)
print("Loaded params from:", params.get("_params_path", "params.json"))

# Build grids and shocks
k_grid = integer_grid(params["grid_k_min"], params["grid_k_max"])

z_grid, P = tauchen(
    n=5,
    mu=params["spot_rate"],
    rho=params["shock_rho"],
    sigma=params["shock_sigma"],
)

print("k_grid:", k_grid)
print("z_grid (sample):", z_grid[:3], "...")

# CORRECTED payoff wrapper (new signature)
def payoff_wrapper(k, z, params_local):
    return profit(k, z, params_local)

# Run VFI (new solver signature)
print("Running value function iteration (this may take a few seconds)...")
V, pol = value_iteration(
    k_grid=k_grid,
    z_grid=z_grid,
    P=P,
    beta=params["beta"],
    payoff_fn=payoff_wrapper,
    params=params,
    tol=1e-5,
    max_iter=2000,
)

print("VFI complete.")

# Run analysis (saves results to results/ and results/figures/)
print("Running analysis and saving outputs...")
results = run_analysis(V, pol, k_grid, z_grid, P, params, simulate_T=200, seed=0)

print("Analysis done. Saved files:")
for k, v in results.items():
    if k.endswith("_fig") or k.endswith("_csv") or k.endswith("_txt") or k == "sim_output":
        print(f"  {k}: {v}")

print("\nTop-level summary:")
import pandas as pd
summary_csv = Path(results["sim_summary_csv"])
if summary_csv.exists():
    df = pd.read_csv(summary_csv, index_col=0)
    print(df.to_string())
else:
    print("No sim summary CSV found at", summary_csv)

print("\nAll finished. Check the results/ directory for CSVs and figures.")

PY

echo "===== COMPILING LATEX DOCUMENT ====="

# Compile LaTeX (run from project root)
pdflatex FinalProject_Munson.tex
bibtex FinalProject_Munson
pdflatex FinalProject_Munson.tex
pdflatex FinalProject_Munson.tex

if [ -f "FinalProject_Munson.pdf" ]; then
    echo "LaTeX compilation successful: FinalProject_Munson.pdf created."
else
    echo "LaTeX compilation failed. Check .log file for details."
fi
echo "===== COMPILING LATEX DOCUMENT ====="

# Compile LaTeX (run from project root)
pdflatex FinalProject_Munson.tex
bibtex FinalProject_Munson
pdflatex FinalProject_Munson.tex
pdflatex FinalProject_Munson.tex

if [ -f "FinalProject_Munson.pdf" ]; then
    echo "LaTeX compilation successful: FinalProject_Munson.pdf created."
else
    echo "LaTeX compilation failed. Check .log file for details."
fi

echo "===== COMPILING LATEX DOCUMENT ====="

# Compile LaTeX (run from project root)
pdflatex FinalProject_Munson.tex
bibtex FinalProject_Munson
pdflatex FinalProject_Munson.tex
pdflatex FinalProject_Munson.tex

if [ -f "FinalProject_Munson.pdf" ]; then
    echo "LaTeX compilation successful: FinalProject_Munson.pdf created."
else
    echo "LaTeX compilation failed. Check .log file for details."
fi

echo "===== COMPILING LATEX DOCUMENT ====="

# Compile LaTeX (run from project root)
pdflatex FinalProject_Munson.tex
bibtex FinalProject_Munson
pdflatex FinalProject_Munson.tex
pdflatex FinalProject_Munson.tex

if [ -f "FinalProject_Munson.pdf" ]; then
    echo "LaTeX compilation successful: FinalProject_Munson.pdf created."
else
    echo "LaTeX compilation failed. Check .log file for details."
fi

echo "===== COMPILING LATEX DOCUMENT ====="

# Compile LaTeX (run from project root)
pdflatex FinalProject_Munson.tex
bibtex FinalProject_Munson
pdflatex FinalProject_Munson.tex
pdflatex FinalProject_Munson.tex

if [ -f "FinalProject_Munson.pdf" ]; then
    echo "LaTeX compilation successful: FinalProject_Munson.pdf created."
else
    echo "LaTeX compilation failed. Check .log file for details."
fi

echo "===== PIPELINE FINISHED ====="
