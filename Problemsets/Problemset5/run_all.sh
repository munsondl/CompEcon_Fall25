#!/usr/bin/env bash
set -euo pipefail

# Run this from ProblemSet5/ directory: ./run_all.sh

echo "1) Create data (may use cached file)..."
python3 - <<'PY'
from src.clean_data import clean_data
df = clean_data()
print("Data rows:", len(df))
PY

echo "2) Create visualizations..."
python3 - <<'PY'
from src.clean_data import clean_data
from src.visualize import create_visuals
df = clean_data()
files = create_visuals(df)
print("Saved images:", files)
PY

echo "3) Estimate models and write results table..."
python3 - <<'PY'
from src.clean_data import clean_data
from src.estimate_model import estimate_models
df = clean_data()
res = estimate_models(df)
print("Saved summary table.")
PY

echo "4) Compile LaTeX (ProblemSet5_LastName.tex)."
# tries pdflatex twice to resolve references. May fail if Tex not installed.
pdflatex -interaction=nonstopmode ProblemSet5_LastName.tex || true
pdflatex -interaction=nonstopmode ProblemSet5_LastName.tex || true

echo "Done. Output: ProblemSet5_LastName.pdf"
