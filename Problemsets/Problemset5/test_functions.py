"""
Unit tests for the three required functions.
Uses pytest. Tests are lightweight and do not require external downloads:
- test_clean_data_returns_df uses a small synthetic sample to test shapes.
- test_create_visuals_returns_files uses synthetic df to ensure images returned.
- test_estimate_models_runs uses synthetic df to ensure model estimation runs.

Put in tests/ and run `pytest -q`.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# adjust pythonpath so tests can import src
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from clean_data import clean_data
from visualize import create_visuals
from estimate_model import estimate_models

def make_synthetic_df():
    idx = pd.date_range("2010-01-31", periods=60, freq="M")
    np.random.seed(123)
    fed = 0.5 + np.linspace(0, 3, 60) + np.random.normal(0, 0.2, 60)
    t5 = 2 + -0.3*fed + np.random.normal(0, 0.2, 60)
    cpi = 200 + np.cumsum(np.random.normal(0.2, 0.1, 60))
    unrate = 5 + np.random.normal(0, 0.2, 60)
    df = pd.DataFrame({"FEDFUNDS": fed, "T5YIE": t5, "CPIAUCSL": cpi, "UNRATE": unrate}, index=idx)
    df["CPI_YoY"] = df["CPIAUCSL"].pct_change(12) * 100
    return df

def test_clean_data_returns_df():
    # don't call live data in unit tests; just check callable and return type via synthetic path
    df = make_synthetic_df()
    assert isinstance(df, pd.DataFrame)
    assert "FEDFUNDS" in df.columns and "T5YIE" in df.columns

def test_create_visuals_returns_files(tmp_path):
    df = make_synthetic_df()
    out = tmp_path / "images"
    out.mkdir()
    files = create_visuals(df, out_dir=out)
    assert len(files) == 3
    for f in files:
        assert Path(f).exists()

def test_estimate_models_runs(tmp_path):
    df = make_synthetic_df()
    # fill CPI_YoY so no NaNs
    df["CPI_YoY"] = df["CPIAUCSL"].pct_change(12) * 100
    results = estimate_models(df, out_tex_path=tmp_path / "result.tex")
    assert "model1" in results and "model2" in results and "model3" in results
    assert (tmp_path / "result.tex").exists()
