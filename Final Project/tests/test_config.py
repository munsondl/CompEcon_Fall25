# tests/test_config_real.py
"""
Real integration test: ensure load_params() reads params.json from the project root
and ensure_dirs() can create the configured output folders.
"""

from src.config import load_params, ensure_dirs
from pathlib import Path
import shutil


def test_load_real_params_and_ensure_dirs():
    # 1) Determine project root: assume test is run from project root
    project_root = Path.cwd()  # must run pytest from project root

    # FIXED: do NOT pass path=None
    params = load_params(project_root=project_root)

    # Basic checks: params.json path exists and metadata present
    assert "_params_path" in params, "load_params() should attach _params_path metadata"
    params_path = Path(params["_params_path"])
    assert params_path.exists(), f"params.json not found at {params_path}"

    # 2) Prepare to call ensure_dirs() but track whether directories existed beforehand
    results_dir = project_root / Path(params.get("results_dir", "results"))
    figures_dir = project_root / Path(params.get("figures_dir", "results/figures"))

    created_results = False
    created_figures = False

    pre_exists_results = results_dir.exists()
    pre_exists_figures = figures_dir.exists()

    try:
        ensure_dirs(params)

        # Assert they exist after ensure_dirs
        assert results_dir.exists()
        assert figures_dir.exists()

        if not pre_exists_results and results_dir.exists():
            created_results = True
        if not pre_exists_figures and figures_dir.exists():
            created_figures = True

    finally:
        try:
            if created_figures and figures_dir.exists():
                shutil.rmtree(figures_dir)
            if created_results and results_dir.exists():
                shutil.rmtree(results_dir)
        except Exception as e:
            print("Warning: cleanup failed:", e)
