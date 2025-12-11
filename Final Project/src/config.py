"""
Configuration helper for the trucking dynamic program.

This file has 3 responsibilities:
  1. Locate the project root
  2. Load params.json
  3. Create results/figures folders defined in params
"""

from pathlib import Path
import json


# ------------------------------------------------------------
#  FIND PROJECT ROOT 
# ------------------------------------------------------------
def get_project_root(start: Path = None, markers=None) -> Path:
    """
    Walk upward from the script location until we find a file/folder
    that identifies the project root.

    This means your code works on your machine AND your professor's.
    """
    if start is None:
        # src/config.py → parent is src/ → parent is project root candidate
        start = Path(__file__).resolve().parent.parent

    if markers is None:
        markers = ["run_all.sh", "requirements.txt", ".git"]

    current = start
    while True:
        for m in markers:
            if (current / m).exists():
                return current

        if current.parent == current:  # reached filesystem root
            raise RuntimeError(
                "Could not find project root. "
                "Add one of these files to your project: run_all.sh, requirements.txt, or .git"
            )

        current = current.parent


# ------------------------------------------------------------
#  LOAD PARAMS (JSON ONLY — REQUIRED)
# ------------------------------------------------------------
def load_params(path: str = "params.json", project_root: Path = None):
    """
    Load parameters from params.json.

    path: relative or absolute path to params.json
    """

    if project_root is None:
        project_root = get_project_root()

    p = Path(path)
    if not p.is_absolute():
        p = project_root / p

    if not p.exists():
        raise FileNotFoundError(
            f"Required parameter file not found: {p}\n"
            f"Make sure params.json is in your project root folder:\n"
            f"{project_root}"
        )

    with p.open() as f:
        params = json.load(f)

    # Add extra information (metadata)
    params["_params_path"] = str(p)
    params["_project_root"] = str(project_root)

    return params


# ------------------------------------------------------------
#  ENSURE OUTPUT DIRECTORIES EXIST
# ------------------------------------------------------------
def ensure_dirs(params: dict, keys=("results_dir", "figures_dir")):
    """
    Create directories for results/figures if missing.
    Uses relative paths based on project root.
    """
    root = Path(params.get("_project_root", "."))
    for k in keys:
        folder = params.get(k)
        if folder is None:
            continue
        p = Path(folder)
        if not p.is_absolute():
            p = root / p
        p.mkdir(parents=True, exist_ok=True)
