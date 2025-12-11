# tests/conftest.py
# Ensure the project root (the parent of this tests/ folder) is on sys.path
# so imports like `from src.config import ...` work when pytest collects tests.

import sys
from pathlib import Path

# project_root is one level above this tests directory
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

# Insert project root at front of sys.path if not present
pr_str = str(PROJECT_ROOT)
if pr_str not in sys.path:
    sys.path.insert(0, pr_str)

# (Optional) quick sanity print during collection if you want to debug:
# print("[conftest] added project root to sys.path:", pr_str)
