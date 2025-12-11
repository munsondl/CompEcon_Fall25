"""Utility functions and small IO helpers.

Examples: parallel_map, ensure_dir, small wrappers for saving results.
"""
import os
from pathlib import Path
from functools import partial
import multiprocessing as mp
import json

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def parallel_map(fn, iterable, processes=None):
    processes = processes or max(1, mp.cpu_count() - 1)
    with mp.Pool(processes) as pool:
        return pool.map(fn, iterable)
