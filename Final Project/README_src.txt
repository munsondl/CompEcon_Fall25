
src package for CompEcon final project.

Modules:
  - config.py     : load/save parameters (JSON/YAML)
  - grids.py      : grid constructors
  - shocks.py     : Tauchen discretization
  - payoffs.py    : immediate payoffs and cost functions
  - solver.py     : value function iteration example
  - simulate.py   : forward simulation helpers
  - analysis.py   : numeric moments and diagnostics
  - utils.py      : helpers and IO

How to use:
  1. Copy the `src` directory into your project root:
     C:\Users\David Munson\Desktop\git\CompEcon_Fall25\Final Project
  2. In your scripts or notebooks, import with:
     from src import grids, shocks, solver, simulate, config, utils
