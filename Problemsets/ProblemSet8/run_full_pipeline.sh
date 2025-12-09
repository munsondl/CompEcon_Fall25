#!/usr/bin/env bash
set -euo pipefail
# run_full_pipeline.sh
# Full pipeline driver for Table 3 replication (safe, staged).
# Edit the FINAL CONFIG block to run a full, expensive replication.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"   # you can set PYTHON=/path/to/python before running
RESULTS_DIR="$PROJECT_ROOT/results"
DATA_DIR="$PROJECT_ROOT/data"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$RESULTS_DIR" "$DATA_DIR" "$LOG_DIR"

echo "=== Run Full Pipeline ==="
echo "Project root: $PROJECT_ROOT"
echo "Results: $RESULTS_DIR"
echo "Data: $DATA_DIR"
echo "Logs: $LOG_DIR"
echo "Python executable: $PYTHON"
echo

# -------------------------
# CONFIG: quick/test settings
# -------------------------
# These are conservative defaults for testing. Edit the FINAL CONFIG section below for final runs.
NK_TEST=120
NA_TEST=7
N_FIRMS_TEST=500
T_TEST=50
BURN_IN_TEST=30

DE_POPSIZE_TEST=8
DE_MAXITER_TEST=6
LOCAL_MAXITER_TEST=50

S_PANELS_TEST=20   # small number for testing covariance routine

# -------------------------
# FINAL CONFIG (edit for final run)
# -------------------------
# WARNING: these settings are expensive. Use them only when you're ready for the full replication.
NK_FINAL=600            # ~600 recommended (paper used large grid)
NA_FINAL=7
N_FIRMS_FINAL=1000      # or larger to match the paper
T_FINAL=200             # example — set to match paper's sim horizon
BURN_IN_FINAL=50

DE_POPSIZE_FINAL=15
DE_MAXITER_FINAL=30
LOCAL_MAXITER_FINAL=200

S_PANELS_FINAL=500      # number of panels to estimate S_psi (very expensive)

# -------------------------
# Helper: run python snippet with cwd at project root, capture logs
# -------------------------
run_python() {
  local label="$1"; shift
  local logfile="$LOG_DIR/${label// /_}.log"
  echo ">> Running: $label  (log -> $logfile)"
  # run Python with unbuffered output to logfile (also tee to console)
  (cd "$PROJECT_ROOT" && "$PYTHON" -u - "$@" 2>&1) | tee "$logfile"
}

# -------------------------
# Stage 0: Pre-checks
# -------------------------
echo "Stage 0: quick environment checks"
"$PYTHON" - <<PY
import sys, pkgutil
need = ["numpy","scipy","pandas"]
missing = [p for p in need if pkgutil.find_loader(p) is None]
if missing:
    print("Warning: missing packages:", missing, file=sys.stderr)
    print("Please pip install:", " ".join(missing))
else:
    print("Basic packages available:", need)
print("Python executable:", sys.executable)
PY

# -------------------------
# Stage 1: quick_check (stationary evaluation)
# -------------------------
echo
echo "=================================="
echo "Stage 1: Quick smoke test (stationary moments)"
echo "=================================="
# create a small python driver that calls quick_check logic
"$PYTHON" - <<PY | tee "$LOG_DIR"/stage1_quick_check.out
from pathlib import Path
from modules.parameters import init_params
from modules.smm_objective import smm_objective, moment_targets
import numpy as np, json, pprint

params = init_params(N_K=$NK_TEST, N_A=$NA_TEST, N_firms=$N_FIRMS_TEST, T=$T_TEST, burn_in=$BURN_IN_TEST,
                     vfi_tol=1e-6, vfi_maxiter=1000)
mapping = ["alpha","gamma","rho","sigma","phi_tilde_0"]
theta0 = [0.6956, 0.1331, 0.0976, 0.8932, 0.0]

J, info = smm_objective(theta0, mapping, params, W=None, use_panel=False,
                        solver_opts={"tol":1e-6, "maxiter":1000, "K_spacing":"power"})
print("Objective J:", J)
print("Simulated moments (stationary):")
pprint.pprint(info["moments_sim"])
# Write compact summary
Path("results").mkdir(parents=True, exist_ok=True)
out = {"stage":"quick_check","J":float(J),"moments":info["moments_sim"]}
with open("results/stage1_quick_check.json","w") as f:
    json.dump(out,f,indent=2)
print("Saved results/stage1_quick_check.json")
PY

# -------------------------
# Stage 2: panel_check (single panel evaluation)
# -------------------------
echo
echo "=================================="
echo "Stage 2: Panel smoke test (single panel evaluation)"
echo "=================================="
"$PYTHON" panel_check.py

# -------------------------
# Stage 3: Stage-1 SMM (global search with use_panel=False)
# -------------------------
echo
echo "=================================="
echo "Stage 3: Stage-1 SMM (global search, W = I, stationary moments)"
echo "=================================="
# This uses modules.optimize_smm.optimize_smm. We run a Python snippet to call it.
"$PYTHON" - <<PY | tee "$LOG_DIR"/stage3_stage1_smm.out
from modules.parameters import init_params
from modules.optimize_smm import optimize_smm
import json
params = init_params(N_K=$NK_TEST, N_A=$NA_TEST, N_firms=$N_FIRMS_TEST, T=$T_TEST, burn_in=$BURN_IN_TEST,
                     vfi_tol=1e-6, vfi_maxiter=1000)
mapping = ["alpha","gamma","rho","sigma","phi_tilde_0"]
theta0 = [0.6956, 0.1331, 0.0976, 0.8932, 0.0]
gres = optimize_smm(mapping=mapping, params_base=params, theta0=theta0,
                    global_opts={"global_method":"diffev","de_popsize":$DE_POPSIZE_TEST,"de_maxiter":$DE_MAXITER_TEST,"seed":1234},
                    local_opts={"local_method":"L-BFGS-B","options":{"maxiter":$LOCAL_MAXITER_TEST}},
                    smm_opts={"use_panel":False,"solver_opts":{"tol":1e-6,"maxiter":1000}},
                    save_path="./results/opt_result_stage1.json",
                    verbose=True)
print("Stage-1 saved to results/opt_result_stage1.json")
print("Best J:", gres.get("fun") if "fun" in gres else gres.get("x"))
PY

# -------------------------
# Stage 4: Estimate moment covariance S_psi at theta_stage1
# -------------------------
echo
echo "=================================="
echo "Stage 4: Estimate S_psi (simulate S panels at theta_stage1)"
echo "=================================="
# This step is expensive. We run a small S for testing (S_PANELS_TEST). For final run set S_PANELS_FINAL.
"$PYTHON" - <<PY | tee "$LOG_DIR"/stage4_Spsi.out
import json, numpy as np
from pathlib import Path
from modules.parameters import init_params
from modules.std_errors import estimate_moment_covariance
# load theta from previous result
optp = Path("results/opt_result_stage1.json")
if not optp.exists():
    raise SystemExit("Stage-1 opt result not found: results/opt_result_stage1.json")
opt = json.load(optp.open())
theta = opt.get("x", opt.get("theta", None))
mapping = opt.get("mapping", None)
if theta is None or mapping is None:
    raise SystemExit("opt_result_stage1.json missing theta/mapping")
params = init_params(N_K=$NK_TEST, N_A=$NA_TEST, N_firms=$N_FIRMS_TEST, T=$T_TEST, burn_in=$BURN_IN_TEST,
                     vfi_tol=1e-6, vfi_maxiter=1000)
S = $S_PANELS_TEST
panel_args = {"mapping": mapping, "theta": theta, "seed":2025, "N_firms":int(params.get("N_firms")), "T":int(params.get("T")), "burn_in":int(params.get("burn_in")), "save_path_csv":None, "save_path_npz":None}
S_psi, info = estimate_moment_covariance(np.asarray(theta), mapping, params, S_panels=S, panel_args=panel_args, use_panel_simulator=True, save_path="./data/moment_cov_test.npz")
print("Estimated S_psi (shape):", S_psi.shape)
np.savez_compressed("results/S_psi_test.npz", S_psi=S_psi)
print("Saved results/S_psi_test.npz")
PY

# -------------------------
# Stage 5: Stage-2 SMM (re-estimate using W = S_psi^{-1})
# -------------------------
echo
echo "=================================="
echo "Stage 5: Stage-2 SMM (two-step re-estimation using S_psi^{-1})"
echo "=================================="
"$PYTHON" - <<PY | tee "$LOG_DIR"/stage5_stage2_smm.out
import json, numpy as np
from pathlib import Path
from modules.parameters import init_params
from modules.optimize_smm import optimize_smm
# load stage-1 theta
optp = Path("results/opt_result_stage1.json")
opt = json.load(optp.open())
theta0 = opt.get("x", opt.get("theta"))
mapping = opt.get("mapping")
# load S_psi
sfile = Path("results/S_psi_test.npz")
if not sfile.exists():
    sfile = Path("data/moment_cov_test.npz")
if not sfile.exists():
    raise SystemExit("S_psi not found; run stage 4 first")
S = np.load(str(sfile))
S_psi = S.get("S_psi") if "S_psi" in S else S["arr_0"]
Wmat = np.linalg.pinv(S_psi)
params = init_params(N_K=$NK_TEST, N_A=$NA_TEST, N_firms=$N_FIRMS_TEST, T=$T_TEST, burn_in=$BURN_IN_TEST,
                     vfi_tol=1e-6, vfi_maxiter=1000)
gres2 = optimize_smm(mapping=mapping, params_base=params, theta0=theta0,
                     global_opts={"global_method":"diffev","de_popsize":$DE_POPSIZE_TEST,"de_maxiter":6,"seed":1234},
                     local_opts={"local_method":"L-BFGS-B","options":{"maxiter":$LOCAL_MAXITER_TEST}},
                     smm_opts={"use_panel":False,"solver_opts":{"tol":1e-6,"maxiter":1000}},
                     save_path="./results/opt_result_stage2.json",
                     verbose=True)
print("Stage-2 result saved to results/opt_result_stage2.json")
PY

# -------------------------
# Stage 6: Compute standard errors (expensive)
# -------------------------
echo
echo "=================================="
echo "Stage 6: Compute standard errors (Jacobian + sandwich; expensive)"
echo "=================================="
"$PYTHON" - <<PY | tee "$LOG_DIR"/stage6_std_errors.out
import json, numpy as np
from pathlib import Path
from modules.parameters import init_params
from modules.std_errors import compute_smm_standard_errors
# load final theta (from stage2 if available else stage1)
p = Path("results/opt_result_stage2.json")
if not p.exists():
    p = Path("results/opt_result_stage1.json")
opt = json.load(p.open())
theta = opt.get("x", opt.get("theta"))
mapping = opt.get("mapping")
params = init_params(N_K=$NK_TEST, N_A=$NA_TEST, N_firms=$N_FIRMS_TEST, T=$T_TEST, burn_in=$BURN_IN_TEST,
                     vfi_tol=1e-6, vfi_maxiter=1000)
# Small test of std errors (use S_panels small). For final run set S_panels to $S_PANELS_FINAL
cov_theta, std_errs, diag = compute_smm_standard_errors(np.asarray(theta), mapping, params,
                                                        S_panels=$S_PANELS_TEST,
                                                        panel_args={"mapping":mapping,"theta":theta,"seed":2025},
                                                        eps=1e-4, use_general_sandwich=False)
import json
out = {"std_err": list(std_errs), "mapping": mapping}
with open("results/smm_std_errors_test.json","w") as f:
    json.dump(out,f,indent=2)
print("Saved results/smm_std_errors_test.json")
PY

# -------------------------
# Stage 7: Produce Table 3 outputs
# -------------------------
echo
echo "=================================="
echo "Stage 7: Produce Table 3 CSV/JSON outputs"
echo "=================================="
"$PYTHON" - <<PY | tee "$LOG_DIR"/stage7_output_table.out
from modules.output_table import make_table3_data
make_table3_data(opt_json_path="./results/opt_result_stage2.json",
                 moments_json_path=None,
                 std_errors_json_path="./results/smm_std_errors_test.json",
                 out_dir="./results/")
print("Table 3 files saved to ./results/")
PY

echo
echo "=== Pipeline finished (logs in $LOG_DIR). Review results/ for outputs. ==="
