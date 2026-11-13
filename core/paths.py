"""Project paths. Relative paths are resolved from the repository root.

Experiment index (aligned with the paper pipeline):
  exp1  IPPO pretraining          -> results/exp1_ippo
  exp2  Static drift degradation  -> results/exp2_drift
  exp3  CM training               -> results/exp3_train_cm
  exp5  Multi-scenario sensitivity -> results/exp5_sensitivity
  exp6  Main table                -> results/exp6_threshold
  exp7  Refresh ablation and co-movement -> results/exp7_refresh
  exp10 Drift frequency and amplitude -> results/exp10_drift_factors
  exp12 Edge scale and bandwidth  -> results/exp12_system (local CPU; latency uses the SD3 benchmark step time)
  exp14 Controlled cache reuse    -> results/exp14_controlled (not in run_all; GPU job on the cluster)
"""
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXP1_DIR = "./results/exp1_ippo"
EXP1_IPPO = f"{EXP1_DIR}/trained_ippo.pt"
EXP2_DIR = "./results/exp2_drift"
EXP3_DIR = "./results/exp3_train_cm"
EXP3_CM = f"{EXP3_DIR}/trained_cm.pt"
EXP5_DIR = "./results/exp5_sensitivity"
EXP6_DIR = "./results/exp6_threshold"
EXP6_THRESHOLDS = f"{EXP6_DIR}/selected_thresholds.json"
EXP7_DIR = "./results/exp7_refresh"
EXP10_DIR = "./results/exp10_drift_factors"
EXP12_DIR = "./results/exp12_system"
EXP14_DIR = "./results/exp14_controlled"


def ensure_project_cwd() -> None:
    """Change the working directory to the project root so ./results/... relative paths work."""
    os.chdir(ROOT_DIR)


def norm_path(path: str) -> str:
    """Normalize a path string. Relative paths stay relative."""
    return os.path.normpath(path)
