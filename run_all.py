"""Experiment pipeline.

Default order: exp1 -> exp2 -> exp3 -> exp5 -> exp6 -> exp7 -> exp10

  python run_all.py
  python run_all.py --from exp6
  python run_all.py --only exp6
  python run_all.py --dry-run

Subprocesses use PYTHON_BIN or CONDA_PYTHON when that file exists, otherwise the current interpreter.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_env_python = os.environ.get("PYTHON_BIN", "").strip() or os.environ.get("CONDA_PYTHON", "").strip()
CONDA_PYTHON = Path(_env_python) if _env_python else Path(sys.executable)
PYTHON = str(CONDA_PYTHON if CONDA_PYTHON.is_file() else Path(sys.executable))

Step = tuple[str, str, list[str], bool]

PIPELINE: list[Step] = [
    ("exp1", "Static-HIS IPPO pretraining", [PYTHON, "experiments/exp1_train_ippo.py"], True),
    ("exp2", "Static-HIS degradation under drift", [PYTHON, "experiments/exp2_drift_degradation.py"], True),
    ("exp3", "CM training -> trained_cm.pt", [PYTHON, "experiments/exp3_train_cm.py"], True),
    ("exp5", "Sensitivity across drift scenarios (Static/IIS/NIRVANA vs default ET)", [PYTHON, "experiments/exp5_sensitivity.py"], True),
    ("exp6", "Threshold and periodic triggers", [PYTHON, "experiments/exp6_threshold.py"], True),
    ("exp7", "Refresh ablation and co-movement", [PYTHON, "experiments/exp7_refresh.py"], True),
    ("exp10", "Drift frequency and amplitude", [PYTHON, "experiments/exp10_drift_factors.py"], True),
]

NAMES = [name for name, _, _, _ in PIPELINE]


def _slice_pipeline(from_step: str | None, only: list[str] | None) -> list[Step]:
    if only:
        unknown = [n for n in only if n not in NAMES]
        if unknown:
            raise SystemExit(f"unknown steps: {unknown}; choose from {NAMES}")
        wanted = set(only)
        return [step for step in PIPELINE if step[0] in wanted]
    if from_step:
        if from_step not in NAMES:
            raise SystemExit(f"unknown --from {from_step!r}; choose from {NAMES}")
        idx = NAMES.index(from_step)
        sliced = PIPELINE[idx:]
        return [step for step in sliced if step[3]]
    return [step for step in PIPELINE if step[3]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ET-CoDI experiment pipeline")
    parser.add_argument(
        "--from", dest="from_step",
        choices=NAMES, default=None,
        help="start at this step and run through the end",
    )
    parser.add_argument("--only", nargs="+", choices=NAMES, default=None, help="run only the named steps")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    args = parser.parse_args()

    steps = _slice_pipeline(args.from_step, args.only)
    print(f"Pipeline ({len(steps)} steps): {' → '.join(n for n, _, _, _ in steps)}\n")

    for i, (name, desc, cmd, _default) in enumerate(steps, 1):
        print("=" * 72)
        print(f"[{i}/{len(steps)}] {name}: {desc}")
        print("  $", " ".join(cmd))
        print("=" * 72, flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            raise SystemExit(f"[FAIL] {name} exited {proc.returncode} after {elapsed:.1f}s")
        print(f"[OK] {name} finished in {elapsed:.1f}s\n", flush=True)

    print("All selected steps completed." if not args.dry_run else "Dry-run done.")


if __name__ == "__main__":
    main()
