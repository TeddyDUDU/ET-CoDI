"""exp6 main table.

The table compares Static, Always-move, Always-recluster, Periodic, Threshold, and ET-CoDI.
Periodic: migrate every 3 steps, and recluster instead every 50 steps. At T=500 that is about 163 migrations, close to ET's 172.
Threshold: recluster when D > 0.85, otherwise migrate when S^pub > 0.15. 0.85 tightens the old 0.75 gate that was being crossed.
ET uses the exp3 weights. This script writes only the main-table summary. It does not write a figure.

  python experiments/exp6_threshold.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.params import CMParams
from core.paths import EXP3_CM, EXP6_DIR, ensure_project_cwd, norm_path
from core.seed import GLOBAL_SEED
from experiments.common_eval import ET_NAME, load_ippo
from experiments.runners import P_DRIFT_DEFAULT, SIGMA_DEFAULT, SLOT_STEPS_DEFAULT
from experiments.scripted_eval import (
    AggRow,
    AlwaysMoveTrigger,
    AlwaysReclusterTrigger,
    KeepTrigger,
    PeriodicMove,
    ThresholdMaintain,
    aggregate_stats,
    run_et_seeds,
    run_scripted_seeds,
)

TEST_SEEDS = [GLOBAL_SEED + i for i in range(5)]
PERIODIC_MOVE_K = 3
PERIODIC_RECLUSTER_K = 50
TAU_S = CMParams().tau_stale
TAU_D = 0.85


def _table(rows: list[AggRow]) -> str:
    header = (
        f"{'Policy':<22} {'U_cum':>16} {'QoS':>16} {'Move':>12} "
        f"{'Recluster':>12} {'Maintenance':>14} {'S':>8}\n"
    )
    lines = [header, "-" * (len(header.rstrip())) + "\n"]

    def pm(mean: float, std: float, prec: int = 2) -> str:
        return f"{mean:.{prec}f}±{std:.{prec}f}"

    for row in rows:
        lines.append(
            f"{row.name:<22} {pm(row.cum_u, row.cum_u_std):>16} "
            f"{pm(row.cum_u_qos, row.cum_u_qos_std):>16} "
            f"{pm(row.move_ops, row.move_ops_std, 1):>12} "
            f"{pm(row.recluster_ops, row.recluster_ops_std, 1):>12} "
            f"{pm(row.maint_cost, row.maint_cost_std, 3):>14} "
            f"{row.s_pub_mean:8.3f}\n"
        )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="exp6 main table")
    parser.add_argument("--T", type=int, default=SLOT_STEPS_DEFAULT)
    parser.add_argument("--p-drift", type=float, default=P_DRIFT_DEFAULT)
    parser.add_argument("--sigma", type=float, default=SIGMA_DEFAULT)
    parser.add_argument("--cm-path", type=str, default=EXP3_CM)
    parser.add_argument("--out-dir", type=str, default=EXP6_DIR)
    args = parser.parse_args()

    ensure_project_cwd()
    out_dir = norm_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cm_path = norm_path(args.cm_path)
    common = dict(T=args.T, p_drift=args.p_drift, sigma=args.sigma)
    ippo = load_ippo()
    ippo.set_frozen(True)

    scripted = (
        ("Static", KeepTrigger, False),
        ("Always-move", AlwaysMoveTrigger, False),
        ("Always-recluster", AlwaysReclusterTrigger, False),
        (
            f"Periodic-{PERIODIC_MOVE_K}/{PERIODIC_RECLUSTER_K}",
            lambda: PeriodicMove(PERIODIC_MOVE_K, PERIODIC_RECLUSTER_K),
            False,
        ),
        ("Threshold", lambda: ThresholdMaintain(TAU_S, TAU_D), False),
    )
    table: list[AggRow] = []
    for name, factory, cache in scripted:
        print(f"Running {name}", flush=True)
        rows, _ = run_scripted_seeds(
            factory, TEST_SEEDS, enable_cache=cache, ippo=ippo, **common,
        )
        agg = aggregate_stats(name, rows)
        table.append(agg)
        print(
            f"  U={agg.cum_u:.2f}  move={agg.move_ops:.1f}  "
            f"recluster={agg.recluster_ops:.1f}  S={agg.s_pub_mean:.3f}",
            flush=True,
        )

    print(f"Running {ET_NAME}", flush=True)
    et = run_et_seeds(
        cm_path, TEST_SEEDS, enable_cache=True,
        p_drift=args.p_drift, sigma=args.sigma, slot_steps=args.T,
        cm_params=CMParams(),
    )
    table.append(aggregate_stats(ET_NAME, et))

    text = (
        "exp6 main table\n"
        f"test_seeds={TEST_SEEDS}  p_drift={args.p_drift} sigma={args.sigma} T={args.T}\n"
        "Compare Static, Always-move, Always-recluster, "
        f"Periodic-{PERIODIC_MOVE_K}/{PERIODIC_RECLUSTER_K}, Threshold, and ET-CoDI.\n"
        f"Periodic moves every {PERIODIC_MOVE_K} steps and reclusters every "
        f"{PERIODIC_RECLUSTER_K} steps.\n"
        f"Threshold reclusters when D > {TAU_D:.2f}, else moves when S^pub > {TAU_S:.2f} "
        f"(S^pub gate is CMParams.tau_stale). "
        "A blocked recluster downgrades to move only if the move gate is also open.\n"
        "This run writes only the main-table summary.\n\n"
        + _table(table)
    )
    summary = os.path.join(out_dir, "summary.txt")
    with open(summary, "w", encoding="utf-8") as f:
        f.write(text)
    print(text, flush=True)
    print(f"Results saved to {summary}", flush=True)


if __name__ == "__main__":
    main()
