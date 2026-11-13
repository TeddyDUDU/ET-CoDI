"""exp10: drift frequency and amplitude. Drift-factor sweep for the policies used in the main table. Policies are not retuned per cell.

Each figure has one axes. The summary writes one exp6-style table for each p or sigma.

  python experiments/exp10_drift_factors.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.params import CMParams
from core.paths import EXP3_CM, EXP10_DIR, ensure_project_cwd, norm_path
from core.seed import GLOBAL_SEED
from experiments.common_eval import ET_NAME, load_ippo
from experiments.exp6_threshold import (
    PERIODIC_MOVE_K,
    PERIODIC_RECLUSTER_K,
    TAU_D,
    TAU_S,
    _table,
)
from experiments.ieee_plot import FIG_W_TS, apply_ieee_style, save_fig
from experiments.runners import SLOT_STEPS_DEFAULT
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

plt.switch_backend("Agg")

TEST_SEEDS = [GLOBAL_SEED + i for i in range(5)]
P_GRID = (0.01, 0.05, 0.1, 0.15, 0.2)
SIGMA_GRID = (0.02, 0.05, 0.1, 0.15, 0.2)
PERIODIC_NAME = f"Periodic-{PERIODIC_MOVE_K}/{PERIODIC_RECLUSTER_K}"


def _curve(xs: tuple[float, ...], xlabel: str, path: str, series: dict[str, list[float]]) -> None:
    apply_ieee_style()
    # Full-width 7 inches, raised to 4.2 so the decline is not flattened by a short figure.
    fig, ax = plt.subplots(figsize=(FIG_W_TS, 4.2))
    for name, ys in series.items():
        ax.plot(xs, ys, marker="o", label="StaticHIS" if name == "Static" else name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$U_{cum}$")
    # Place the legend in the gap between the pink line and the baselines so it does not cover ET-CoDI.
    ax.legend(
        loc="center", bbox_to_anchor=(0.40, 0.62),
        ncol=2, framealpha=0.92, edgecolor="0.8",
    )
    ax.grid(True)
    fig.tight_layout()
    save_fig(fig, path)


def _eval_cell(cm_path: str, ippo, T: int, p_drift: float, sigma: float) -> list[AggRow]:
    common = dict(T=T, p_drift=p_drift, sigma=sigma, ippo=ippo)
    scripted = (
        ("Static", KeepTrigger, False),
        ("Always-move", AlwaysMoveTrigger, False),
        ("Always-recluster", AlwaysReclusterTrigger, False),
        (PERIODIC_NAME, lambda: PeriodicMove(PERIODIC_MOVE_K, PERIODIC_RECLUSTER_K), False),
        ("Threshold", lambda: ThresholdMaintain(TAU_S, TAU_D), False),
    )
    rows: list[AggRow] = []
    for name, factory, cache in scripted:
        stats, _ = run_scripted_seeds(factory, TEST_SEEDS, enable_cache=cache, **common)
        rows.append(aggregate_stats(name, stats))
    et = run_et_seeds(
        cm_path, TEST_SEEDS, enable_cache=True,
        p_drift=p_drift, sigma=sigma, slot_steps=T, cm_params=CMParams(),
    )
    rows.append(aggregate_stats(ET_NAME, et))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="exp10 drift frequency and magnitude")
    parser.add_argument("--T", type=int, default=SLOT_STEPS_DEFAULT)
    parser.add_argument("--cm-path", type=str, default=EXP3_CM)
    parser.add_argument("--out-dir", type=str, default=EXP10_DIR)
    args = parser.parse_args()

    ensure_project_cwd()
    out_dir = norm_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cm_path = norm_path(args.cm_path)
    ippo = load_ippo()
    ippo.set_frozen(True)
    names = (
        "Static",
        "Always-move",
        "Always-recluster",
        PERIODIC_NAME,
        "Threshold",
        ET_NAME,
    )

    lines = [
        "exp10 drift factors\n",
        f"test_seeds={TEST_SEEDS} T={args.T}\n\n",
    ]

    freq: dict[str, list[float]] = {name: [] for name in names}
    lines.append("sigma=0.1\n\n")
    for p_drift in P_GRID:
        print(f"p_drift={p_drift} sigma=0.1", flush=True)
        rows = _eval_cell(cm_path, ippo, args.T, p_drift, 0.1)
        for row in rows:
            freq[row.name].append(row.cum_u)
        lines.append(f"p_drift={p_drift:.2f}\n")
        lines.append(_table(rows))
        lines.append("\n")
    _curve(P_GRID, r"$p^{drift}$", os.path.join(out_dir, "U_vs_p_drift.png"), freq)

    mag: dict[str, list[float]] = {name: [] for name in names}
    lines.append("p_drift=0.1\n\n")
    for sigma in SIGMA_GRID:
        print(f"p_drift=0.1 sigma={sigma}", flush=True)
        rows = _eval_cell(cm_path, ippo, args.T, 0.1, sigma)
        for row in rows:
            mag[row.name].append(row.cum_u)
        lines.append(f"sigma={sigma:.2f}\n")
        lines.append(_table(rows))
        lines.append("\n")
    _curve(SIGMA_GRID, r"$\sigma$", os.path.join(out_dir, "U_vs_sigma.png"), mag)

    summary = os.path.join(out_dir, "summary.txt")
    with open(summary, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
