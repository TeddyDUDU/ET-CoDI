"""exp5: sensitivity across drift scenarios (Static-HIS / IIS / NIRVANA / ET-CoDI).

Compare ET-CoDI with published baselines (Static, IIS) and NIRVANA.
NIRVANA uses its own approximate cache.

  python experiments/exp5_sensitivity.py
  python experiments/exp5_sensitivity.py --plot-only
  python experiments/exp5_sensitivity.py --scenario default
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.cluster_manager import CMTrainer
from baselines.cluster_policies import StaticHISPolicy
from baselines.published_policies import IISPolicy
from core.hiqm import make_hiqm
from core.params import (
    CacheOverheadRefs,
    CacheThresholds,
    CMParams,
    ClusterOverheadRefs,
    HardwareParams,
    HyperParams,
    UtilityNormRefs,
)
from core.paths import EXP1_IPPO, EXP3_CM, EXP5_DIR, ensure_project_cwd, norm_path
from core.seed import GLOBAL_SEED, set_seed
from env.et_stream_env import EpisodeClusterStats, ETStreamEnv
from experiments.published_eval import (
    EvalResult,
    eval_scheme_seeds,
    load_ippo,
    run_baseline_episode,
    run_nirvana_episode,
)
from experiments.ieee_plot import FIG_H_BAR, FIG_H_PANEL, FIG_W_BAR, FIG_W_PANEL, apply_ieee_style, save_fig

plt.switch_backend("Agg")

SLOT_STEPS = 500
SEEDS = [GLOBAL_SEED + i for i in range(5)]
ET_NAME = "ET-CoDI"
POLICY_ORDER = (ET_NAME, "Static-HIS", "NIRVANA", "IIS")
SHORT_LABELS = {
    "Static-HIS": "StaticHIS",
    "IIS": "IIS",
    "NIRVANA": "NIRVANA",
    ET_NAME: "ET-CoDI",
}
COLORS = {
    "Static-HIS": "#4C72B0",
    "IIS": "#8172B3",
    "NIRVANA": "#DD8452",
    ET_NAME: "#54A24B",
}
RESULTS_JSON = "results.json"

SCENARIOS: tuple[tuple[str, float, float], ...] = (
    ("low", 0.01, 0.05),
    ("med", 0.05, 0.08),
    ("default", 0.1, 0.1),
    ("high", 0.2, 0.15),
)


@dataclass
class PolicyResult:
    u_mean: float
    u_std: float
    cum_u: float
    move_ops: float
    recluster_ops: float
    schedule_ops: float = 0.0
    ddrs_qs: float = 0.0


def _to_policy_result(r: EvalResult) -> PolicyResult:
    return PolicyResult(
        u_mean=r.u_mean,
        u_std=r.u_std,
        cum_u=r.cum_u,
        move_ops=r.move_ops,
        recluster_ops=r.recluster_ops,
        schedule_ops=r.schedule_ops,
        ddrs_qs=r.ddrs_qs,
    )


def _make_et_env(p_drift: float, sigma: float) -> ETStreamEnv:
    hp, hw = HyperParams(), HardwareParams()
    return ETStreamEnv(
        hp, hw, make_hiqm(hp), CMParams(), ClusterOverheadRefs(),
        CacheOverheadRefs(), CacheThresholds(), UtilityNormRefs(),
        slot_steps=SLOT_STEPS, enable_drift=True, enable_cache=True,
        p_drift=p_drift, sigma=sigma,
    )


def _eval_et(
    cm_path: str,
    p_drift: float,
    sigma: float,
    seeds: list[int],
    ippo,
) -> PolicyResult:
    if not os.path.isfile(cm_path):
        raise FileNotFoundError(f"CM weights not found: {cm_path}")

    cm = CMTrainer(HyperParams(), CMParams())
    cm.load(cm_path)
    cm.set_frozen(True)
    ippo.set_frozen(True)
    env = _make_et_env(p_drift, sigma)

    cum_us, moves, reclusters = [], [], []
    for seed in seeds:
        set_seed(seed)
        cm_obs = env.reset(seed)
        ep = EpisodeClusterStats()
        n_steps = 0
        while True:
            mask = env.cm_action_mask()
            action, _ = cm.select_deploy_action(cm_obs, mask=mask)
            cm_obs, _, _, _, info, done = env.step(action, ippo, False)
            ep.add_step(info, streamhac_step=False)
            n_steps += 1
            if done:
                break
        ep.finalize(n_steps)
        cum_us.append(ep.cum_u)
        moves.append(ep.move_ops)
        reclusters.append(ep.recluster_ops)

    cum = np.array(cum_us, dtype=float)
    return PolicyResult(
        u_mean=float(cum.mean() / SLOT_STEPS),
        u_std=float((cum / SLOT_STEPS).std(ddof=0)),
        cum_u=float(cum.mean()),
        move_ops=float(np.mean(moves)),
        recluster_ops=float(np.mean(reclusters)),
    )


def _eval_published(
    name: str,
    scheme: str,
    policy,
    p_drift: float,
    sigma: float,
    seeds: list[int],
    ippo,
) -> PolicyResult:
    hp, hw = HyperParams(), HardwareParams()
    hiqm = make_hiqm(hp)
    overhead = ClusterOverheadRefs()
    cache_refs = CacheOverheadRefs()
    utility_refs = UtilityNormRefs()
    # Published baselines in this comparison: Static and IIS.
    enable_cache = False

    def _run(seed: int) -> EvalResult:
        if policy is None:
            raise ValueError(f"Missing policy for scheme {scheme}")
        rng = np.random.RandomState(seed + 30000)
        return run_baseline_episode(
            policy, scheme, hp, hw, hiqm, rng, seed, SLOT_STEPS,
            p_drift, sigma, enable_cache, overhead, cache_refs, utility_refs, ippo,
        )

    r = eval_scheme_seeds(_run, seeds, SLOT_STEPS)
    return _to_policy_result(r)


def _eval_nirvana(
    p_drift: float,
    sigma: float,
    seeds: list[int],
    ippo,
) -> PolicyResult:
    hp, hw = HyperParams(), HardwareParams()
    hiqm = make_hiqm(hp)
    utility_refs = UtilityNormRefs()

    def _run(seed: int) -> EvalResult:
        return run_nirvana_episode(
            hp, hw, hiqm, seed, SLOT_STEPS, p_drift, sigma, utility_refs, ippo,
        )

    r = eval_scheme_seeds(_run, seeds, SLOT_STEPS)
    return _to_policy_result(r)


def _bar_ylim(means: list[float]) -> tuple[float, float]:
    lo, hi = min(means), max(means)
    span = max(hi - lo, 1e-6)
    return max(0.0, lo - span * 0.08), hi + span * 0.22


def _save_fig(fig, path: str) -> None:
    save_fig(fig, path, pad_inches=0.04)


def _sorted_names(results: dict[str, PolicyResult]) -> list[str]:
    """Sort by u_mean from high to low (for plotting)."""
    present = [n for n in POLICY_ORDER if n in results]
    return sorted(present, key=lambda n: results[n].u_mean, reverse=True)


def _plot_scenario(
    scenario: str,
    p_drift: float,
    sigma: float,
    results: dict[str, PolicyResult],
    out_path: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    apply_ieee_style()
    names = _sorted_names(results)
    means = [results[n].u_mean for n in names]
    labels = [SHORT_LABELS.get(n, n) for n in names]
    y_lo, y_hi = ylim if ylim else _bar_ylim(means)

    # Match one cell of the 2x2 panel figure (FIG_W_PANEL x FIG_H_PANEL).
    fig, ax = plt.subplots(figsize=(FIG_W_PANEL / 2, FIG_H_PANEL / 2))
    x = np.arange(len(names))
    bars = ax.bar(
        x, means, width=0.58,
        color=[COLORS.get(n, "#888") for n in names],
        edgecolor="white", linewidth=0.8,
    )
    bars[0].set_edgecolor("#2E7D32")
    bars[0].set_linewidth(2.0)

    ax.set_ylabel(r"$\bar{u}$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(y_lo, y_hi)
    ax.margins(x=0.08)
    for b, m in zip(bars, means):
        ax.text(
            b.get_x() + b.get_width() / 2, m + (y_hi - y_lo) * 0.02,
            f"{m:.3f}", ha="center", va="bottom", fontsize=7, clip_on=True,
        )
    ax.grid(axis="y")
    fig.tight_layout()
    _save_fig(fig, out_path)


def _plot_all_scenarios(
    all_results: dict[str, dict[str, PolicyResult]],
    scenario_meta: dict[str, tuple[float, float]],
    out_path: str,
    ylim: tuple[float, float],
) -> None:
    apply_ieee_style()
    keys = [s[0] for s in SCENARIOS if s[0] in all_results]
    cols, rows = 2, (len(keys) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(FIG_W_PANEL, FIG_H_PANEL * rows / 2))
    axes = np.atleast_1d(axes).flatten()
    y_lo, y_hi = ylim

    for i, (ax, key) in enumerate(zip(axes, keys)):
        res = all_results[key]
        names = _sorted_names(res)
        means = [res[n].u_mean for n in names]
        labels = [SHORT_LABELS.get(n, n) for n in names]
        x = np.arange(len(names))
        bars = ax.bar(x, means, color=[COLORS.get(n, "#888") for n in names], width=0.62)
        if bars:
            bars[0].set_edgecolor("#2E7D32")
            bars[0].set_linewidth(2.0)
        ax.text(
            0.02, 0.98, f"({chr(ord('a') + i)})",
            transform=ax.transAxes, va="top", ha="left", fontsize=9, fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(r"$\bar{u}$")
        ax.set_ylim(y_lo, y_hi)
        ax.margins(x=0.1)
        ax.grid(axis="y")
        for b, m in zip(ax.patches, means):
            ax.text(
                b.get_x() + b.get_width() / 2, m + (y_hi - y_lo) * 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=7, clip_on=True,
            )

    for ax in axes[len(keys):]:
        ax.axis("off")

    fig.tight_layout()
    _save_fig(fig, out_path)


GROUPED_ORDER = (ET_NAME, "Static-HIS", "NIRVANA", "IIS")
GROUPED_LABELS = {
    ET_NAME: "ET-CoDI", "Static-HIS": "StaticHIS", "NIRVANA": "NIRVANA", "IIS": "IIS",
}
SCENARIO_LABELS = {"low": "Low", "med": "Medium", "default": "Default", "high": "High"}


def _plot_grouped(
    all_results: dict[str, dict[str, PolicyResult]],
    scenario_meta: dict[str, tuple[float, float]],
    out_path: str,
) -> None:
    apply_ieee_style()
    keys = [s[0] for s in SCENARIOS if s[0] in all_results]
    names = [n for n in GROUPED_ORDER if all(n in all_results[k] for k in keys)]
    width = 0.8 / len(names)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(FIG_W_BAR, FIG_H_BAR))
    for i, name in enumerate(names):
        means = [all_results[k][name].u_mean for k in keys]
        bars = ax.bar(
            x + (i - (len(names) - 1) / 2) * width, means, width=width * 0.92,
            color=COLORS.get(name, "#888"), edgecolor="white", linewidth=0.6,
            label=GROUPED_LABELS.get(name, name),
        )
        for b, m in zip(bars, means):
            ax.text(
                b.get_x() + b.get_width() / 2, m + 0.003,
                f"{m:.3f}", ha="center", va="bottom", fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{SCENARIO_LABELS.get(k, k)}\n"
        rf"($p^{{\mathrm{{drift}}}}$={scenario_meta[k][0]:g}, $\sigma$={scenario_meta[k][1]:g})"
        for k in keys
    ])
    ax.set_ylabel(r"$\bar{u}$")
    ax.set_ylim(0.0, max(r.u_mean for k in keys for r in all_results[k].values()) * 1.22)
    ax.legend(loc="upper center", ncol=len(names), framealpha=0.92, edgecolor="0.8")
    ax.grid(axis="y")
    fig.tight_layout()
    _save_fig(fig, out_path)


def _global_ylim(all_results: dict[str, dict[str, PolicyResult]]) -> tuple[float, float]:
    means = [r.u_mean for res in all_results.values() for r in res.values()]
    return _bar_ylim(means)


def _save_results_json(
    out_dir: str,
    all_results: dict[str, dict[str, PolicyResult]],
    scenario_meta: dict[str, tuple[float, float]],
) -> None:
    payload = {
        "scenarios": {
            key: {
                "p_drift": scenario_meta[key][0],
                "sigma": scenario_meta[key][1],
                "policies": {n: asdict(r) for n, r in res.items()},
            }
            for key, res in all_results.items()
        },
    }
    with open(os.path.join(out_dir, RESULTS_JSON), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_summary(
    out_dir: str,
    all_results: dict[str, dict[str, PolicyResult]],
    scenario_meta: dict[str, tuple[float, float]],
    cm_path: str,
) -> None:
    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "exp5: Compare ET-CoDI with published baselines (Static, IIS) and NIRVANA. "
            "NIRVANA uses its own approximate cache.\n"
        )
        f.write(f"ippo={norm_path(EXP1_IPPO)}\n")
        f.write(f"cm={cm_path}\n")
        f.write(f"T={SLOT_STEPS} seeds={SEEDS}\n\n")
        for key, res in all_results.items():
            p_drift, sigma = scenario_meta[key]
            f.write(f"=== scenario {key} p_drift={p_drift} sigma={sigma} ===\n")
            for name in POLICY_ORDER:
                if name not in res:
                    continue
                r = res[name]
                f.write(
                    f"  {name:<22} u_mean={r.u_mean:.4f}±{r.u_std:.4f} "
                    f"cum_U={r.cum_u:.2f} move={r.move_ops:.0f} recluster={r.recluster_ops:.0f} "
                    f"sched={r.schedule_ops:.0f}"
                )
                f.write("\n")
            if ET_NAME in res and "Static-HIS" in res:
                d = res[ET_NAME].u_mean - res["Static-HIS"].u_mean
                f.write(f"  delta_u(ET-Static)={d:+.4f}\n")
            f.write("\n")


def _load_results_json(out_dir: str):
    path = os.path.join(out_dir, RESULTS_JSON)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    all_results = {}
    scenario_meta = {}
    for key, block in payload["scenarios"].items():
        scenario_meta[key] = (block["p_drift"], block["sigma"])
        all_results[key] = {
            n: PolicyResult(
                u_mean=d["u_mean"],
                u_std=d["u_std"],
                cum_u=d["cum_u"],
                move_ops=d["move_ops"],
                recluster_ops=d["recluster_ops"],
                schedule_ops=d.get("schedule_ops", 0.0),
                ddrs_qs=d.get("ddrs_qs", 0.0),
            )
            for n, d in block["policies"].items()
        }
    return all_results, scenario_meta


def replot_from_disk(out_dir: str) -> None:
    all_results, scenario_meta = _load_results_json(out_dir)
    # figure not written; not used in the paper (per-scenario bars and the all-scenario panel)
    if len(all_results) > 1:
        _plot_grouped(all_results, scenario_meta, os.path.join(out_dir, "utility_bars_grouped.png"))
    print(f"Replotted -> {out_dir}")


def run_experiments(
    out_dir: str,
    cm_path: str,
    seeds: list[int],
    scenario_filter: str | None,
) -> dict[str, dict[str, PolicyResult]]:
    hp = HyperParams()
    ippo = load_ippo(hp, norm_path(EXP1_IPPO))

    baselines = [
        ("Static-HIS", "static", StaticHISPolicy()),
        ("IIS", "iis", IISPolicy()),
    ]

    all_results: dict[str, dict[str, PolicyResult]] = {}
    scenario_meta: dict[str, tuple[float, float]] = {}

    for key, p_drift, sigma in SCENARIOS:
        if scenario_filter and key != scenario_filter:
            continue
        scenario_meta[key] = (p_drift, sigma)
        print(f"\n=== scenario {key}  p_drift={p_drift}  sigma={sigma} ===")
        results: dict[str, PolicyResult] = {}

        for bname, scheme, policy in baselines:
            print(f"  {bname} ...", flush=True)
            r = _eval_published(bname, scheme, policy, p_drift, sigma, seeds, ippo)
            results[bname] = r
            print(
                f"    u_mean={r.u_mean:.4f}  move={r.move_ops:.0f}  "
                f"recluster={r.recluster_ops:.0f}  sched={r.schedule_ops:.0f}"
            )

        print("  NIRVANA ...", flush=True)
        results["NIRVANA"] = _eval_nirvana(p_drift, sigma, seeds, ippo)
        r = results["NIRVANA"]
        print(f"    u_mean={r.u_mean:.4f}  move={r.move_ops:.0f}  recluster={r.recluster_ops:.0f}")

        print(f"  {ET_NAME} ...", flush=True)
        results[ET_NAME] = _eval_et(cm_path, p_drift, sigma, seeds, ippo)
        r = results[ET_NAME]
        print(f"    u_mean={r.u_mean:.4f}  move={r.move_ops:.0f}  recluster={r.recluster_ops:.0f}")

        all_results[key] = results

    # figure not written; not used in the paper (per-scenario bars and the all-scenario panel)
    if len(all_results) > 1:
        _plot_grouped(all_results, scenario_meta, os.path.join(out_dir, "utility_bars_grouped.png"))

    _save_results_json(out_dir, all_results, scenario_meta)
    _write_summary(out_dir, all_results, scenario_meta, cm_path)
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="exp5: ET-CoDI vs published baselines and NIRVANA")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--scenario", type=str, default=None, choices=[s[0] for s in SCENARIOS])
    parser.add_argument("--cm-path", type=str, default=EXP3_CM)
    parser.add_argument("--out-dir", type=str, default=EXP5_DIR)
    args = parser.parse_args()

    ensure_project_cwd()
    out_dir = norm_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cm_path = norm_path(args.cm_path)

    if args.plot_only:
        replot_from_disk(out_dir)
        return

    run_experiments(out_dir, cm_path, SEEDS, args.scenario)
    print(f"\nDone -> {out_dir}")


if __name__ == "__main__":
    main()
