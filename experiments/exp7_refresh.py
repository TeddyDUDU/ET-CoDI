"""exp7: refresh ablation and co-movement.

  python experiments/exp7_refresh.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.cluster_manager import CMTrainer
from core.params import CMParams, HyperParams
from core.paths import EXP3_CM, EXP7_DIR, ensure_project_cwd, norm_path
from core.seed import GLOBAL_SEED
from experiments.common_eval import ET_NAME
from experiments.ieee_plot import FIG_H_TS, FIG_W_TS, apply_ieee_style, save_fig
from experiments.runners import P_DRIFT_DEFAULT, SIGMA_DEFAULT, SLOT_STEPS_DEFAULT
from experiments.scripted_eval import (
    AlwaysReclusterTrigger,
    KeepTrigger,
    RefreshOnly,
    aggregate_stats,
    format_agg_table,
    run_et_seeds,
    run_scripted_episode,
    run_scripted_seeds,
)
from experiments.common_eval import load_ippo, make_stream_env

plt.switch_backend("Agg")

TEST_SEEDS = [GLOBAL_SEED + i for i in range(5)]
COMOVE_CANDIDATES = (
    (0.12, 0.02),
    (0.18, 0.01),
    (0.25, 0.01),
    (0.30, 0.005),
    (0.40, 0.005),
)


def _dissociation(trace: dict) -> tuple[float, float, bool]:
    d = np.asarray(trace["d"], dtype=float)
    s = np.asarray(trace["s"], dtype=float)
    head, tail = 20, 100
    d_rise = float(d[-tail:].mean() - d[:head].mean())
    s_rise = float(s[-tail:].mean() - s[:head].mean())
    ok = d_rise < 0.05 and s_rise > 0.05
    return d_rise, s_rise, ok


def _calibrate(T: int, ippo) -> tuple[float, float, str]:
    notes = []
    best_score = None
    best_pair = COMOVE_CANDIDATES[0]
    met = False
    for sigma_v, sigma_eps in COMOVE_CANDIDATES:
        ep, trace = run_scripted_episode(
            KeepTrigger(), ippo, TEST_SEEDS[0],
            enable_cache=False, T=T, p_drift=P_DRIFT_DEFAULT, sigma=SIGMA_DEFAULT,
            drift_mode="comove", sigma_v=sigma_v, sigma_eps=sigma_eps, record_trace=True,
        )
        assert trace is not None
        d_rise, s_rise, ok = _dissociation(trace)
        line = (
            f"sigma_v={sigma_v} sigma_eps={sigma_eps} "
            f"d_rise={d_rise:.4f} s_rise={s_rise:.4f} ok={ok} U={ep.cum_u:.2f}"
        )
        notes.append(line)
        if ok:
            best_pair = (sigma_v, sigma_eps)
            met = True
            break
        score = s_rise - max(0.0, d_rise - 0.05)
        if best_score is None or score > best_score:
            best_score = score
            best_pair = (sigma_v, sigma_eps)
    sigma_v, sigma_eps = best_pair
    status = "criterion met" if met else "no candidate fully separated D and S; using highest score"
    return sigma_v, sigma_eps, status + "\n" + "\n".join(notes)


_TRACE_COLOR = {
    "Static": "C0",
    "ET w/o refresh": "C1",
    "ET with refresh": "C2",
    "Always-recluster": "C4",
}


def _plot_traces(series: dict[str, tuple[np.ndarray, np.ndarray]], path: str) -> None:
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(FIG_W_TS, FIG_H_TS))
    for name, (_, s) in series.items():
        ax.plot(np.arange(len(s)), s, label=name, color=_TRACE_COLOR.get(name))
    ax.set_ylabel(r"$S_t^{pub}$")
    ax.set_xlabel("Time slot")
    ax.legend(loc="upper right", framealpha=0.92, edgecolor="0.8")
    ax.grid(True)
    fig.tight_layout()
    save_fig(fig, path)


def _et_mean_trace(cm_path: str, cm_params: CMParams, seeds: list[int], T: int, comove: dict, ippo) -> tuple[np.ndarray, np.ndarray]:
    cm = CMTrainer(HyperParams(), cm_params or CMParams())
    cm.load(cm_path)
    cm.set_frozen(True)
    ds, ss = [], []
    for seed in seeds:
        env = make_stream_env(
            enable_cache=True, slot_steps=T, cm_params=cm_params,
            p_drift=P_DRIFT_DEFAULT, sigma=SIGMA_DEFAULT, **comove,
        )
        obs = env.reset(seed)
        d_hist, s_hist = [], []
        while True:
            action, _ = cm.select_deploy_action(obs, mask=env.cm_action_mask())
            obs, _, _, _, info, done = env.step(action, ippo, False)
            d_hist.append(float(info.d_t))
            s_hist.append(float(info.s_pub_t))
            if done:
                break
        ds.append(d_hist)
        ss.append(s_hist)
    return np.mean(np.asarray(ds), axis=0), np.mean(np.asarray(ss), axis=0)


def _mean_trace(traces: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    d = np.stack([tr["d"] for tr in traces], axis=0)
    s = np.stack([tr["s"] for tr in traces], axis=0)
    return d.mean(axis=0), s.mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="exp7 refresh ablation and co-movement")
    parser.add_argument("--T", type=int, default=SLOT_STEPS_DEFAULT)
    parser.add_argument("--p-drift", type=float, default=P_DRIFT_DEFAULT)
    parser.add_argument("--sigma", type=float, default=SIGMA_DEFAULT)
    parser.add_argument("--cm-path", type=str, default=EXP3_CM)
    parser.add_argument("--out-dir", type=str, default=EXP7_DIR)
    args = parser.parse_args()

    ensure_project_cwd()
    out_dir = norm_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cm_path = norm_path(args.cm_path)
    ippo = load_ippo()
    ippo.set_frozen(True)
    common = dict(T=args.T, p_drift=args.p_drift, sigma=args.sigma, ippo=ippo)

    off = replace(CMParams(), enable_rule_refresh=False)
    on = CMParams()
    variants = [
        ("Static", None, False, False),
        ("ET w/o refresh/reuse", off, False, True),
        ("ET + refresh", on, False, True),
        ("ET + reuse", off, True, True),
        (f"Full {ET_NAME}", on, True, True),
    ]
    default_rows = []
    print("Default-drift ablation", flush=True)
    for name, cm_params, cache, is_et in variants:
        if not is_et:
            rows, _ = run_scripted_seeds(KeepTrigger, TEST_SEEDS, enable_cache=False, **common)
        else:
            rows = run_et_seeds(
                cm_path, TEST_SEEDS, enable_cache=cache, cm_params=cm_params,
                p_drift=args.p_drift, sigma=args.sigma, slot_steps=args.T,
            )
        agg = aggregate_stats(name, rows)
        default_rows.append(agg)
        print(f"  {name}  U={agg.cum_u:.2f}  refresh={agg.refresh_ops:.1f}", flush=True)

    print("Co-movement calibration", flush=True)
    sigma_v, sigma_eps, calib_note = _calibrate(args.T, ippo)
    comove = dict(
        drift_mode="comove", sigma_v=sigma_v, sigma_eps=sigma_eps, p_group=0.1,
    )
    print(f"Using sigma_v={sigma_v} sigma_eps={sigma_eps}", flush=True)

    comove_specs = [
        ("ET w/o refresh", off, True),
        ("ET with refresh", on, True),
    ]
    comove_rows = []
    plot_series = {}
    for name, cm_params, cache in comove_specs:
        rows = run_et_seeds(
            cm_path, TEST_SEEDS, enable_cache=cache, cm_params=cm_params,
            p_drift=args.p_drift, sigma=args.sigma, slot_steps=args.T, **comove,
        )
        comove_rows.append(aggregate_stats(name, rows))
        plot_series[name] = _et_mean_trace(cm_path, cm_params, TEST_SEEDS, args.T, comove, ippo)
        print(f"  {name}  U={comove_rows[-1].cum_u:.2f}", flush=True)

    scripted_specs = [
        ("Refresh-only", lambda: RefreshOnly(CMParams().tau_stale), True),
        ("Always-recluster", AlwaysReclusterTrigger, True),
    ]
    for name, factory, cache in scripted_specs:
        on_figure = name != "Refresh-only"
        rows, traces = run_scripted_seeds(
            factory, TEST_SEEDS, enable_cache=cache, record_trace=on_figure, **common, **comove,
        )
        comove_rows.append(aggregate_stats(name, rows))
        if on_figure:
            plot_series[name] = _mean_trace([tr for tr in traces if tr is not None])
        print(f"  {name}  U={comove_rows[-1].cum_u:.2f}", flush=True)

    static_rows, static_traces = run_scripted_seeds(
        KeepTrigger, TEST_SEEDS, enable_cache=True, record_trace=True, **common, **comove,
    )
    plot_series = {"Static": _mean_trace([tr for tr in static_traces if tr is not None]), **plot_series}
    _plot_traces(plot_series, os.path.join(out_dir, "comove_S.png"))

    with open(os.path.join(out_dir, "comove_params.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"sigma_v": sigma_v, "sigma_eps": sigma_eps, "p_group": 0.1},
            f, indent=2,
        )

    parts = [
        "exp7 refresh ablation and co-movement\n",
        f"seeds={TEST_SEEDS} default p_drift={args.p_drift} sigma={args.sigma} T={args.T}\n\n",
        "Default drift\n",
        format_agg_table(default_rows),
        "\nCo-movement\n",
        f"sigma_v={sigma_v} sigma_eps={sigma_eps} p_group=0.1\n",
        calib_note + "\n\n",
        format_agg_table(comove_rows + [aggregate_stats("Static", static_rows)]),
    ]
    summary = os.path.join(out_dir, "summary.txt")
    with open(summary, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
