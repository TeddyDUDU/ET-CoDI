"""exp12: scale and bandwidth on one edge GPU.

Common denoising is timed at 173.717 ms/step, measured from the SD3 benchmark (see experiments/sd3_utils.py and exp14) on an RTX 4090 D at 1024.
One card runs only one SD3 at a time, so M clusters queue serially. A hit executes the cached remaining steps.
A miss executes every common step. The terminal upload uses each user's effective rate and does not generate another image per user.

  python experiments/exp12_system.py
  python experiments/exp12_system.py --T 50 --seeds 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.cluster_manager import CMTrainer
from algorithms.shared_ippo import SharedIPPOTrainer
from core.params import CMParams, HyperParams
from core.paths import EXP1_IPPO, EXP3_CM, EXP12_DIR, ensure_project_cwd, norm_path
from core.seed import GLOBAL_SEED, set_seed
from experiments.common_eval import make_stream_env
from experiments.ieee_plot import FIG_H_HALF, FIG_W_HALF, apply_ieee_style, save_fig
from experiments.runners import P_DRIFT_DEFAULT, SIGMA_DEFAULT

plt.switch_backend("Agg")

# Step time measured from the SD3 benchmark (see experiments/sd3_utils.py and exp14).
STEP_MS = 173.717
STEP_S = STEP_MS / 1000.0
USER_COUNTS = (20, 50, 100, 200)
BANDWIDTH_MBPS = (2, 5, 10, 20, 50, 100)


class _SharedView:
    """The shared IPPO only has select_action(obs). The environment calls select_action(n, obs) per user."""

    def __init__(self, shared: SharedIPPOTrainer):
        self._shared = shared

    def set_frozen(self, frozen: bool) -> None:
        self._shared.set_frozen(frozen)

    def select_action(self, n, obs):
        return self._shared.select_action(obs)


def load_step_policy(n_users: int, ippo_path: str):
    """All N users share one actor from trained_ippo.pt.

    That file holds 50 user networks and cannot be resized to 20, 100, or 200.
    The scale comparison must use one step policy, so one of the 50 networks is selected and then shared by every user.
    """
    hp = HyperParams()
    hp.N = n_users
    probe = SharedIPPOTrainer(HyperParams())
    agent_id = probe.init_from_exp0(ippo_path)
    if agent_id < 0:
        raise SystemExit(f"Missing IPPO weights: {ippo_path}")
    shared = SharedIPPOTrainer(hp)
    shared.init_from_exp0(ippo_path, agent_id=agent_id)
    shared.set_frozen(True)
    return _SharedView(shared), f"shared-agent-{agent_id}"


def payload_bits(hp: HyperParams) -> float:
    return float(hp.n_h * hp.n_w * hp.n_c * hp.rho)


def run_episode(n_users: int, seed: int, enable_cache: bool, slot_steps: int, cm: CMTrainer, policy, p_drift: float, sigma: float) -> dict:
    hp = HyperParams()
    hp.N = n_users
    set_seed(seed)
    env = make_stream_env(
        enable_cache=enable_cache,
        p_drift=p_drift,
        sigma=sigma,
        slot_steps=slot_steps,
        hp=hp,
        cm_params=cm.cm_params,
    )
    obs = env.reset(seed)
    gpu_steps = []
    reused = []
    hits = misses = 0
    while True:
        action, _ = cm.select_deploy_action(obs, mask=env.cm_action_mask())
        obs, _, _, _, info, done = env.step(action, policy, False)
        gpu_steps.append(float(sum(info.x_common_exec)))
        reused.append(float(info.reused_steps))
        hits += info.hits
        misses += info.misses
        if done:
            break
    lookups = hits + misses
    return {
        "gpu_steps_mean": float(np.mean(gpu_steps)),
        "reused_steps_mean": float(np.mean(reused)),
        "hit_ratio": hits / lookups if lookups else 0.0,
    }


def aggregate(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    return {
        "gpu_steps_mean": mean("gpu_steps_mean"),
        "reused_steps_mean": mean("reused_steps_mean"),
        "hit_ratio": mean("hit_ratio"),
        "gpu_makespan_s": mean("gpu_steps_mean") * STEP_S,
    }


def tx_seconds(bits: float, mbps: float) -> float:
    return bits / (mbps * 1e6)


MODES = (
    ("scratch", "w/o reuse", "#D55E00", "s"),
    ("reuse", "with reuse", "#0072B2", "o"),
)


def plot_scale(rows: list[dict], path: str) -> None:
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(FIG_W_HALF, FIG_H_HALF))
    for mode, label, color, marker in MODES:
        group = [row for row in rows if row["mode"] == mode]
        ax.plot(
            [row["n_users"] for row in group],
            [row["gpu_makespan_s"] for row in group],
            marker=marker,
            color=color,
            label=label,
        )
    ax.set_xticks(USER_COUNTS)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Number of users $N$")
    ax.set_ylabel("Slot makespan (s)")
    ax.legend(loc="center right", framealpha=0.92, edgecolor="0.8")
    ax.grid(True)
    fig.tight_layout()
    save_fig(fig, path)


def plot_bandwidth(rows: list[dict], path: str) -> None:
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(FIG_W_HALF, FIG_H_HALF))
    for mode, label, color, marker in MODES:
        group = [row for row in rows if row["mode"] == mode]
        ax.plot(
            [row["bandwidth_mbps"] for row in group],
            [row["user_latency_s"] for row in group],
            marker=marker,
            color=color,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xticks(BANDWIDTH_MBPS)
    ax.set_xticklabels([str(v) for v in BANDWIDTH_MBPS])
    ax.minorticks_off()
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Downlink rate (Mbps)")
    ax.set_ylabel("Per-user latency (s)")
    ax.legend(loc="upper right", framealpha=0.92, edgecolor="0.8")
    ax.grid(True)
    fig.tight_layout()
    save_fig(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="exp12 edge scalability and bandwidth")
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--p-drift", type=float, default=P_DRIFT_DEFAULT)
    parser.add_argument("--sigma", type=float, default=SIGMA_DEFAULT)
    parser.add_argument("--cm-path", type=str, default=EXP3_CM)
    parser.add_argument("--out-dir", type=str, default=EXP12_DIR)
    parser.add_argument("--plot-only", action="store_true", help="redraw figures from summary.json")
    args = parser.parse_args()

    ensure_project_cwd()
    out_dir = norm_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if args.plot_only:
        with open(os.path.join(out_dir, "summary.json"), encoding="utf-8") as f:
            saved = json.load(f)
        plot_scale(saved["scale"], os.path.join(out_dir, "makespan_vs_users.png"))
        plot_bandwidth(saved["bandwidth"], os.path.join(out_dir, "latency_vs_bandwidth.png"))
        print(f"Figures written to {out_dir}", flush=True)
        return
    cm_path = norm_path(args.cm_path)
    ippo_path = norm_path(EXP1_IPPO)
    if not os.path.isfile(cm_path):
        raise SystemExit(f"Missing CM weights: {cm_path}")
    if not os.path.isfile(ippo_path):
        raise SystemExit(f"Missing IPPO weights: {ippo_path}")
    seeds = [GLOBAL_SEED + i for i in range(args.seeds)]
    hp = HyperParams()
    bits = payload_bits(hp)

    scale_rows = []
    lines = [
        f"{'N':>6} {'mode':<8} {'hit':>8} {'reused':>8} {'gpu_s':>8}\n",
    ]
    for n_users in USER_COUNTS:
        hp_n = HyperParams()
        hp_n.N = n_users
        cm = CMTrainer(hp_n, CMParams())
        cm.load(cm_path)
        cm.set_frozen(True)
        policy, policy_name = load_step_policy(n_users, ippo_path)
        lines.append(f"N={n_users} policy={policy_name}\n")
        print(f"N={n_users} policy={policy_name}", flush=True)
        for enable_cache, mode in ((False, "scratch"), (True, "reuse")):
            episodes = [
                run_episode(
                    n_users, seed, enable_cache, args.T, cm, policy, args.p_drift, args.sigma,
                )
                for seed in seeds
            ]
            stats = aggregate(episodes)
            row = {"n_users": n_users, "mode": mode, **stats}
            scale_rows.append(row)
            lines.append(
                f"{n_users:6d} {mode:<8} {stats['hit_ratio']:8.3f} "
                f"{stats['reused_steps_mean']:8.2f} {stats['gpu_makespan_s']:8.3f}\n"
            )
            print(
                f"N={n_users} {mode} hit={stats['hit_ratio']:.3f} "
                f"gpu={stats['gpu_makespan_s']:.3f}s",
                flush=True,
            )
    plot_scale(scale_rows, os.path.join(out_dir, "makespan_vs_users.png"))

    n50 = {row["mode"]: row for row in scale_rows if row["n_users"] == 50}
    bandwidth_rows = []
    lines.append(f"\nBandwidth at N=50. Payload={bits:.0f} bits.\n")
    lines.append(f"{'Mbps':>8} {'mode':<8} {'tx_s':>8} {'user_s':>8}\n")
    for mbps in BANDWIDTH_MBPS:
        tx_s = tx_seconds(bits, mbps)
        for mode in ("scratch", "reuse"):
            user_s = n50[mode]["gpu_makespan_s"] + tx_s
            bandwidth_rows.append({
                "bandwidth_mbps": mbps,
                "mode": mode,
                "tx_s": tx_s,
                "user_latency_s": user_s,
                "gpu_makespan_s": n50[mode]["gpu_makespan_s"],
            })
            lines.append(f"{mbps:8.0f} {mode:<8} {tx_s:8.3f} {user_s:8.3f}\n")
    plot_bandwidth(bandwidth_rows, os.path.join(out_dir, "latency_vs_bandwidth.png"))

    payload = {
        "step_ms": STEP_MS,
        "payload_bits": bits,
        "scale": scale_rows,
        "bandwidth": bandwidth_rows,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Results saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
