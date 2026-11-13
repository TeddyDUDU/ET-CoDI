"""
exp2: Static-HIS degradation under drift (IPPO eval)

The HAC partition and the t=0 public prompts stay fixed. Load trained_ippo.pt for the inference-step decision
and record how D_t^geom, D_t^pub, phi, Q, and U degrade as intents drift.

  python experiments/exp2_drift_degradation.py
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.clustering import compute_phi, hac_clustering
from core.degradation import build_cluster_embeds, system_degradation, system_prompt_degradation
from core.hiqm import HIQM, make_hiqm
from core.ippo import IPPOTrainer, normalize_obs
from core.params import HardwareParams, HyperParams, UtilityNormRefs
from core.utility import compute_step_metrics
from core.paths import EXP1_IPPO, EXP2_DIR, ensure_project_cwd, norm_path
from core.seed import GLOBAL_SEED, device, set_seed
from env.drift import apply_drift_step, make_drift_trajectory
from env.resources import make_resource_trajectory, resource_rng
from env.et_static_env import init_embeds
from experiments.ieee_plot import FIG_H_TS, FIG_W_TS, apply_ieee_style, save_fig

plt.switch_backend("Agg")

EMBED_DIM = 768
P_DRIFT_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]
DEFAULT_SEEDS = [GLOBAL_SEED + i for i in range(5)]

DRIFT_COLORS = {
    0.0: "#2166AC",
    0.01: "#92C5DE",
    0.05: "#4393C3",
    0.1: "#F4A582",
    0.2: "#B2182B",
}
DRIFT_LABELS = {
    0.0: r"$p^{\mathrm{drift}}=0$",
    0.01: r"$p^{\mathrm{drift}}=0.01$",
    0.05: r"$p^{\mathrm{drift}}=0.05$",
    0.1: r"$p^{\mathrm{drift}}=0.1$",
    0.2: r"$p^{\mathrm{drift}}=0.2$",
}


def select_ippo_actions(trainer, embeds, cluster_assign, cluster_embeds, hp, v_max_ES, v_UE):
    actions = []
    for n in range(hp.N):
        m = cluster_assign[n]
        phi = compute_phi(embeds[n], cluster_embeds[m])
        obs = normalize_obs(phi, v_max_ES, v_UE, hp)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            logits = trainer.actors[n](obs_t)
            act = logits.cpu().numpy()
        x_n = float(np.clip(round(act[0]), 0, hp.x_max))
        y_n = float(np.clip(round(act[1]), hp.y_min, hp.y_max))
        if x_n + y_n > hp.T_max:
            y_n = max(hp.y_min, hp.T_max - x_n)
        actions.append((x_n, y_n))
    return actions


def run_single(
    p_drift: float,
    sigma: float,
    T: int,
    hp: HyperParams,
    hw: HardwareParams,
    hiqm: HIQM,
    trainer: IPPOTrainer,
    initial_embeds: list,
    cluster_assign: list,
    drift_rng: np.random.RandomState,
    v_es_traj: np.ndarray,
    v_ue_traj: np.ndarray,
    refs: UtilityNormRefs,
):
    embeds = [e.copy() for e in initial_embeds]
    frozen_cluster_embeds = build_cluster_embeds(
        cluster_assign, initial_embeds, hp.M, EMBED_DIM, drift_rng
    )
    drift_masks, drift_noise = make_drift_trajectory(drift_rng, T, hp.N, EMBED_DIM, p_drift, sigma)

    d_hist, d_pub_hist, q_hist, u_hist, phi_hist = [], [], [], [], []

    for _t in range(T):
        embeds = apply_drift_step(embeds, drift_masks[_t], drift_noise[_t])
        v_max_ES = int(v_es_traj[_t])
        v_UE = int(v_ue_traj[_t])
        d_t = system_degradation(cluster_assign, embeds, hp.M)
        d_pub = system_prompt_degradation(
            cluster_assign, embeds, frozen_cluster_embeds, hp.M,
        )
        actions = select_ippo_actions(
            trainer, embeds, cluster_assign, frozen_cluster_embeds, hp, v_max_ES, v_UE
        )
        q_mean, u_mean, phi_mean, _, _ = compute_step_metrics(
            embeds, cluster_assign, frozen_cluster_embeds,
            hp, hw, hiqm, actions, v_max_ES, v_UE, refs,
        )
        d_hist.append(d_t)
        d_pub_hist.append(d_pub)
        q_hist.append(q_mean)
        u_hist.append(u_mean)
        phi_hist.append(phi_mean)

    u_arr = np.array(u_hist)
    return {
        "D_t": np.array(d_hist),
        "D_t_pub": np.array(d_pub_hist),
        "Q": np.array(q_hist),
        "U": u_arr,
        "phi": np.array(phi_hist),
        "cum_U": np.cumsum(u_arr),
    }


def aggregate_runs(runs: list):
    keys = runs[0].keys()
    mean, std = {}, {}
    for k in keys:
        stacked = np.stack([r[k] for r in runs], axis=0)
        mean[k] = stacked.mean(axis=0)
        std[k] = stacked.std(axis=0)
    return mean, std


def _tight_ylim(all_mean: dict, metric_key: str, min_pad_ratio: float = 0.08, min_pad_abs: float = 1e-6):
    y_min, y_max = np.inf, -np.inf
    for p_drift in P_DRIFT_VALUES:
        mean = all_mean[p_drift][metric_key]
        y_min = min(y_min, float(np.min(mean)))
        y_max = max(y_max, float(np.max(mean)))
    span = y_max - y_min
    pad = max(span * min_pad_ratio, min_pad_abs) if span >= 1e-12 else max(abs(y_max) * 1e-4, min_pad_abs)
    return y_min - pad, y_max + pad


def plot_metric(
    all_mean: dict,
    all_std: dict,
    metric_key: str,
    ylabel: str,
    out_path: str,
    T: int,
    tight_ylim: bool = True,
    show_std: bool = True,
):
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(FIG_W_TS, FIG_H_TS))
    t_axis = np.arange(T)
    for p_drift in P_DRIFT_VALUES:
        mean = all_mean[p_drift][metric_key]
        std = all_std[p_drift][metric_key]
        color = DRIFT_COLORS[p_drift]
        lw = 2.0 if metric_key == "U" else 1.6
        ax.plot(t_axis, mean, color=color, linewidth=lw, label=DRIFT_LABELS[p_drift])
        if show_std:
            ax.fill_between(t_axis, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)

    if tight_ylim:
        ylo, yhi = _tight_ylim(all_mean, metric_key)
        ax.set_ylim(ylo, yhi)

    ax.set_xlabel(r"Time slot $t$")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", framealpha=0.92, edgecolor="0.8")
    ax.grid(True, axis="both")
    ax.set_xlim(0, T - 1)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_delta_vs_baseline(
    all_mean: dict,
    metric_key: str,
    ylabel: str,
    out_path: str,
    T: int,
    baseline_p: float = 0.0,
):
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(FIG_W_TS, FIG_H_TS))
    t_axis = np.arange(T)
    baseline = all_mean[baseline_p][metric_key]
    for p_drift in P_DRIFT_VALUES:
        if p_drift == baseline_p:
            continue
        ax.plot(
            t_axis, all_mean[p_drift][metric_key] - baseline,
            color=DRIFT_COLORS[p_drift], linewidth=1.6, label=DRIFT_LABELS[p_drift],
        )
    ax.axhline(0.0, color="0.45", linestyle="--", linewidth=1.0, label=DRIFT_LABELS[baseline_p])
    ax.set_xlabel(r"Time slot $t$")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", framealpha=0.92, edgecolor="0.8")
    ax.grid(True, axis="both")
    ax.set_xlim(0, T - 1)
    fig.tight_layout()
    save_fig(fig, out_path)


def load_trainer(hp: HyperParams, ippo_path: str) -> IPPOTrainer:
    if not os.path.isfile(ippo_path):
        raise FileNotFoundError(
            f"IPPO weights not found: {ippo_path}\n"
            "Run `python experiments/exp1_train_ippo.py` first to generate trained_ippo.pt."
        )
    trainer = IPPOTrainer(hp)
    state = torch.load(ippo_path, map_location="cpu", weights_only=False)
    for n in range(hp.N):
        trainer.actors[n].load_state_dict(state[f"actor_{n}"])
        trainer.critics[n].load_state_dict(state[f"critic_{n}"])
    for actor in trainer.actors:
        actor.eval()
    return trainer


def main():
    parser = argparse.ArgumentParser(description="exp2: Static-HIS degradation under drift (IPPO eval)")
    parser.add_argument("--T", type=int, default=500, help="number of time slots")
    parser.add_argument("--sigma", type=float, default=0.1, help="drift magnitude")
    parser.add_argument("--ippo-path", type=str, default=EXP1_IPPO)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--out-dir", type=str, default=EXP2_DIR)
    args = parser.parse_args()

    ensure_project_cwd()
    args.ippo_path = norm_path(args.ippo_path)
    args.out_dir = norm_path(args.out_dir)

    hp = HyperParams()
    hw = HardwareParams()
    hiqm = make_hiqm(hp)
    refs = UtilityNormRefs()
    os.makedirs(args.out_dir, exist_ok=True)

    trainer = load_trainer(hp, args.ippo_path)
    print(f"Loaded IPPO weights from {args.ippo_path}")

    all_mean, all_std = {}, {}
    for p_drift in P_DRIFT_VALUES:
        print(f"Running p_drift={p_drift} ...")
        runs = []
        for seed in args.seeds:
            set_seed(seed)
            init_rng = np.random.RandomState(seed)
            drift_rng = np.random.RandomState(seed + 10000)
            res_rng = resource_rng(seed)
            initial_embeds = init_embeds(init_rng, hp.N, hp.M)
            cluster_assign = hac_clustering(initial_embeds, hp.M)
            v_es_traj, v_ue_traj = make_resource_trajectory(res_rng, args.T, hp)
            runs.append(run_single(
                p_drift=p_drift,
                sigma=args.sigma,
                T=args.T,
                hp=hp,
                hw=hw,
                hiqm=hiqm,
                trainer=trainer,
                initial_embeds=initial_embeds,
                cluster_assign=cluster_assign,
                drift_rng=drift_rng,
                v_es_traj=v_es_traj,
                v_ue_traj=v_ue_traj,
                refs=refs,
            ))
        mean, std = aggregate_runs(runs)
        all_mean[p_drift] = mean
        all_std[p_drift] = std
        print(
            f"  final D_t={mean['D_t'][-1]:.4f}  D_t_pub={mean['D_t_pub'][-1]:.4f}  "
            f"Q={mean['Q'][-1]:.4f}  U={mean['U'][-1]:.4f}  cum_U={mean['cum_U'][-1]:.2f}"
        )

    out = args.out_dir
    # figure not written; not used in the paper (D_t_vs_t, phi_vs_t, U_delta_vs_t)
    plot_metric(
        all_mean, all_std, "D_t_pub", r"$D_t^{\mathrm{pub}}$",
        f"{out}/D_t_pub_vs_t.png", args.T,
    )
    plot_metric(
        all_mean, all_std, "Q", r"$\bar{Q}$ (HIQM)",
        f"{out}/Q_vs_t.png", args.T,
    )
    plot_metric(
        all_mean, all_std, "U", r"$\bar{U}$",
        f"{out}/U_vs_t.png", args.T,
        show_std=False,
    )

    summary_path = os.path.join(out, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"mode=ippo, ippo_path={args.ippo_path}\n")
        f.write(f"sigma={args.sigma}, T={args.T}, seeds={args.seeds}\n\n")
        for p_drift in P_DRIFT_VALUES:
            m = all_mean[p_drift]
            s = all_std[p_drift]
            f.write(f"p_drift={p_drift}\n")
            f.write(f"  D_t:      mean_final={m['D_t'][-1]:.6f}  std_final={s['D_t'][-1]:.6f}\n")
            f.write(f"  D_t_pub:  mean_final={m['D_t_pub'][-1]:.6f}  std_final={s['D_t_pub'][-1]:.6f}\n")
            f.write(f"  phi:      mean_final={m['phi'][-1]:.6f}  std_final={s['phi'][-1]:.6f}\n")
            f.write(f"  Q:     mean_final={m['Q'][-1]:.6f}  std_final={s['Q'][-1]:.6f}\n")
            f.write(f"  U:     mean_final={m['U'][-1]:.6f}  std_final={s['U'][-1]:.6f}\n")
            f.write(f"  cum_U: mean_final={m['cum_U'][-1]:.6f}  std_final={s['cum_U'][-1]:.6f}\n")
            f.write(f"  (U_qos = U, cum_U_qos = cum_U)\n\n")

    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
