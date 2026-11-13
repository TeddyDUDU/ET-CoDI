"""exp1: Static-HIS per-agent IPPO pretraining -> trained_ippo.pt.

The HAC partition and the t=0 public prompts stay fixed. Train N independent Actor/Critic pairs for the inference-step decision.

  python experiments/exp1_train_ippo.py
  python experiments/exp1_train_ippo.py --plot-only
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.clustering import compute_phi, hac_clustering
from core.hiqm import make_hiqm
from core.ippo import IPPOTrainer, normalize_obs
from core.metrics import compute_latency_and_energy
from core.params import HardwareParams, HyperParams, UtilityNormRefs
from core.paths import EXP1_DIR, ensure_project_cwd
from core.seed import GLOBAL_SEED, set_seed
from core.utility import compute_agent_reward, compute_step_metrics, es_energy_share, hiqm_quality
from experiments.ieee_plot import FIG_H_TS, FIG_W_TS, apply_ieee_style, save_fig

plt.switch_backend("Agg")

CURVES_NPZ = "training_curves.npz"
REWARD_PNG = "reward_curve_StaticHIS.png"
UTILITY_PNG = "utility_curve_StaticHIS.png"


def _save_curves_npz(reward_curve: list, utility_curve: list, decision_points: int, path: str):
    np.savez(
        path,
        reward=np.asarray(reward_curve, dtype=float),
        utility=np.asarray(utility_curve, dtype=float),
        decision_points=np.int32(decision_points),
    )


def _save_reward_curve(reward_curve: list, path: str, decision_points: int):
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(FIG_W_TS, FIG_H_TS))
    steps = np.arange(1, len(reward_curve) + 1)
    ax.plot(steps, reward_curve, linewidth=0.9, alpha=0.45, color="#4393C3", label="Per-step")
    if len(reward_curve) >= 50:
        window = 50
        smoothed = np.convolve(reward_curve, np.ones(window) / window, mode="valid")
        ax.plot(
            steps[window - 1:], smoothed,
            linewidth=1.8, color="#B2182B", label=rf"MA-{window}",
        )
    n_steps = len(reward_curve)
    n_ep = n_steps // max(1, decision_points)
    ax.set_xlabel(r"Training step")
    ax.set_ylabel(r"Mean reward $\bar{r}$")
    ax.legend(loc="best", framealpha=0.92, edgecolor="0.8")
    ax.grid(True, axis="both")
    fig.tight_layout()
    save_fig(fig, path)


def _save_utility_curve(utility_curve: list, decision_points: int, path: str):
    """Aggregate Eq. (23) network utility by episode."""
    n_ep = len(utility_curve) // decision_points
    if n_ep == 0:
        return
    apply_ieee_style()
    episodes = np.arange(1, n_ep + 1)
    ep_util = [
        float(np.mean(utility_curve[i * decision_points:(i + 1) * decision_points]))
        for i in range(n_ep)
    ]
    fig, ax = plt.subplots(figsize=(FIG_W_TS, FIG_H_TS))
    ax.plot(episodes, ep_util, linewidth=1.8, color="#2166AC")
    ax.set_xlabel(r"Episode")
    ax.set_ylabel(r"Mean utility $\bar{U}$")
    y_min, y_max = min(ep_util), max(ep_util)
    pad = max((y_max - y_min) * 0.12, 0.02)
    ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))
    ax.grid(True, axis="both")
    fig.tight_layout()
    save_fig(fig, path)


def _save_trainer_weights(trainer: IPPOTrainer, path: str):
    state = {}
    for n in range(trainer.hp.N):
        state[f"actor_{n}"] = trainer.actors[n].state_dict()
        state[f"critic_{n}"] = trainer.critics[n].state_dict()
    base, _ = os.path.splitext(path)
    tmp_path = base + "._tmp.pt"
    torch.save(state, tmp_path)
    try:
        os.replace(tmp_path, path)
    except OSError:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def train_and_plot():
    ensure_project_cwd()
    set_seed(GLOBAL_SEED)
    hp = HyperParams()
    hw = HardwareParams()
    trainer = IPPOTrainer(hp)
    hiqm = make_hiqm(hp)

    prompt_embeds = [np.random.randn(768) for _ in range(hp.N)]
    cluster_assign = hac_clustering(prompt_embeds, hp.M)

    refs = UtilityNormRefs()
    reward_curve = []
    utility_curve = []
    os.makedirs(EXP1_DIR, exist_ok=True)
    curves_npz_path = f"{EXP1_DIR}/{CURVES_NPZ}"
    ippo_path = f"{EXP1_DIR}/trained_ippo.pt"

    for ep in range(hp.episodes):
        cluster_embeds = []
        for m in range(hp.M):
            members = [n for n in range(hp.N) if cluster_assign[n] == m]
            if members:
                mean_embed = np.mean([prompt_embeds[n] for n in members], axis=0)
            else:
                mean_embed = np.random.randn(768)
            cluster_embeds.append(mean_embed / (np.linalg.norm(mean_embed) + 1e-8))

        for _ in range(hp.decision_points):
            v_max_ES = random.randint(*hp.v_max_ES_range)
            v_UE = random.randint(*hp.v_UE_range)

            alpha1 = np.ones(hp.M)
            alpha2 = np.ones(hp.M)
            denom = np.sum(np.sqrt(alpha1 + alpha2))
            v_m_ES = v_max_ES * np.sqrt(alpha1 + alpha2) / (denom + 1e-8)

            actions, logps, obs_list = [], [], []
            for n in range(hp.N):
                m = cluster_assign[n]
                phi = compute_phi(prompt_embeds[n], cluster_embeds[m])
                o_n = normalize_obs(phi, v_max_ES, v_UE, hp)
                obs_list.append(o_n)

                a_n, logp_n = trainer.select_action(n, o_n)
                x_n = float(np.clip(round(a_n[0]), 0, hp.x_max))
                y_n = float(np.clip(round(a_n[1]), hp.y_min, hp.y_max))
                if x_n + y_n > hp.T_max:
                    y_n = max(hp.y_min, hp.T_max - x_n)
                actions.append((x_n, y_n))
                logps.append(logp_n)

            step_rewards = []
            for n in range(hp.N):
                x_n, y_n = actions[n]
                z_n = hp.T_max - x_n - y_n
                m = cluster_assign[n]

                x_m_max = max([actions[j][0] for j in range(hp.N) if cluster_assign[j] == m])
                y_sum_cluster = sum([actions[j][1] for j in range(hp.N) if cluster_assign[j] == m]) + 1e-8
                x_sum = sum([actions[j][0] for j in range(hp.N) if cluster_assign[j] == m]) + 1e-8

                T_n, E_n, E_m_C, E_m_PE = compute_latency_and_energy(
                    x_n, y_n, z_n,
                    x_m_max, y_sum_cluster,
                    v_UE, v_m_ES[m], hp, hw
                )
                phi_raw = compute_phi(prompt_embeds[n], cluster_embeds[m])
                Q_n = hiqm_quality(hiqm, x_n, phi_raw, hp)

                es_share = es_energy_share(x_n, y_n, x_sum, y_sum_cluster, E_m_C, E_m_PE)
                r_n = compute_agent_reward(Q_n, T_n, E_n, es_share, hp)
                step_rewards.append(r_n)
                r_n_norm = trainer.rnorms[n].normalize(r_n)

                next_v_max_ES = random.randint(*hp.v_max_ES_range)
                next_v_UE = random.randint(*hp.v_UE_range)
                next_obs = normalize_obs(phi_raw, next_v_max_ES, next_v_UE, hp)

                trainer.buffers[n].store(
                    obs_list[n], np.array([x_n, y_n]), logps[n],
                    r_n_norm, next_obs, 0.0
                )

            _, u_mean, _, _, _ = compute_step_metrics(
                prompt_embeds, cluster_assign, cluster_embeds,
                hp, hw, hiqm, actions, v_max_ES, v_UE, refs,
            )
            reward_curve.append(float(np.mean(step_rewards)))
            utility_curve.append(u_mean)

        trainer.update()

        if (ep + 1) % 10 == 0:
            recent_r = np.mean(reward_curve[-hp.decision_points:])
            recent_u = np.mean(utility_curve[-hp.decision_points:])
            print(
                f"Episode {ep + 1}/{hp.episodes}  "
                f"Recent avg reward: {recent_r:.4f}  utility: {recent_u:.4f}",
                flush=True,
            )
            # figure not written; not used in the paper
            _save_curves_npz(reward_curve, utility_curve, hp.decision_points, curves_npz_path)

    # figure not written; not used in the paper
    _save_curves_npz(reward_curve, utility_curve, hp.decision_points, curves_npz_path)
    _save_trainer_weights(trainer, ippo_path)
    n_ep = len(utility_curve) // hp.decision_points
    print(
        f"Done ({n_ep} episodes). Outputs in {EXP1_DIR}/:\n"
        f"  {CURVES_NPZ}\n"
        f"  {os.path.basename(ippo_path)}"
    )


def replot_from_disk():
    ensure_project_cwd()
    curves_npz_path = f"{EXP1_DIR}/{CURVES_NPZ}"
    if not os.path.isfile(curves_npz_path):
        raise FileNotFoundError(
            f"Training curves not found: {curves_npz_path}\n"
            "Run `python experiments/exp1_train_ippo.py` first."
        )
    data = np.load(curves_npz_path)
    _ = int(data["decision_points"])
    # figure not written; not used in the paper
    print(f"Loaded {curves_npz_path}; curve figures are not written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exp1: Static-HIS IPPO pretraining")
    parser.add_argument("--plot-only", action="store_true", help="load training_curves.npz; curve figures are not written")
    args = parser.parse_args()
    if args.plot_only:
        replot_from_disk()
    else:
        train_and_plot()
