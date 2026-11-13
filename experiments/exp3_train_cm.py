"""exp3: CM-PPO training -> trained_cm.pt (enable_cache=True).

  python experiments/exp3_train_cm.py
  python experiments/exp3_train_cm.py --episodes 100
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.cluster_manager import CM_ACTIONS, CM_KEEP, CM_RECLUSTER, CM_STREAM_MOVE, CMTrainer
from core.paths import EXP3_CM, EXP3_DIR, ensure_project_cwd, norm_path
from core.seed import GLOBAL_SEED, set_seed
from env.et_stream_env import EpisodeClusterStats
from experiments.common_eval import load_ippo, make_stream_env

SLOT_STEPS = 500
P_DRIFT, SIGMA = 0.1, 0.1


def _train_cm(
    env,
    cm: CMTrainer,
    ippo,
    episodes: int,
    seed: int,
) -> list[dict]:
    cm_params = env.cm_params
    ippo.set_frozen(True)
    cm.set_frozen(False)
    rng = np.random.RandomState(seed + 777)
    warmup = cm_params.actor_warmup_ep
    update_every = max(cm_params.update_every_ep, 1)
    env.p_drift, env.sigma = P_DRIFT, SIGMA
    log: list[dict] = []

    for ep in range(episodes):
        in_warmup = ep < warmup
        cm.set_actor_frozen(in_warmup)
        cm.set_entropy_coef(cm_params.entropy_coef_at(ep, episodes))
        ep_seed = seed + 1000 + ep
        set_seed(ep_seed)

        cm_obs = env.reset(ep_seed)
        ep_stats = EpisodeClusterStats()
        action_counts: Counter[int] = Counter()
        ep_return = 0.0
        n_steps = 0
        move_floor = cm_params.move_explore_floor_at(ep)
        recluster_floor = cm_params.recluster_explore_floor_at(ep)

        while True:
            mask = env.cm_action_mask()
            action, logp = cm.select_action(cm_obs, mask=mask)
            if move_floor > 0 and mask[CM_STREAM_MOVE] and action == CM_KEEP and rng.random() < move_floor:
                action = CM_STREAM_MOVE
                logp = cm.action_log_prob(cm_obs, CM_STREAM_MOVE, mask=mask)
            elif recluster_floor > 0 and mask[CM_RECLUSTER] and action == CM_KEEP and rng.random() < recluster_floor:
                action = CM_RECLUSTER
                logp = cm.action_log_prob(cm_obs, CM_RECLUSTER, mask=mask)

            next_obs, cm_r, _, _, info, done = env.step(action, ippo, False)
            ep_return += float(cm_r)
            cm.buffer.store(cm_obs, action, logp, cm_r, next_obs, float(done))
            action_counts[action] += 1
            ep_stats.add_step(info, streamhac_step=False)
            n_steps += 1
            cm_obs = next_obs
            if done:
                break

        post = ep - warmup + 1
        if len(cm.buffer.obs) > 0 and ep >= warmup and (post % update_every == 0 or ep == episodes - 1):
            cm.update()
            cm.buffer.clear()

        ep_stats.finalize(n_steps)
        log.append({
            "episode": ep + 1,
            "u_mean": float(ep_stats.u_mean),
            "cum_u": float(ep_stats.cum_u),
            "episode_return": float(ep_return),
        })
        n = sum(action_counts.values()) or 1
        acts = " ".join(f"{CM_ACTIONS[a]}={action_counts[a]/n:.0%}" for a in range(len(CM_ACTIONS)))
        tag = " [warmup]" if in_warmup else ""
        if (ep + 1) % 10 == 0 or ep + 1 == episodes:
            print(
                f"  ep {ep+1}/{episodes}  u={ep_stats.u_mean:.4f}  "
                f"move={ep_stats.move_ops:.0f}  recluster={ep_stats.recluster_ops:.0f}  {acts}{tag}"
            )

    return log


def train(out_dir: str, episodes: int, seed: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    cm_path = os.path.join(out_dir, os.path.basename(norm_path(EXP3_CM)))

    env = make_stream_env(
        enable_cache=True, p_drift=P_DRIFT, sigma=SIGMA, slot_steps=SLOT_STEPS,
    )
    ippo = load_ippo()
    cm = CMTrainer(env.hp, env.cm_params)

    print(f"=== CM train ({episodes} ep, cache on, p={P_DRIFT}, σ={SIGMA}, seed={seed}) ===")
    _train_cm(env, cm, ippo, episodes, seed)
    cm.save(cm_path)
    print(f"  saved -> {cm_path}")
    return cm_path


def main() -> None:
    parser = argparse.ArgumentParser(description="exp3 ET-CoDI CM training")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--out-dir", type=str, default=EXP3_DIR)
    args = parser.parse_args()

    ensure_project_cwd()
    train(norm_path(args.out_dir), args.episodes, args.seed)


if __name__ == "__main__":
    main()
