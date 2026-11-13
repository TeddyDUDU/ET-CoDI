"""Shared experiment runner: fixed seed split and baseline/ET simulation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from algorithms.cluster_manager import CMTrainer
from core.clustering import compute_phi, hac_clustering
from core.degradation import system_degradation
from core.hiqm import HIQM, make_hiqm
from core.ippo import IPPOTrainer
from core.params import CMParams, ClusterOverheadRefs, HardwareParams, HyperParams
from core.seed import set_seed
from core.utility import policy_cluster_overhead, policy_step_ops
from env.drift import make_drift_trajectory
from env.resources import make_resource_trajectory, resource_rng
from env.et_static_env import (
    EMBED_DIM,
    ClusterStats,
    EpisodeResult,
    ETStaticEnv,
    init_embeds,
)
from experiments.common_eval import load_ippo, make_stream_env, run_et_eval

P_DRIFT_DEFAULT = 0.1
SIGMA_DEFAULT = 0.1
SLOT_STEPS_DEFAULT = 500


@dataclass
class GroupStats:
    u_mean: float
    cum_U: float
    cum_U_qos: float
    Q_final: float
    move_ops: float
    recluster_ops: float
    u_mean_std: float = 0.0
    cum_U_std: float = 0.0
    cum_U_qos_std: float = 0.0
    move_ops_std: float = 0.0
    recluster_ops_std: float = 0.0
    refresh_ops: float = 0.0
    hit_ratio: float = 0.0
    reused_steps: float = 0.0
    false_safe_rate: float = 0.0


def _mean_std(values: List[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate_group_rows(rows: List[GroupStats]) -> GroupStats:
    u_m, u_s = _mean_std([r.u_mean for r in rows])
    c_m, c_s = _mean_std([r.cum_U for r in rows])
    q_m, q_s = _mean_std([r.cum_U_qos for r in rows])
    mv_m, mv_s = _mean_std([r.move_ops for r in rows])
    rc_m, rc_s = _mean_std([r.recluster_ops for r in rows])
    qf_m, _ = _mean_std([r.Q_final for r in rows])
    rf_m, _ = _mean_std([r.refresh_ops for r in rows])
    hr_m, _ = _mean_std([r.hit_ratio for r in rows])
    rs_m, _ = _mean_std([r.reused_steps for r in rows])
    fs_m, _ = _mean_std([r.false_safe_rate for r in rows])
    return GroupStats(
        u_mean=u_m, cum_U=c_m, cum_U_qos=q_m, Q_final=qf_m,
        move_ops=mv_m, recluster_ops=rc_m,
        u_mean_std=u_s, cum_U_std=c_s, cum_U_qos_std=q_s,
        move_ops_std=mv_s, recluster_ops_std=rc_s,
        refresh_ops=rf_m, hit_ratio=hr_m, reused_steps=rs_m, false_safe_rate=fs_m,
    )

def episode_seed_bundle(seed: int, T: int, p_drift: float, sigma: float, hp: HyperParams):
    """Fixed RNG split: init=seed, drift=seed+10000, resources=seed+20000 (length-T trajectories)."""
    set_seed(seed)
    init_rng = np.random.RandomState(seed)
    drift_rng = np.random.RandomState(seed + 10000)
    res_rng = resource_rng(seed)
    initial_embeds = init_embeds(init_rng, hp.N, hp.M)
    initial_assign = hac_clustering(initial_embeds, hp.M)
    drift_masks, drift_noise = make_drift_trajectory(
        drift_rng, T, hp.N, EMBED_DIM, p_drift, sigma,
    )
    v_max_ES, v_UE = make_resource_trajectory(res_rng, T, hp)
    return initial_embeds, initial_assign, drift_masks, drift_noise, v_max_ES, v_UE, drift_rng


def run_baseline_episode(
    policy,
    hp: HyperParams,
    hw: HardwareParams,
    hiqm: HIQM,
    initial_embeds: list,
    initial_assign: list,
    drift_masks: np.ndarray,
    drift_noise: np.ndarray,
    v_max_ES,
    v_UE,
    rng: np.random.RandomState,
    T: int,
    ippo_trainer: IPPOTrainer,
    overhead_refs: ClusterOverheadRefs,
) -> EpisodeResult:
    """Evaluate a baseline policy."""
    env = ETStaticEnv(
        hp, hw, hiqm, initial_embeds, initial_assign,
        drift_masks, drift_noise, v_max_ES, v_UE, rng,
        prompt_refresh=policy.prompt_refresh,
        ippo_trainer=ippo_trainer,
    )
    policy.reset(env.cluster_assign)

    q_hist, u_hist, u_qos_hist, phi_hist, d_pub_hist = [], [], [], [], []
    stats = ClusterStats()
    policy_name = policy.name

    for t in range(T):
        env.apply_drift(t)
        d_t = system_degradation(env.cluster_assign, env.embeds, hp.M)
        new_assign, info = policy.update(t, env.embeds, env.cluster_assign, d_t, hp.M)
        env.cluster_assign = new_assign
        if info.refresh_prompt:
            affected = getattr(info, "affected_clusters", None)
            if info.recluster or not affected:
                env.refresh_public_prompts()
            else:
                env.refresh_selected_prompts(affected)

        move_scan = bool(getattr(info, "move_scan", False))
        m_ops, r_ops = policy_step_ops(policy_name, info.recluster, move_scan)
        stats.move_ops += m_ops
        stats.recluster_ops += r_ops

        q, u_qos, phi = env.step_metrics()
        d_pub = env.service_prompt_degradation()
        u = u_qos - policy_cluster_overhead(
            info.recluster, info.recluster_time, info.moved, overhead_refs, policy_name,
            move_scan=move_scan,
        )

        q_hist.append(q)
        u_qos_hist.append(u_qos)
        u_hist.append(u)
        phi_hist.append(phi)
        d_pub_hist.append(d_pub)

    u_arr = np.array(u_hist)
    u_qos_arr = np.array(u_qos_hist)
    return EpisodeResult(
        Q=np.array(q_hist),
        U=u_arr,
        U_qos=u_qos_arr,
        phi=np.array(phi_hist),
        D_t_pub=np.array(d_pub_hist),
        cum_U=np.cumsum(u_arr),
        cum_U_qos=np.cumsum(u_qos_arr),
        final_D_t=system_degradation(env.cluster_assign, env.embeds, hp.M),
        stats=stats,
    )


def run_baseline_group(
    policy,
    *,
    seeds: List[int],
    T: int,
    p_drift: float,
    sigma: float,
    overhead_refs: ClusterOverheadRefs,
    ippo_trainer: IPPOTrainer,
) -> GroupStats:
    hp, hw = HyperParams(), HardwareParams()
    hiqm = make_hiqm(hp)
    rows: List[GroupStats] = []

    for seed in seeds:
        initial_embeds, initial_assign, drift_masks, drift_noise, v_max_ES, v_UE, drift_rng = (
            episode_seed_bundle(seed, T, p_drift, sigma, hp)
        )
        ep = run_baseline_episode(
            policy, hp, hw, hiqm, initial_embeds, initial_assign,
            drift_masks, drift_noise, v_max_ES, v_UE, drift_rng, T,
            ippo_trainer, overhead_refs,
        )
        rows.append(GroupStats(
            u_mean=float(ep.cum_U[-1]) / T,
            cum_U=float(ep.cum_U[-1]),
            cum_U_qos=float(ep.cum_U_qos[-1]),
            Q_final=float(ep.Q[-1]),
            move_ops=float(ep.stats.move_ops),
            recluster_ops=float(ep.stats.recluster_ops),
        ))

    return aggregate_group_rows(rows)


def run_et_group(
    cm_path: str,
    *,
    cache_enabled: bool,
    seeds: List[int],
    p_drift: float,
    sigma: float,
    T: int,
    cm_params: CMParams | None = None,
    drift_mode: str = "independent",
    p_group: float = 0.1,
    sigma_v: float = 0.12,
    sigma_eps: float = 0.02,
    es_scale: float = 1.0,
    b_max: int | None = None,
    cache_thresholds=None,
) -> GroupStats:
    rows: List[GroupStats] = []
    for seed in seeds:
        et = run_et_eval(
            cm_path,
            enable_cache=cache_enabled,
            seeds=[seed],
            p_drift=p_drift,
            sigma=sigma,
            slot_steps=T,
            cm_params=cm_params,
            drift_mode=drift_mode,
            p_group=p_group,
            sigma_v=sigma_v,
            sigma_eps=sigma_eps,
            es_scale=es_scale,
            b_max=b_max,
            cache_thresholds=cache_thresholds,
        )
        rows.append(GroupStats(
            u_mean=et.u_mean,
            cum_U=et.cum_u,
            cum_U_qos=et.cum_u_qos,
            Q_final=0.0,
            move_ops=float(et.move_ops),
            recluster_ops=float(et.recluster_ops),
            refresh_ops=float(et.refresh_ops),
            hit_ratio=float(et.hit_ratio),
            reused_steps=float(et.reused_steps),
            false_safe_rate=float(et.false_safe_rate),
        ))
    return aggregate_group_rows(rows)


def run_et_timeseries(
    cm_path: str,
    seed: int,
    *,
    enable_cache: bool,
    p_drift: float = P_DRIFT_DEFAULT,
    sigma: float = SIGMA_DEFAULT,
    T: int = SLOT_STEPS_DEFAULT,
    cm_params: CMParams | None = None,
) -> EpisodeResult:
    if not os.path.isfile(cm_path):
        raise FileNotFoundError(f"CM weights not found: {cm_path}")

    set_seed(seed)
    ippo = load_ippo()
    ippo.set_frozen(True)
    cm = CMTrainer(HyperParams(), cm_params or CMParams())
    cm.load(cm_path)
    cm.set_frozen(True)

    env = make_stream_env(
        enable_cache=enable_cache, p_drift=p_drift, sigma=sigma, slot_steps=T,
        cm_params=cm_params,
    )
    cm_obs = env.reset(seed)

    q_hist, u_hist, u_qos_hist, phi_hist, d_pub_hist = [], [], [], [], []
    s_pub_hist, hit_hist, refresh_hist = [], [], []
    stats = ClusterStats()

    while True:
        mask = env.cm_action_mask()
        action, _ = cm.select_deploy_action(cm_obs, mask=mask)
        cm_obs, _, _, _, info, done = env.step(action, ippo, False)
        if info.move_triggered:
            stats.move_ops += 1
        if info.recluster:
            stats.recluster_ops += 1

        cluster_embeds = env._cluster_embeds()
        phi = float(np.mean([
            compute_phi(env.embeds[n], cluster_embeds[env.cluster_assign[n]])
            for n in range(env.hp.N)
        ]))
        q_hist.append(info.q_mean)
        u_hist.append(info.u_t)
        u_qos_hist.append(info.u_qos)
        phi_hist.append(phi)
        d_pub_hist.append(1.0 - phi)
        s_pub_hist.append(info.s_pub_t)
        hit_hist.append(info.hit_ratio)
        refresh_hist.append(1.0 if info.rule_refresh else 0.0)
        if done:
            break

    u_arr = np.array(u_hist)
    u_qos_arr = np.array(u_qos_hist)
    return EpisodeResult(
        Q=np.array(q_hist),
        U=u_arr,
        U_qos=u_qos_arr,
        phi=np.array(phi_hist),
        D_t_pub=np.array(d_pub_hist),
        cum_U=np.cumsum(u_arr),
        cum_U_qos=np.cumsum(u_qos_arr),
        final_D_t=system_degradation(env.cluster_assign, env.embeds, env.M),
        stats=stats,
        S_t_pub=np.array(s_pub_hist),
        hit_ratio=np.array(hit_hist),
        refresh=np.array(refresh_hist),
    )


def aggregate_episodes(episodes: List[EpisodeResult]) -> Dict:
    q_stack = np.stack([e.Q for e in episodes], axis=0)
    u_stack = np.stack([e.U for e in episodes], axis=0)
    u_qos_stack = np.stack([e.U_qos for e in episodes], axis=0)
    phi_stack = np.stack([e.phi for e in episodes], axis=0)
    d_pub_stack = np.stack([e.D_t_pub for e in episodes], axis=0)
    cum_u_stack = np.stack([e.cum_U for e in episodes], axis=0)
    cum_u_qos_stack = np.stack([e.cum_U_qos for e in episodes], axis=0)
    return {
        "Q_mean": q_stack.mean(axis=0),
        "Q_std": q_stack.std(axis=0),
        "U_mean": u_stack.mean(axis=0),
        "U_std": u_stack.std(axis=0),
        "U_qos_mean": u_qos_stack.mean(axis=0),
        "U_qos_std": u_qos_stack.std(axis=0),
        "phi_mean": phi_stack.mean(axis=0),
        "phi_std": phi_stack.std(axis=0),
        "D_t_pub_mean": d_pub_stack.mean(axis=0),
        "D_t_pub_std": d_pub_stack.std(axis=0),
        "cum_U_mean": cum_u_stack.mean(axis=0),
        "cum_U_std": cum_u_stack.std(axis=0),
        "cum_U_qos_mean": cum_u_qos_stack.mean(axis=0),
        "cum_U_qos_std": cum_u_qos_stack.std(axis=0),
        "move_ops": float(np.mean([e.stats.move_ops for e in episodes])),
        "recluster_ops": float(np.mean([e.stats.recluster_ops for e in episodes])),
        "D_t_final": float(np.mean([e.final_D_t for e in episodes])),
        "D_t_pub_final": float(np.mean([e.D_t_pub[-1] for e in episodes])),
        "Q_final": float(np.mean([e.Q[-1] for e in episodes])),
        "U_final": float(np.mean([e.U[-1] for e in episodes])),
        "cum_U_final": float(np.mean([e.cum_U[-1] for e in episodes])),
    }
