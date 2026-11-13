"""Shared evaluation entry: Static-HIS / IIS / NIRVANA."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
import torch

from algorithms.cache_manager import ClusterCacheManager
from baselines.published_policies import ClusterPolicy
from core.clustering import compute_phi, hac_clustering
from core.degradation import build_cluster_embeds, cluster_centroid, system_degradation
from core.hiqm import HIQM
from core.ippo import IPPOTrainer
from core.metrics import compute_latency_and_energy
from core.params import (
    CacheOverheadRefs,
    CacheThresholds,
    ClusterOverheadRefs,
    HardwareParams,
    HyperParams,
    UtilityNormRefs,
)
from core.seed import set_seed
from core.utility import (
    compute_cache_overhead,
    compute_network_utility,
    compute_step_metrics_with_cache,
    es_energy_share,
    hiqm_quality,
    policy_cluster_overhead,
)
from env.drift import apply_drift_step, make_drift_trajectory
from env.resources import make_resource_trajectory, resource_rng
from env.et_static_env import EMBED_DIM, ETStaticEnv, init_embeds, select_ippo_actions

SchemeKind = Literal["static", "iis"]


@dataclass
class EvalResult:
    u_mean: float
    u_std: float
    cum_u: float
    move_ops: float
    recluster_ops: float
    schedule_ops: float = 0.0
    ddrs_qs: float = 0.0


def load_ippo(hp: HyperParams, path: str) -> IPPOTrainer:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    trainer = IPPOTrainer(hp)
    state = torch.load(path, map_location="cpu", weights_only=False)
    for n in range(hp.N):
        trainer.actors[n].load_state_dict(state[f"actor_{n}"])
        trainer.critics[n].load_state_dict(state[f"critic_{n}"])
    for actor in trainer.actors:
        actor.eval()
    trainer.set_frozen(True)
    return trainer


def _iis_actions(
    ippo_actions: List[Tuple[float, float]],
    hp: HyperParams,
) -> List[Tuple[float, float]]:
    return [(0.0, float(min(y, hp.T_max))) for _x, y in ippo_actions]


def run_baseline_episode(
    policy: ClusterPolicy,
    scheme: SchemeKind,
    hp: HyperParams,
    hw: HardwareParams,
    hiqm: HIQM,
    seeds_rng: np.random.RandomState,
    seed: int,
    T: int,
    p_drift: float,
    sigma: float,
    enable_cache: bool,
    overhead_refs: ClusterOverheadRefs,
    cache_refs: CacheOverheadRefs,
    utility_refs: UtilityNormRefs,
    ippo: IPPOTrainer,
) -> EvalResult:
    set_seed(seed)

    init_rng = np.random.RandomState(seed)
    drift_rng = np.random.RandomState(seed + 10000)
    res_rng = resource_rng(seed)

    embeds = init_embeds(init_rng, hp.N, hp.M)
    assign = hac_clustering(embeds, hp.M)
    masks, noise = make_drift_trajectory(drift_rng, T, hp.N, EMBED_DIM, p_drift, sigma)
    v_es_traj, v_ue_traj = make_resource_trajectory(res_rng, T, hp)

    cache = ClusterCacheManager(enable_cache, CacheThresholds())

    env = ETStaticEnv(
        hp, hw, hiqm, embeds, assign, masks, noise, v_es_traj, v_ue_traj, seeds_rng,
        prompt_refresh=policy.prompt_refresh,
        ippo_trainer=ippo,
    )
    policy.reset(env.cluster_assign)
    cache.reset()

    u_hist: List[float] = []
    move_ops = recluster_ops = schedule_ops = 0

    for t in range(T):
        env.apply_drift(t)
        v_es, v_ue = int(env.v_max_ES), int(env.v_UE)

        old_assign = list(env.cluster_assign)
        new_assign, info = policy.update(
            t, env.embeds, env.cluster_assign,
            system_degradation(env.cluster_assign, env.embeds, hp.M), hp.M,
        )
        env.cluster_assign = new_assign
        if info.refresh_prompt:
            affected = getattr(info, "affected_clusters", None)
            if info.recluster or not affected:
                env.refresh_public_prompts()
            else:
                env.refresh_selected_prompts(affected)
        if info.recluster:
            cache.invalidate_on_recluster()
            recluster_ops += 1
        elif info.moved > 0:
            cache.inherit_on_move(
                old_assign, env.cluster_assign, env.service_cluster_embeds(), env.embeds, t,
            )
        if getattr(info, "move_scan", False):
            move_ops += 1
        if getattr(info, "schedule", False):
            schedule_ops += 1

        cluster_embeds = env.service_cluster_embeds()
        ippo_actions = select_ippo_actions(
            ippo, env.embeds, env.cluster_assign, cluster_embeds, hp, v_es, v_ue,
        )

        if scheme == "iis":
            actions = _iis_actions(ippo_actions, hp)
        else:
            actions = ippo_actions

        x_common_eff = [0.0] * hp.M
        hits = misses = 0
        cluster_x: Dict[int, float] = {}

        for m in range(hp.M):
            members = [n for n in range(hp.N) if env.cluster_assign[n] == m]
            if not members:
                continue
            x_m = max(actions[n][0] for n in members)
            cluster_x[m] = x_m
            mu = cluster_centroid(members, env.embeds)
            lookup = cache.lookup(m, cluster_embeds[m], mu, x_m, t)
            if lookup.hit:
                hits += 1
                x_common_eff[m] = max(0.0, x_m - lookup.x_saved)
            else:
                misses += 1
                x_common_eff[m] = x_m

        _, u_qos, _, _, _ = compute_step_metrics_with_cache(
            env.embeds, env.cluster_assign, cluster_embeds,
            hp, hw, hiqm, actions, v_es, v_ue, utility_refs,
            x_common_eff=x_common_eff,
        )
        u = u_qos - policy_cluster_overhead(
            info.recluster, info.recluster_time, info.moved, overhead_refs, policy.name,
            move_scan=bool(getattr(info, "move_scan", False)),
        )
        if cache.enabled:
            u -= compute_cache_overhead(hits, misses, cache_refs)

        for m, x_m in cluster_x.items():
            members = [n for n in range(hp.N) if env.cluster_assign[n] == m]
            cache.store(m, cluster_embeds[m], cluster_centroid(members, env.embeds), x_m, t)

        u_hist.append(u)

    cum = float(np.sum(u_hist))
    return EvalResult(
        u_mean=cum / T,
        u_std=0.0,
        cum_u=cum,
        move_ops=float(move_ops),
        recluster_ops=float(recluster_ops),
        schedule_ops=float(schedule_ops),
    )


NIRVANA_CHECKPOINTS = (5, 10, 15, 20, 25)


NIRVANA_SAMPLER_STEPS = 50


def nirvana_skip(sim: float) -> int:
    """NSDI 2024 Fig. 10: similarity to the number of steps that can be skipped on a 50-step sampler."""
    if sim > 0.95:
        return 25
    if sim > 0.9:
        return 20
    if sim > 0.85:
        return 15
    if sim > 0.75:
        return 10
    if sim > 0.65:
        return 5
    return 0


def nirvana_local_steps(k_paper: int, hp: HyperParams) -> int:
    """Map the skip depth on a 50-step sampler onto this simulation's step budget by S/50."""
    if k_paper <= 0:
        return 0
    return int(round(k_paper * hp.T_max / NIRVANA_SAMPLER_STEPS))


def _nirvana_match(
    sims: np.ndarray,
    cache_bits: np.ndarray,
    ckpt_bit: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Take the entry with the largest cosine. Equal cosines keep the later write. If that entry has no checkpoint, the skip depth is 0."""
    n_cached = sims.shape[1]
    rev = (n_cached - 1) - np.argmax(sims[:, ::-1], axis=1)
    best_sims = sims[np.arange(sims.shape[0]), rev]
    bits = cache_bits[rev]
    k_skips = np.zeros(sims.shape[0], dtype=np.int64)
    phis = np.zeros(sims.shape[0], dtype=np.float64)
    for n in range(sims.shape[0]):
        k_skip = nirvana_skip(float(best_sims[n]))
        if k_skip != 0 and (int(bits[n]) & ckpt_bit[k_skip]) != 0:
            k_skips[n] = k_skip
            phis[n] = float(best_sims[n])
    return k_skips, phis


def run_nirvana_episode(
    hp: HyperParams,
    hw: HardwareParams,
    hiqm: HIQM,
    seed: int,
    T: int,
    p_drift: float,
    sigma: float,
    utility_refs: UtilityNormRefs,
    ippo: IPPOTrainer,
) -> EvalResult:
    """One approximate cache. The ladder depth is mapped by S/50, and the steps still executed follow the IIS split."""
    set_seed(seed)
    init_rng = np.random.RandomState(seed)
    drift_rng = np.random.RandomState(seed + 10000)
    res_rng = resource_rng(seed)

    embeds = init_embeds(init_rng, hp.N, hp.M)
    assign = hac_clustering(embeds, hp.M)
    cluster_embeds = build_cluster_embeds(assign, embeds, hp.M, EMBED_DIM, init_rng)
    members = [
        [n for n in range(hp.N) if assign[n] == m]
        for m in range(hp.M)
    ]
    masks, noise = make_drift_trajectory(drift_rng, T, hp.N, EMBED_DIM, p_drift, sigma)
    v_es_traj, v_ue_traj = make_resource_trajectory(res_rng, T, hp)

    cap = T * hp.N
    cache_emb = np.empty((cap, EMBED_DIM), dtype=np.float64)
    cache_bits = np.zeros(cap, dtype=np.uint8)
    ckpt_bit = {k: 1 << i for i, k in enumerate(NIRVANA_CHECKPOINTS)}
    n_cached = 0
    u_hist: List[float] = []

    for t in range(T):
        embeds = apply_drift_step(embeds, masks[t], noise[t])
        query = np.stack(embeds).astype(np.float64, copy=False)
        v_es = float(v_es_traj[t])
        v_ue = float(v_ue_traj[t])

        if n_cached == 0:
            k_skips = np.zeros(hp.N, dtype=np.int64)
            phis = np.zeros(hp.N, dtype=np.float64)
        else:
            base = cache_emb[:n_cached]
            qn = np.linalg.norm(query, axis=1, keepdims=True)
            cn = np.linalg.norm(base, axis=1, keepdims=True)
            sims = (query @ base.T) / (qn * cn.T + 1e-8)
            np.clip(sims, 0.0, 1.0, out=sims)
            k_skips, phis = _nirvana_match(sims, cache_bits[:n_cached], ckpt_bit)

        actions = select_ippo_actions(
            ippo, embeds, assign, cluster_embeds, hp, int(v_es), int(v_ue),
        )
        y_exec = np.zeros(hp.N)
        z_exec = np.zeros(hp.N)
        for n in range(hp.N):
            y_n = float(min(actions[n][1], hp.T_max))
            z_n = float(hp.T_max - y_n)
            k_exec = float(nirvana_local_steps(int(k_skips[n]), hp))
            skip_y = min(k_exec, y_n)
            skip_z = min(max(k_exec - y_n, 0.0), z_n)
            y_exec[n] = y_n - skip_y
            z_exec[n] = z_n - skip_z

        v_m = v_es / hp.M
        y_sum = []
        for group in members:
            y_sum.append(float(y_exec[group].sum()) + 1e-8)

        utilities = []
        for n in range(hp.N):
            m = assign[n]
            k_skip = int(k_skips[n])
            T_n, E_n, E_m_C, E_m_PE = compute_latency_and_energy(
                0.0, float(y_exec[n]), float(z_exec[n]), 0.0, y_sum[m],
                v_ue, v_m, hp, hw,
            )
            es_share = es_energy_share(0.0, float(y_exec[n]), 1.0, y_sum[m], E_m_C, E_m_PE)
            if k_skip > 0:
                Q_n = hiqm_quality(
                    hiqm, float(nirvana_local_steps(k_skip, hp)), float(phis[n]), hp,
                )
            else:
                Q_n = hiqm_quality(hiqm, 0.0, compute_phi(embeds[n], cluster_embeds[m]), hp)
            utilities.append(
                compute_network_utility(Q_n, T_n, E_n + es_share, hp, utility_refs)
            )
        u_hist.append(float(np.mean(utilities)))

        for n in range(hp.N):
            k_skip = int(k_skips[n])
            mask = 0
            for ck in NIRVANA_CHECKPOINTS:
                if ck <= hp.T_max and (k_skip == 0 or k_skip < ck):
                    mask |= ckpt_bit[ck]
            cache_emb[n_cached] = query[n]
            cache_bits[n_cached] = mask
            n_cached += 1

    cum = float(np.sum(u_hist))
    return EvalResult(
        u_mean=cum / T,
        u_std=0.0,
        cum_u=cum,
        move_ops=0.0,
        recluster_ops=0.0,
    )


def eval_scheme_seeds(
    run_fn: Callable[[int], EvalResult],
    seeds: List[int],
    slot_steps: int,
) -> EvalResult:
    cum_us, moves, reclusters, scheds = [], [], [], []
    for s in seeds:
        r = run_fn(s)
        cum_us.append(r.cum_u)
        moves.append(r.move_ops)
        reclusters.append(r.recluster_ops)
        scheds.append(r.schedule_ops)
    cum = np.array(cum_us, dtype=float)
    u_means = cum / slot_steps
    return EvalResult(
        u_mean=float(u_means.mean()),
        u_std=float(u_means.std(ddof=0)),
        cum_u=float(cum.mean()),
        move_ops=float(np.mean(moves)),
        recluster_ops=float(np.mean(reclusters)),
        schedule_ops=float(np.mean(scheds)),
    )
