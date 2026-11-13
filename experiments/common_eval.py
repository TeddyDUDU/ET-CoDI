"""Shared experiment helpers: baseline order, ET evaluation, and IPPO loading."""

from __future__ import annotations

import os
import time
from collections import Counter
from typing import Dict

import numpy as np
import torch

from algorithms.cluster_manager import CM_ACTIONS, CM_KEEP, CMTrainer
from baselines.cluster_policies import BASELINE_ORDER, build_baseline_policies
from core.hiqm import make_hiqm
from core.ippo import IPPOTrainer
from core.params import (
    CacheOverheadRefs,
    CacheThresholds,
    CMParams,
    ClusterOverheadRefs,
    HardwareParams,
    HyperParams,
    UtilityNormRefs,
)
from core.paths import EXP1_IPPO, norm_path
from core.seed import set_seed
from env.et_stream_env import EpisodeClusterStats, ETStreamEnv

ET_NAME = "ET-CoDI"
ALL_METHODS = (*BASELINE_ORDER, ET_NAME)


def load_ippo(hp: HyperParams | None = None, ippo_path: str | None = None) -> IPPOTrainer:
    hp = hp or HyperParams()
    path = norm_path(ippo_path or EXP1_IPPO)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"IPPO weights not found: {path}")
    trainer = IPPOTrainer(hp)
    state = torch.load(path, map_location="cpu", weights_only=False)
    for n in range(hp.N):
        trainer.actors[n].load_state_dict(state[f"actor_{n}"])
        trainer.critics[n].load_state_dict(state[f"critic_{n}"])
    for actor in trainer.actors:
        actor.eval()
    return trainer


ET_AVG_FIELDS = (
    "cum_u", "cum_u_qos", "move_ops", "recluster_ops", "refresh_ops", "u_mean",
    "hits", "misses", "false_safe_hits", "reused_steps", "hit_ratio",
    "false_safe_rate", "d_mean", "s_pub_mean", "q_mean", "latency_mean",
    "energy_mean", "runtime_seconds", "maint_cost",
    "lookup_comparisons", "lookup_calls",
)


def make_stream_env(
    *,
    enable_cache: bool,
    p_drift: float = 0.1,
    sigma: float = 0.1,
    slot_steps: int = 500,
    cm_params: CMParams | None = None,
    hp: HyperParams | None = None,
    hw: HardwareParams | None = None,
    overhead_refs: ClusterOverheadRefs | None = None,
    cache_refs: CacheOverheadRefs | None = None,
    cache_thresholds: CacheThresholds | None = None,
    utility_refs: UtilityNormRefs | None = None,
    drift_mode: str = "independent",
    p_group: float = 0.1,
    sigma_v: float = 0.12,
    sigma_eps: float = 0.02,
    es_scale: float = 1.0,
    b_max: int | None = None,
) -> ETStreamEnv:
    hp, hw = hp or HyperParams(), hw or HardwareParams()
    return ETStreamEnv(
        hp, hw, make_hiqm(hp), cm_params or CMParams(),
        overhead_refs or ClusterOverheadRefs(),
        cache_refs or CacheOverheadRefs(),
        cache_thresholds or CacheThresholds(),
        utility_refs or UtilityNormRefs(),
        slot_steps=slot_steps, enable_drift=True, enable_cache=enable_cache,
        p_drift=p_drift, sigma=sigma,
        drift_mode=drift_mode, p_group=p_group, sigma_v=sigma_v, sigma_eps=sigma_eps,
        es_scale=es_scale, b_max=b_max,
    )


def run_et_eval(
    cm_path: str,
    *,
    enable_cache: bool,
    seeds: list[int],
    p_drift: float = 0.1,
    sigma: float = 0.1,
    slot_steps: int = 500,
    cm_params: CMParams | None = None,
    disabled_actions: tuple[int, ...] = (),
    overhead_refs: ClusterOverheadRefs | None = None,
    cache_refs: CacheOverheadRefs | None = None,
    cache_thresholds: CacheThresholds | None = None,
    drift_mode: str = "independent",
    p_group: float = 0.1,
    sigma_v: float = 0.12,
    sigma_eps: float = 0.02,
    es_scale: float = 1.0,
    b_max: int | None = None,
    hp: HyperParams | None = None,
) -> EpisodeClusterStats:
    if not os.path.isfile(cm_path):
        raise FileNotFoundError(f"CM weights not found: {cm_path}")

    hp = hp or HyperParams()
    ippo = load_ippo(hp)
    ippo.set_frozen(True)
    cm = CMTrainer(hp, cm_params or CMParams())
    cm.load(cm_path)
    cm.set_frozen(True)

    totals = EpisodeClusterStats()
    action_counts: Counter[int] = Counter()
    n_ep = 0

    for seed in seeds:
        started = time.perf_counter()
        set_seed(seed)
        env = make_stream_env(
            enable_cache=enable_cache, p_drift=p_drift, sigma=sigma, slot_steps=slot_steps,
            cm_params=cm_params,
            overhead_refs=overhead_refs, cache_refs=cache_refs,
            cache_thresholds=cache_thresholds,
            drift_mode=drift_mode, p_group=p_group, sigma_v=sigma_v, sigma_eps=sigma_eps,
            es_scale=es_scale, b_max=b_max, hp=hp,
        )
        cm_obs = env.reset(seed)
        ep = EpisodeClusterStats()
        n_steps = 0
        while True:
            mask = env.cm_action_mask()
            for action in disabled_actions:
                if action != CM_KEEP:
                    mask[action] = False
            action, _ = cm.select_deploy_action(cm_obs, mask=mask)
            cm_obs, _, _, _, info, done = env.step(action, ippo, False)
            action_counts[action] += 1
            ep.add_step(info, streamhac_step=False)
            n_steps += 1
            if done:
                break
        ep.finalize(n_steps)
        ep.lookup_comparisons = float(env.cache.lookup_comparisons)
        ep.lookup_calls = float(env.cache.lookup_calls)
        ep.runtime_seconds = time.perf_counter() - started
        for f in ET_AVG_FIELDS:
            setattr(totals, f, getattr(totals, f) + getattr(ep, f))
        n_ep += 1

    for f in ET_AVG_FIELDS:
        setattr(totals, f, getattr(totals, f) / max(n_ep, 1))
    total = sum(action_counts.values()) or 1
    totals.action_dist = {n: action_counts[i] / total for i, n in enumerate(CM_ACTIONS)}
    return totals


def baseline_policies():
    by_name = {p.name: p for p in build_baseline_policies()}
    return [by_name[n] for n in BASELINE_ORDER]
