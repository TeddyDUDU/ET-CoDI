"""ET-CoDI: the CM learns keep / stream_move / recluster. Rule refresh breaks false hits."""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from baselines.cluster_policies import ClusterPolicy

import numpy as np

from algorithms.cache_manager import ClusterCacheManager
from algorithms.cluster_manager import (
    CM_KEEP,
    CMStepInfo,
    build_cm_obs,
    cm_action_mask,
    execute_cm_action,
    select_rule_refresh_clusters,
)
from core.ippo import IPPOTrainer, normalize_obs
from core.clustering import compute_phi, hac_clustering
from core.degradation import (
    build_cluster_embeds,
    cluster_centroid,
    cluster_degradation,
    cluster_pub_staleness,
    system_degradation,
    system_pub_staleness,
)
from core.hiqm import HIQM
from core.metrics import compute_latency_and_energy
from core.params import (
    CacheOverheadRefs,
    CacheThresholds,
    CMParams,
    ClusterOverheadRefs,
    HardwareParams,
    HyperParams,
    UtilityNormRefs,
)
from core.utility import (
    compute_agent_reward,
    compute_cache_overhead,
    compute_cluster_overhead,
    compute_step_metrics_with_cache,
    es_energy_share,
    hiqm_quality,
)
from env.drift import apply_comove_step, apply_drift_step, make_comove_trajectory, make_drift_trajectory
from env.resources import make_resource_trajectory, resource_rng
from env.et_static_env import EMBED_DIM, init_embeds


@dataclass
class StreamStepInfo:
    cm_action: int = CM_KEEP
    u_t: float = 0.0
    u_qos: float = 0.0
    q_mean: float = 0.0
    hit_ratio: float = 0.0
    moved: int = 0
    move_triggered: bool = False
    recluster: bool = False
    rule_refresh: bool = False
    M: int = 0
    hits: int = 0
    misses: int = 0
    false_safe_hits: int = 0
    reused_steps: float = 0.0
    x_common_exec: tuple = ()
    refreshed_clusters: int = 0
    d_t: float = 0.0
    s_pub_t: float = 0.0
    latency_mean: float = 0.0
    energy_mean: float = 0.0
    maint_cost: float = 0.0


@dataclass
class EpisodeClusterStats:
    cum_u: float = 0.0
    cum_u_qos: float = 0.0
    move_ops: int = 0
    recluster_ops: int = 0
    refresh_ops: int = 0
    u_mean: float = 0.0
    action_dist: Optional[Dict[str, float]] = None
    hits: int = 0
    misses: int = 0
    false_safe_hits: int = 0
    reused_steps: float = 0.0
    hit_ratio: float = 0.0
    false_safe_rate: float = 0.0
    d_mean: float = 0.0
    s_pub_mean: float = 0.0
    q_mean: float = 0.0
    latency_mean: float = 0.0
    energy_mean: float = 0.0
    runtime_seconds: float = 0.0
    maint_cost: float = 0.0
    lookup_comparisons: float = 0.0
    lookup_calls: float = 0.0
    _d_sum: float = 0.0
    _s_pub_sum: float = 0.0
    _q_sum: float = 0.0
    _latency_sum: float = 0.0
    _energy_sum: float = 0.0

    def add_step(self, info: StreamStepInfo, *, streamhac_step: bool) -> None:
        self.cum_u += info.u_t
        self.cum_u_qos += info.u_qos
        if info.rule_refresh:
            self.refresh_ops += 1
        self.hits += info.hits
        self.misses += info.misses
        self.false_safe_hits += info.false_safe_hits
        self.reused_steps += info.reused_steps
        self.maint_cost += info.maint_cost
        self._d_sum += info.d_t
        self._s_pub_sum += info.s_pub_t
        self._q_sum += info.q_mean
        self._latency_sum += info.latency_mean
        self._energy_sum += info.energy_mean
        if streamhac_step:
            self.move_ops += 1
            if info.recluster:
                self.recluster_ops += 1
        else:
            if info.move_triggered:
                self.move_ops += 1
            if info.recluster:
                self.recluster_ops += 1

    def finalize(self, n_steps: int) -> None:
        self.u_mean = self.cum_u / max(n_steps, 1)
        lookups = self.hits + self.misses
        self.hit_ratio = self.hits / lookups if lookups else 0.0
        self.false_safe_rate = self.false_safe_hits / self.hits if self.hits else 0.0
        self.d_mean = self._d_sum / max(n_steps, 1)
        self.s_pub_mean = self._s_pub_sum / max(n_steps, 1)
        self.q_mean = self._q_sum / max(n_steps, 1)
        self.latency_mean = self._latency_sum / max(n_steps, 1)
        self.energy_mean = self._energy_sum / max(n_steps, 1)


class ETStreamEnv:
    """The CM chooses keep / stream_move / recluster. Refresh is a deterministic rule."""

    def __init__(
        self,
        hp: HyperParams,
        hw: HardwareParams,
        hiqm: HIQM,
        cm_params: CMParams,
        overhead_refs: ClusterOverheadRefs,
        cache_refs: CacheOverheadRefs,
        cache_thresholds: CacheThresholds,
        utility_refs: UtilityNormRefs,
        slot_steps: int = 500,
        enable_drift: bool = True,
        enable_cache: bool = True,
        p_drift: float = 0.1,
        sigma: float = 0.1,
        drift_mode: str = "independent",
        p_group: float = 0.1,
        sigma_v: float = 0.12,
        sigma_eps: float = 0.02,
        es_scale: float = 1.0,
        b_max: Optional[int] = None,
    ):
        self.hp = hp
        self.hw = hw
        self.hiqm = hiqm
        self.cm_params = cm_params
        self.overhead_refs = overhead_refs
        self.cache_refs = cache_refs
        self.utility_refs = utility_refs
        self.slot_steps = slot_steps
        self.enable_drift = enable_drift
        self.enable_cache = enable_cache
        self.p_drift = p_drift
        self.sigma = sigma
        self.drift_mode = drift_mode
        self.p_group = p_group
        self.sigma_v = sigma_v
        self.sigma_eps = sigma_eps
        self.es_scale = es_scale
        self.b_max = b_max

        self.cache = ClusterCacheManager(enable_cache, cache_thresholds, b_max=b_max)
        self.comove_groups: Optional[np.ndarray] = None
        self._comove_shared: Optional[np.ndarray] = None
        self._comove_eps: Optional[np.ndarray] = None
        self.rng: Optional[np.random.RandomState] = None
        self.embeds: List[np.ndarray] = []
        self.cluster_assign: List[int] = []
        self.frozen_cluster_embeds: List[np.ndarray] = []
        self.drift_masks: Optional[np.ndarray] = None
        self.drift_noise: Optional[np.ndarray] = None
        self.v_es_traj: Optional[np.ndarray] = None
        self.v_ue_traj: Optional[np.ndarray] = None
        self.t = 0
        self.steps_since_move = 0
        self.steps_since_recluster = 0
        self.steps_since_refresh = 0
        self.v_max_ES = 0
        self.v_UE = 0
        self._recent_u: deque = deque(maxlen=20)
        self._recent_hit: deque = deque(maxlen=20)
        self._d_before = 0.0
        self._slot_refresh: List[int] = []

    @property
    def M(self) -> int:
        return self.hp.M

    def reset(self, seed: int) -> np.ndarray:
        self.rng = np.random.RandomState(seed)
        init_rng = np.random.RandomState(seed)
        drift_rng = np.random.RandomState(seed + 10000)
        res_rng = resource_rng(seed)

        initial_embeds = init_embeds(init_rng, self.hp.N, self.M)
        self.cluster_assign = hac_clustering(initial_embeds, self.M)
        self.embeds = [e.copy() for e in initial_embeds]
        self.frozen_cluster_embeds = build_cluster_embeds(
            self.cluster_assign, self.embeds, self.M, EMBED_DIM, self.rng,
        )

        self.comove_groups = None
        self._comove_shared = None
        self._comove_eps = None
        if self.enable_drift and self.drift_mode == "comove":
            self.comove_groups = np.asarray(self.cluster_assign, dtype=np.int64)
            self._comove_shared, self._comove_eps = make_comove_trajectory(
                drift_rng, self.slot_steps, self.hp.N, EMBED_DIM, self.comove_groups,
                self.p_group, self.sigma_v, self.sigma_eps,
            )
            self.drift_masks = np.zeros((self.slot_steps, self.hp.N), dtype=bool)
            self.drift_noise = np.zeros((self.slot_steps, self.hp.N, EMBED_DIM))
        elif self.enable_drift:
            self.drift_masks, self.drift_noise = make_drift_trajectory(
                drift_rng, self.slot_steps, self.hp.N, EMBED_DIM, self.p_drift, self.sigma,
            )
        else:
            self.drift_masks = np.zeros((self.slot_steps, self.hp.N), dtype=bool)
            self.drift_noise = np.zeros((self.slot_steps, self.hp.N, EMBED_DIM))

        self.v_es_traj, self.v_ue_traj = make_resource_trajectory(
            res_rng, self.slot_steps, self.hp,
        )
        if self.es_scale != 1.0:
            scaled = np.rint(self.v_es_traj.astype(np.float64) * float(self.es_scale))
            self.v_es_traj = np.maximum(1, scaled).astype(np.int64)

        self.cache.reset()
        self.t = 0
        self.steps_since_move = 0
        self.steps_since_recluster = 0
        self.steps_since_refresh = 0
        self._recent_u.clear()
        self._recent_hit.clear()
        self.v_max_ES = int(self.v_es_traj[0])
        self.v_UE = int(self.v_ue_traj[0])
        return self.begin_slot()

    def _slot_resources(self, t: int) -> Tuple[int, int]:
        assert self.v_es_traj is not None and self.v_ue_traj is not None
        idx = min(max(t, 0), len(self.v_es_traj) - 1)
        return int(self.v_es_traj[idx]), int(self.v_ue_traj[idx])

    def _cluster_embeds(self) -> List[np.ndarray]:
        return [e.copy() for e in self.frozen_cluster_embeds]

    def _refresh_frozen_prompts(self) -> None:
        self.frozen_cluster_embeds = build_cluster_embeds(
            self.cluster_assign, self.embeds, self.M, EMBED_DIM, self.rng,
        )

    def _refresh_selected_prompts(self, cluster_ids: List[int]) -> None:
        for m in cluster_ids:
            members = [n for n in range(self.hp.N) if self.cluster_assign[n] == m]
            if members:
                self.frozen_cluster_embeds[m] = cluster_centroid(members, self.embeds)

    def _per_cluster_degradation(self) -> List[float]:
        return [
            cluster_degradation(
                [n for n in range(self.hp.N) if self.cluster_assign[n] == m], self.embeds,
            )
            for m in range(self.M)
        ]

    def _per_cluster_pub_staleness(self) -> List[float]:
        embeds = self._cluster_embeds()
        return [
            cluster_pub_staleness(
                [n for n in range(self.hp.N) if self.cluster_assign[n] == m],
                self.embeds,
                embeds[m],
            )
            for m in range(self.M)
        ]

    def get_cm_obs(self) -> np.ndarray:
        cluster_embeds = self._cluster_embeds()
        return build_cm_obs(
            system_degradation(self.cluster_assign, self.embeds, self.M),
            system_pub_staleness(
                self.cluster_assign, self.embeds, cluster_embeds, self.M,
            ),
            self._per_cluster_degradation(),
            self._per_cluster_pub_staleness(),
            self.steps_since_move,
            self.steps_since_refresh,
            self.steps_since_recluster,
            self.M,
            self.slot_steps,
            self.cm_params,
        )

    def begin_slot(self) -> np.ndarray:
        """First half of the slot: drift, then D and S^pub, then rule refresh, then the post-refresh CM observation."""
        self._apply_drift()
        self._d_before = system_degradation(self.cluster_assign, self.embeds, self.M)
        self._slot_refresh = self._apply_rule_refresh()
        return self.get_cm_obs()

    def cm_action_mask(self) -> np.ndarray:
        return cm_action_mask(
            self.steps_since_recluster,
            self.steps_since_move,
            self.cm_params,
        )

    def _apply_drift(self) -> None:
        if not self.enable_drift or self.t >= self.slot_steps:
            return
        if self.drift_mode == "comove" and self._comove_shared is not None:
            self.embeds = apply_comove_step(
                self.embeds, self.comove_groups, self._comove_shared[self.t], self._comove_eps[self.t],
            )
            return
        self.embeds = apply_drift_step(
            self.embeds, self.drift_masks[self.t], self.drift_noise[self.t],
        )

    def _apply_cluster_change(self, old_assign: List[int], cm_info: CMStepInfo) -> None:
        if cm_info.recluster:
            self._refresh_frozen_prompts()
            self.cache.invalidate_on_recluster()
        elif cm_info.moved > 0:
            self._refresh_selected_prompts(cm_info.affected_clusters or [])
            self.cache.inherit_on_move(
                old_assign, self.cluster_assign, self._cluster_embeds(), self.embeds, self.t,
            )

    def _apply_rule_refresh(self) -> List[int]:
        """Rule R runs before the CM decision and depends only on post-drift S^pub and D."""
        if not self.cm_params.enable_rule_refresh:
            return []
        targets = select_rule_refresh_clusters(
            self.cluster_assign,
            self.embeds,
            self._cluster_embeds(),
            self.M,
            self.cm_params,
            self.steps_since_refresh,
        )
        if not targets:
            return []
        self._refresh_selected_prompts(targets)
        self.cache.force_miss_clusters(targets)
        self.steps_since_refresh = 0
        return targets

    def _v_m_es(self) -> np.ndarray:
        alpha = np.ones(self.M)
        return self.v_max_ES * np.sqrt(alpha) / (np.sum(np.sqrt(alpha)) + 1e-8)

    def _ippo_actions(
        self, ippo: IPPOTrainer, cluster_embeds: List[np.ndarray],
    ) -> Tuple[List[Tuple[float, float]], List[np.ndarray], List[float]]:
        actions, obs_list, logps = [], [], []
        for n in range(self.hp.N):
            phi = compute_phi(self.embeds[n], cluster_embeds[self.cluster_assign[n]])
            obs = normalize_obs(phi, self.v_max_ES, self.v_UE, self.hp)
            act_raw, logp = ippo.select_action(n, obs)
            x_n = float(np.clip(round(act_raw[0]), 0, self.hp.x_max))
            y_n = float(np.clip(round(act_raw[1]), self.hp.y_min, self.hp.y_max))
            if x_n + y_n > self.hp.T_max:
                y_n = max(self.hp.y_min, self.hp.T_max - x_n)
            actions.append((x_n, y_n))
            obs_list.append(obs)
            logps.append(logp)
        return actions, obs_list, logps

    def _cache_pass(
        self,
        actions: List[Tuple[float, float]],
        cluster_embeds: List[np.ndarray],
    ) -> Tuple[List[float], int, int, float, int, float]:
        x_common_eff = [0.0] * self.M
        hits = misses = false_safe = 0
        reused_steps = 0.0
        store_queue: List[Tuple[int, float, List[int]]] = []

        for m in range(self.M):
            members = [n for n in range(self.hp.N) if self.cluster_assign[n] == m]
            if not members:
                continue
            x_m = max(actions[n][0] for n in members)
            mu = cluster_centroid(members, self.embeds)
            lookup = self.cache.lookup(m, cluster_embeds[m], mu, x_m, self.t)
            if lookup.hit:
                hits += 1
                reused_steps += lookup.x_saved
                x_common_eff[m] = max(0.0, x_m - lookup.x_saved)
                # False-hit audit: the soft threshold accepts the entry, but its prompt key is stale relative to the current centroid.
                if lookup.entry is not None:
                    stale = 1.0 - compute_phi(mu, lookup.entry.e_pub)
                    if stale > self.cm_params.tau_stale:
                        false_safe += 1
            else:
                misses += 1
                x_common_eff[m] = x_m
            store_queue.append((m, x_m, members))

        for m, x_m, members in store_queue:
            self.cache.store(m, cluster_embeds[m], cluster_centroid(members, self.embeds), x_m, self.t)

        active = sum(1 for m in range(self.M) if any(self.cluster_assign[n] == m for n in range(self.hp.N)))
        hit_ratio = hits / active if active else 0.0
        return x_common_eff, hits, misses, hit_ratio, false_safe, reused_steps

    def _pbrs_term(self, d_before: float, d_after: float) -> float:
        """Same-slot degradation-reduction shaping：Φ = −λ D ⇒ F = λ(D − γ D')."""
        lam = self.cm_params.pbrs_lambda
        if lam == 0.0:
            return 0.0
        return lam * (d_before - self.cm_params.gamma * d_after)

    def _finalize_slot(
        self,
        cm_action: int,
        cm_info: CMStepInfo,
        ippo: IPPOTrainer,
        train_ippo: bool,
        d_before: float,
    ) -> Tuple[np.ndarray, float, float, List[float], StreamStepInfo, bool]:
        cluster_embeds = self._cluster_embeds()
        actions, obs_list, logps = self._ippo_actions(ippo, cluster_embeds)
        x_common_eff, hits, misses, hit_ratio, false_safe, reused_steps = self._cache_pass(
            actions, cluster_embeds,
        )
        d_t = system_degradation(self.cluster_assign, self.embeds, self.M)
        s_pub_t = system_pub_staleness(
            self.cluster_assign, self.embeds, cluster_embeds, self.M,
        )

        q_mean, u_qos, _, _, _, latency_mean, energy_mean = compute_step_metrics_with_cache(
            self.embeds, self.cluster_assign, cluster_embeds,
            self.hp, self.hw, self.hiqm, actions, self.v_max_ES, self.v_UE,
            self.utility_refs, x_common_eff=x_common_eff, return_details=True,
        )
        maint = compute_cluster_overhead(
            cm_info.recluster, cm_info.recluster_time, cm_info.moved,
            self.overhead_refs, move_triggered=cm_info.move_triggered,
            rule_refresh=cm_info.rule_refresh,
        )
        u_t = u_qos - maint
        if self.enable_cache:
            u_t -= compute_cache_overhead(hits, misses, self.cache_refs)

        self._recent_u.append(u_t)
        self._recent_hit.append(hit_ratio)

        v_m_es = self._v_m_es()
        next_v_es, next_v_ue = self._slot_resources(self.t + 1)
        ippo_rewards: List[float] = []

        for n in range(self.hp.N):
            x_n, y_n = actions[n]
            m = self.cluster_assign[n]
            members_m = [j for j in range(self.hp.N) if self.cluster_assign[j] == m]
            x_sum = sum(actions[j][0] for j in members_m) + 1e-8
            y_sum = sum(actions[j][1] for j in members_m) + 1e-8
            T_n, E_n, E_m_C, E_m_PE = compute_latency_and_energy(
                x_n, y_n, self.hp.T_max - x_n - y_n,
                max(actions[j][0] for j in members_m), y_sum,
                self.v_UE, v_m_es[m], self.hp, self.hw, x_m_common=x_common_eff[m],
            )
            phi = compute_phi(self.embeds[n], cluster_embeds[m])
            r_n = compute_agent_reward(
                hiqm_quality(self.hiqm, x_n, phi, self.hp), T_n, E_n,
                es_energy_share(x_n, y_n, x_sum, y_sum, E_m_C, E_m_PE), self.hp,
            )
            ippo_rewards.append(r_n)
            if train_ippo and not ippo._frozen:
                ippo.buffers[n].store(
                    obs_list[n], np.array([x_n, y_n]), logps[n],
                    ippo.rnorms[n].normalize(r_n),
                    normalize_obs(phi, next_v_es, next_v_ue, self.hp), 0.0,
                )

        self.t += 1
        self.v_max_ES, self.v_UE = self._slot_resources(self.t)
        done = self.t >= self.slot_steps

        d_after = system_degradation(self.cluster_assign, self.embeds, self.M)
        cm_r = u_t + self._pbrs_term(d_before, d_after)

        info = StreamStepInfo(
            cm_action=cm_action, u_t=u_t, u_qos=u_qos, q_mean=q_mean,
            hit_ratio=hit_ratio, moved=cm_info.moved,
            move_triggered=cm_info.move_triggered, recluster=cm_info.recluster,
            rule_refresh=cm_info.rule_refresh, M=self.M,
            hits=hits, misses=misses, false_safe_hits=false_safe,
            reused_steps=reused_steps,
            x_common_exec=tuple(float(x) for x in x_common_eff),
            refreshed_clusters=len(cm_info.refreshed_clusters or []),
            d_t=d_t, s_pub_t=s_pub_t,
            latency_mean=latency_mean, energy_mean=energy_mean,
            maint_cost=maint,
        )
        self.steps_since_refresh += 1
        next_obs = self.get_cm_obs() if done else self.begin_slot()
        return next_obs, cm_r, u_t, ippo_rewards, info, done

    def _advance_timers(self, cm_info: CMStepInfo) -> None:
        if cm_info.recluster:
            self.steps_since_recluster = 0
        else:
            self.steps_since_recluster += 1
        self.steps_since_move = 0 if cm_info.move_triggered else self.steps_since_move + 1

    def step(
        self,
        cm_action: int,
        ippo: IPPOTrainer,
        train_ippo: bool,
    ) -> Tuple[np.ndarray, float, float, List[float], StreamStepInfo, bool]:
        """Second half of the slot: CM structural action, then StreamHAC, reuse, and the GIE. Drift and refresh already ran in begin_slot."""
        old_assign = list(self.cluster_assign)

        new_assign, cm_info = execute_cm_action(
            cm_action, self.cluster_assign, self.embeds, self.M, self.cm_params,
        )
        self.cluster_assign = new_assign
        cm_info.rule_refresh = bool(self._slot_refresh)
        cm_info.refreshed_clusters = list(self._slot_refresh) or None

        self._advance_timers(cm_info)
        self._apply_cluster_change(old_assign, cm_info)

        return self._finalize_slot(cm_action, cm_info, ippo, train_ippo, self._d_before)

    def step_with_rule_policy(
        self,
        policy: "ClusterPolicy",
        ippo: IPPOTrainer,
        train_ippo: bool = False,
    ) -> Tuple[np.ndarray, float, float, List[float], StreamStepInfo, bool]:
        """A scripted policy uses the same slot order. The structural action comes from the policy, not the CM."""
        old_assign = list(self.cluster_assign)

        new_assign, pinfo = policy.update(
            self.t, self.embeds, self.cluster_assign, self._d_before, self.M,
        )
        self.cluster_assign = list(new_assign)
        move_triggered = bool(pinfo.move_scan) or pinfo.moved > 0
        extra_refresh = list(pinfo.refresh_clusters or [])
        if self._slot_refresh:
            extra_refresh = []
        refreshed = list(self._slot_refresh or []) + extra_refresh
        cm_info = CMStepInfo(
            recluster=pinfo.recluster,
            move_triggered=move_triggered,
            moved=pinfo.moved,
            recluster_time=pinfo.recluster_time,
            refresh_prompt=pinfo.refresh_prompt or bool(extra_refresh),
            rule_refresh=bool(refreshed),
            refreshed_clusters=refreshed or None,
        )
        if move_triggered and pinfo.moved > 0:
            cm_info.affected_clusters = sorted({
                a for a0, a1 in zip(old_assign, self.cluster_assign)
                for a in (a0, a1) if a0 != a1
            })
        elif pinfo.affected_clusters:
            cm_info.affected_clusters = list(pinfo.affected_clusters)

        self._advance_timers(cm_info)
        self._apply_cluster_change(old_assign, cm_info)
        if extra_refresh:
            self._refresh_selected_prompts(extra_refresh)
            self.cache.force_miss_clusters(extra_refresh)
            self.steps_since_refresh = 0

        return self._finalize_slot(CM_KEEP, cm_info, ippo, train_ippo, self._d_before)
