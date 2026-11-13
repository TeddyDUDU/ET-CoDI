"""Scripted triggers. They share ETStreamEnv bookkeeping with ET-CoDI."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Sequence

import numpy as np

from algorithms.stream_hac import hac_recluster, stream_move
from baselines.cluster_policies import PolicyStepInfo
from core.degradation import cluster_pub_staleness, system_degradation, system_pub_staleness
from core.params import CMParams, CacheThresholds, HyperParams
from env.et_stream_env import EpisodeClusterStats, ETStreamEnv
from experiments.common_eval import load_ippo, make_stream_env, run_et_eval

ACTION_KEEP = "keep"
ACTION_MOVE = "stream_move"
ACTION_RECLUSTER = "recluster"
ACTION_REFRESH = "refresh"


@dataclass
class AggRow:
    name: str
    cum_u: float
    cum_u_std: float
    cum_u_qos: float
    cum_u_qos_std: float
    move_ops: float
    move_ops_std: float
    recluster_ops: float
    recluster_ops_std: float
    refresh_ops: float
    refresh_ops_std: float
    maint_cost: float
    maint_cost_std: float
    hit_ratio: float
    false_safe_rate: float
    reused_steps: float
    d_mean: float
    s_pub_mean: float
    q_mean: float
    latency_mean: float
    lookup_comparisons: float
    lookup_calls: float


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate_stats(name: str, rows: Sequence[EpisodeClusterStats]) -> AggRow:
    def ms(field: str) -> tuple[float, float]:
        return _mean_std([float(getattr(r, field)) for r in rows])

    cu, cu_s = ms("cum_u")
    qos, qos_s = ms("cum_u_qos")
    mv, mv_s = ms("move_ops")
    rc, rc_s = ms("recluster_ops")
    rf, rf_s = ms("refresh_ops")
    mc, mc_s = ms("maint_cost")
    hit, _ = ms("hit_ratio")
    fs, _ = ms("false_safe_rate")
    rs, _ = ms("reused_steps")
    d_m, _ = ms("d_mean")
    s_m, _ = ms("s_pub_mean")
    q_m, _ = ms("q_mean")
    lat, _ = ms("latency_mean")
    comp, _ = ms("lookup_comparisons")
    calls, _ = ms("lookup_calls")
    return AggRow(
        name=name,
        cum_u=cu, cum_u_std=cu_s,
        cum_u_qos=qos, cum_u_qos_std=qos_s,
        move_ops=mv, move_ops_std=mv_s,
        recluster_ops=rc, recluster_ops_std=rc_s,
        refresh_ops=rf, refresh_ops_std=rf_s,
        maint_cost=mc, maint_cost_std=mc_s,
        hit_ratio=hit, false_safe_rate=fs, reused_steps=rs,
        d_mean=d_m, s_pub_mean=s_m, q_mean=q_m, latency_mean=lat,
        lookup_comparisons=comp, lookup_calls=calls,
    )


def maintenance_events(row: EpisodeClusterStats | AggRow) -> float:
    return float(row.move_ops) + float(row.recluster_ops) + float(row.refresh_ops)


class Trigger:
    name = "trigger"
    use_rule_refresh = False
    respect_cooldown = True
    tau_s = 0.15

    def reset(self) -> None:
        return None

    def decide(self, t: int, d_t: float, s_pub: float) -> str:
        return ACTION_KEEP


class KeepTrigger(Trigger):
    name = "Static-HIS"
    use_rule_refresh = False
    respect_cooldown = True


class PeriodicMove(Trigger):
    """Fixed period: migrate every k_move steps, and recluster instead every k_recluster steps."""

    name = "Periodic-K"
    use_rule_refresh = False

    def __init__(self, k_move: int, k_recluster: int):
        self.k_move = int(k_move)
        self.k_recluster = int(k_recluster)
        self.name = f"Periodic-{self.k_move}/{self.k_recluster}"

    def decide(self, t: int, d_t: float, s_pub: float) -> str:
        step = t + 1
        if self.k_recluster > 0 and step % self.k_recluster == 0:
            return ACTION_RECLUSTER
        if self.k_move > 0 and step % self.k_move == 0:
            return ACTION_MOVE
        return ACTION_KEEP


class ThresholdD(Trigger):
    name = "Threshold-D"
    use_rule_refresh = True

    def __init__(self, tau_d: float):
        self.tau_d = float(tau_d)
        self.name = "Threshold-D"

    def decide(self, t: int, d_t: float, s_pub: float) -> str:
        if d_t > self.tau_d:
            return ACTION_MOVE
        return ACTION_KEEP


class ThresholdMaintain(Trigger):
    """Two-level threshold: recluster when D exceeds its gate, otherwise migrate when S^pub exceeds its gate.

    If the recluster cooldown has not elapsed and the move condition still holds, downgrade to a move. Never downgrade to a refresh.
    """

    name = "Threshold"
    use_rule_refresh = False
    downgrade_recluster_to_move = True

    def __init__(self, tau_s: float = 0.15, tau_d: float = 0.85):
        self.tau_s = float(tau_s)
        self.tau_d = float(tau_d)
        self._s_pub = 0.0
        self.name = "Threshold"

    def decide(self, t: int, d_t: float, s_pub: float) -> str:
        self._s_pub = float(s_pub)
        if d_t > self.tau_d:
            return ACTION_RECLUSTER
        if s_pub > self.tau_s:
            return ACTION_MOVE
        return ACTION_KEEP

    def wants_move(self) -> bool:
        return self._s_pub > self.tau_s


class RefreshOnly(Trigger):
    name = "Refresh-only"
    use_rule_refresh = False

    def __init__(self, tau_s: float = 0.15):
        self.tau_s = float(tau_s)

    def decide(self, t: int, d_t: float, s_pub: float) -> str:
        if s_pub > self.tau_s:
            return ACTION_REFRESH
        return ACTION_KEEP


class AlwaysMoveTrigger(Trigger):
    name = "Always-move"
    use_rule_refresh = False
    respect_cooldown = False

    def decide(self, t: int, d_t: float, s_pub: float) -> str:
        return ACTION_MOVE


class AlwaysReclusterTrigger(Trigger):
    name = "Always-recluster"
    use_rule_refresh = False
    respect_cooldown = False

    def decide(self, t: int, d_t: float, s_pub: float) -> str:
        return ACTION_RECLUSTER


def coerce_action(action: str, env: ETStreamEnv, params: CMParams, trigger: Trigger) -> str:
    """Keep the action if the cooldown has not elapsed. A recluster is downgraded to a move only when that is explicitly allowed, never to a refresh."""
    if not trigger.respect_cooldown:
        return action
    if action == ACTION_RECLUSTER:
        if env.steps_since_recluster >= params.recluster_min_interval:
            return ACTION_RECLUSTER
        if (
            getattr(trigger, "downgrade_recluster_to_move", False)
            and trigger.wants_move()
            and env.steps_since_move >= params.min_steps_between_move
        ):
            return ACTION_MOVE
        return ACTION_KEEP
    if action == ACTION_MOVE:
        if env.steps_since_move >= params.min_steps_between_move:
            return ACTION_MOVE
        return ACTION_KEEP
    if action == ACTION_REFRESH:
        if env.steps_since_refresh >= params.refresh_min_interval:
            return ACTION_REFRESH
        return ACTION_KEEP
    return ACTION_KEEP


def _stale_clusters(env: ETStreamEnv, tau_s: float) -> List[int]:
    embeds = env._cluster_embeds()
    ids: List[int] = []
    nonempty: List[int] = []
    for m in range(env.M):
        members = [n for n in range(env.hp.N) if env.cluster_assign[n] == m]
        if not members:
            continue
        nonempty.append(m)
        stale = cluster_pub_staleness(members, env.embeds, embeds[m])
        if stale > tau_s:
            ids.append(m)
    return ids or nonempty


class ActionExecutor:
    name = "scripted"
    prompt_refresh = "frozen"

    def __init__(self, cm_params: CMParams):
        self.cm_params = cm_params
        self.action = ACTION_KEEP
        self.refresh_ids: List[int] = []

    def reset(self, cluster_assign: List[int]) -> None:
        return None

    def prepare(self, action: str, env: ETStreamEnv, tau_s: float) -> None:
        self.action = action
        self.refresh_ids = _stale_clusters(env, tau_s) if action == ACTION_REFRESH else []

    def update(self, t, embeds, cluster_assign, d_t, M):
        if self.action == ACTION_RECLUSTER:
            assign, elapsed = hac_recluster(embeds, M)
            return assign, PolicyStepInfo(
                recluster=True, recluster_time=elapsed, refresh_prompt=True,
            )
        if self.action == ACTION_MOVE:
            old = list(cluster_assign)
            assign, moved = stream_move(
                cluster_assign, embeds, M,
                tau_move=self.cm_params.tau_move,
                max_moves=self.cm_params.max_moves,
            )
            affected = sorted({
                a for a0, a1 in zip(old, assign) for a in (a0, a1) if a0 != a1
            }) if moved > 0 else None
            return assign, PolicyStepInfo(
                moved=moved, move_scan=True, refresh_prompt=moved > 0,
                affected_clusters=affected,
            )
        info = PolicyStepInfo()
        if self.action == ACTION_REFRESH and self.refresh_ids:
            info.refresh_clusters = list(self.refresh_ids)
            info.refresh_prompt = True
        return list(cluster_assign), info


def slot_signals(env: ETStreamEnv) -> tuple[float, float]:
    d_t = system_degradation(env.cluster_assign, env.embeds, env.M)
    s_pub = system_pub_staleness(
        env.cluster_assign, env.embeds, env._cluster_embeds(), env.M,
    )
    return d_t, s_pub


def run_scripted_episode(
    trigger: Trigger,
    ippo,
    seed: int,
    *,
    enable_cache: bool,
    T: int,
    p_drift: float,
    sigma: float,
    cm_params: CMParams | None = None,
    drift_mode: str = "independent",
    p_group: float = 0.1,
    sigma_v: float = 0.12,
    sigma_eps: float = 0.02,
    es_scale: float = 1.0,
    b_max: int | None = None,
    cache_thresholds: CacheThresholds | None = None,
    hp: HyperParams | None = None,
    record_trace: bool = False,
) -> tuple[EpisodeClusterStats, dict | None]:
    base = cm_params or CMParams()
    params = replace(base, enable_rule_refresh=bool(trigger.use_rule_refresh))
    env = make_stream_env(
        enable_cache=enable_cache,
        p_drift=p_drift,
        sigma=sigma,
        slot_steps=T,
        cm_params=params,
        cache_thresholds=cache_thresholds,
        drift_mode=drift_mode,
        p_group=p_group,
        sigma_v=sigma_v,
        sigma_eps=sigma_eps,
        es_scale=es_scale,
        b_max=b_max,
        hp=hp,
    )
    trigger.reset()
    executor = ActionExecutor(params)
    env.reset(seed)
    executor.reset(env.cluster_assign)
    ep = EpisodeClusterStats()
    d_hist: List[float] = []
    s_hist: List[float] = []
    n_steps = 0
    while True:
        action = coerce_action(trigger.decide(env.t, *slot_signals(env)), env, params, trigger)
        executor.prepare(action, env, getattr(trigger, "tau_s", 0.15))
        _, _, _, _, info, done = env.step_with_rule_policy(executor, ippo, False)
        ep.add_step(info, streamhac_step=False)
        if record_trace:
            d_hist.append(float(info.d_t))
            s_hist.append(float(info.s_pub_t))
        n_steps += 1
        if done:
            break
    ep.finalize(n_steps)
    ep.lookup_comparisons = float(env.cache.lookup_comparisons)
    ep.lookup_calls = float(env.cache.lookup_calls)
    trace = None
    if record_trace:
        trace = {"d": np.asarray(d_hist), "s": np.asarray(s_hist)}
    return ep, trace


def run_scripted_seeds(
    trigger_factory: Callable[[], Trigger],
    seeds: Sequence[int],
    *,
    enable_cache: bool,
    T: int,
    p_drift: float,
    sigma: float,
    ippo=None,
    record_trace: bool = False,
    **env_kwargs,
) -> tuple[list[EpisodeClusterStats], list[dict | None]]:
    if ippo is None:
        ippo = load_ippo()
        ippo.set_frozen(True)
    rows: list[EpisodeClusterStats] = []
    traces: list[dict | None] = []
    for seed in seeds:
        ep, trace = run_scripted_episode(
            trigger_factory(), ippo, int(seed),
            enable_cache=enable_cache, T=T, p_drift=p_drift, sigma=sigma,
            record_trace=record_trace, **env_kwargs,
        )
        rows.append(ep)
        traces.append(trace)
    return rows, traces


def run_et_seeds(cm_path: str, seeds: Sequence[int], **kwargs) -> list[EpisodeClusterStats]:
    return [run_et_eval(cm_path, seeds=[int(seed)], **kwargs) for seed in seeds]


def pick_best(candidates: Sequence[tuple[str, AggRow]]) -> tuple[str, AggRow]:
    """The validation objective is mean U_cum. Ties go to the candidate with fewer maintenance events."""
    return min(
        candidates,
        key=lambda item: (-item[1].cum_u, maintenance_events(item[1]), item[0]),
    )


def format_agg_table(rows: Sequence[AggRow]) -> str:
    header = (
        f"{'Policy':<22} {'U_cum':>16} {'QoS':>16} {'Move':>12} "
        f"{'Refresh':>12} {'Recluster':>12} {'Maint':>12} "
        f"{'hit':>8} {'fsafe':>8} {'Q':>8} {'D':>8} {'S':>8} {'T':>8}\n"
    )
    lines = [header, "-" * len(header) + "\n"]

    def pm(mean: float, std: float, prec: int = 2) -> str:
        return f"{mean:.{prec}f}±{std:.{prec}f}"

    for row in rows:
        lines.append(
            f"{row.name:<22} {pm(row.cum_u, row.cum_u_std):>16} "
            f"{pm(row.cum_u_qos, row.cum_u_qos_std):>16} "
            f"{pm(row.move_ops, row.move_ops_std, 1):>12} "
            f"{pm(row.refresh_ops, row.refresh_ops_std, 1):>12} "
            f"{pm(row.recluster_ops, row.recluster_ops_std, 1):>12} "
            f"{pm(row.maint_cost, row.maint_cost_std, 3):>12} "
            f"{row.hit_ratio:8.3f} {row.false_safe_rate:8.3f} "
            f"{row.q_mean:8.3f} {row.d_mean:8.3f} {row.s_pub_mean:8.3f} "
            f"{row.latency_mean:8.3f}\n"
        )
    return "".join(lines)
