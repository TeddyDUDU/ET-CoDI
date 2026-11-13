"""HIS paper utility: Eq. (23) network utility and Eq. (33) IPPO reward."""

from typing import List, Tuple

import numpy as np

from core.clustering import compute_phi
from core.hiqm import HIQM, _map_t, map_embed_phi_to_hiqm
from core.metrics import compute_latency_and_energy
from core.params import (
    CacheOverheadRefs,
    ClusterOverheadRefs,
    HardwareParams,
    HyperParams,
    UtilityNormRefs,
)


def effective_phi(phi_sim: float, hp: HyperParams) -> float:
    """Map embed phi_sim to the paper HIQM lookup phi. Optional phi^exp, used as an ablation only when hiqm_phi_map is none."""
    if hp.hiqm_phi_map != "none":
        phi = map_embed_phi_to_hiqm(phi_sim, hp)
    else:
        phi = float(phi_sim)
    if hp.phi_q_exponent != 1.0:
        phi = float(phi ** hp.phi_q_exponent)
    return phi


def hiqm_quality(hiqm: HIQM, x: float, phi_sim: float, hp: HyperParams) -> float:
    phi_hiqm = effective_phi(phi_sim, hp)
    q = hiqm(x, phi_hiqm)
    blend = hp.hiqm_drift_q_blend
    if blend > 0.0 and hp.hiqm_phi_map != "none":
        t = _map_t(phi_sim, hp)
        q *= (1.0 - blend) + blend * t
    return q


def es_energy_share(
    x_n: float,
    y_n: float,
    x_sum: float,
    y_sum_cluster: float,
    E_m_C: float,
    E_m_PE: float,
) -> float:
    return (x_n / x_sum) * E_m_C + (y_n / y_sum_cluster) * E_m_PE


def compute_cache_overhead(hits: int, misses: int, refs: CacheOverheadRefs) -> float:
    """Per-slot cache cost: a small cost on a hit, a cold-start penalty on a miss."""
    return refs.lambda_reuse * hits + refs.lambda_miss * misses


def compute_cluster_overhead(
    recluster: bool,
    recluster_time: float,
    moved: int,
    refs: ClusterOverheadRefs,
    move_triggered: bool = False,
    rule_refresh: bool = False,
) -> float:
    """Cluster maintenance cost. A positive value is subtracted from U_t."""
    cost = 0.0
    if recluster:
        cost += refs.lambda_f + refs.lambda_t * recluster_time
    if move_triggered:
        cost += refs.lambda_move_op
    if refs.lambda_m != 0.0:
        cost += refs.lambda_m * moved
    if rule_refresh:
        cost += refs.lambda_refresh
    return cost


ALWAYS_MOVE_POLICY = "Always-move"
NET_UTILITY_LAMBDAS = (0.0, 0.001, 0.01, 0.05, 0.1)


def policy_cluster_overhead(
    recluster: bool,
    recluster_time: float,
    moved: int,
    refs: ClusterOverheadRefs,
    policy_name: str,
    move_scan: bool = False,
) -> float:
    """A slot that runs a migration scan pays one move cost, independent of how many users move."""
    return compute_cluster_overhead(
        recluster, recluster_time, moved, refs,
        move_triggered=(move_scan or policy_name == ALWAYS_MOVE_POLICY),
    )


def policy_step_ops(
    policy_name: str, recluster: bool, move_scan: bool = False,
) -> tuple[int, int]:
    if move_scan or policy_name == ALWAYS_MOVE_POLICY:
        return 1, int(recluster)
    return 0, int(recluster)


def net_utility_qos(cum_u_qos: float, move_ops: float, recluster_ops: float, lam: float) -> float:
    """Net utility = gross QoS cumulative utility - lambda * (move + recluster) counts."""
    return cum_u_qos - lam * (move_ops + recluster_ops)


def format_overhead_refs(refs: ClusterOverheadRefs) -> str:
    return (
        f"lambda_f={refs.lambda_f}, lambda_t={refs.lambda_t}, "
        f"lambda_move_op={refs.lambda_move_op}, lambda_m={refs.lambda_m}, "
        f"lambda_refresh={refs.lambda_refresh}"
    )


def compute_agent_reward(
    Q_n: float,
    T_n: float,
    E_n: float,
    es_share: float,
    hp: HyperParams,
) -> float:
    """Paper Eq. (33) per-user reward. Training scales it by r_w_* to stabilize IPPO."""
    return (
        hp.r_w_Q * Q_n
        - hp.r_w_T * T_n
        - hp.r_w_E * E_n
        - hp.r_w_E * es_share
    )


def normalize_qte(
    Q: float,
    T: float,
    E: float,
    refs: UtilityNormRefs,
) -> Tuple[float, float, float]:
    """Fig. 18 normalization: Q is the dimensionless HIQM value. T/E are scaled to a [0, 1] cost by the reference upper bound."""
    t_cost = min(T / refs.T_ref, 1.0)
    e_cost = min(E / refs.E_ref, 1.0)
    return Q, t_cost, e_cost


def compute_network_utility(
    Q: float,
    T: float,
    E: float,
    hp: HyperParams,
    refs: UtilityNormRefs,
) -> float:
    """
    Paper Eq. (23) network utility (about 0.2-0.35).

    T/E are first normalized by refs to a [0, 1] cost, then combined with Q as (1 - cost).
    """
    q, t_cost, e_cost = normalize_qte(Q, T, E, refs)
    t_score = 1.0 - t_cost
    e_score = 1.0 - e_cost
    return hp.w_Q * q + hp.w_T * t_score + hp.w_E * e_score


def decompose_network_utility(
    Q: float,
    T: float,
    E: float,
    hp: HyperParams,
    refs: UtilityNormRefs,
) -> Tuple[float, float, float]:
    """Split U into the quality term and the latency/energy term."""
    q, t_cost, e_cost = normalize_qte(Q, T, E, refs)
    u_q = hp.w_Q * q
    u_te = hp.w_T * (1.0 - t_cost) + hp.w_E * (1.0 - e_cost)
    return u_q, u_te, u_q + u_te


def compute_step_metrics(
    embeds: List,
    cluster_assign: List[int],
    cluster_embeds: List,
    hp: HyperParams,
    hw: HardwareParams,
    hiqm: HIQM,
    actions: List[Tuple[float, float]],
    v_max_ES: int,
    v_UE: int,
    refs: UtilityNormRefs,
) -> Tuple[float, float, float, float, float]:
    """One slot: return Q_mean, U_mean, phi_mean, U_Q_mean, U_TE_mean."""
    M_eff = len(cluster_embeds) if cluster_embeds else hp.M
    alpha1 = np.ones(M_eff)
    alpha2 = np.ones(M_eff)
    denom = np.sum(np.sqrt(alpha1 + alpha2))
    v_m_ES = v_max_ES * np.sqrt(alpha1 + alpha2) / (denom + 1e-8)

    q_list, u_list, phi_list = [], [], []
    u_q_list, u_te_list = [], []
    t_mean_acc, e_mean_acc = [], []

    for n in range(hp.N):
        x_n, y_n = actions[n]
        z_n = hp.T_max - x_n - y_n
        m = cluster_assign[n]

        x_m_max = max(actions[j][0] for j in range(hp.N) if cluster_assign[j] == m)
        y_sum_cluster = sum(actions[j][1] for j in range(hp.N) if cluster_assign[j] == m) + 1e-8
        x_sum = sum(actions[j][0] for j in range(hp.N) if cluster_assign[j] == m) + 1e-8

        T_n, E_n, E_m_C, E_m_PE = compute_latency_and_energy(
            x_n, y_n, z_n, x_m_max, y_sum_cluster, v_UE, v_m_ES[m], hp, hw
        )
        phi = compute_phi(embeds[n], cluster_embeds[m])
        Q_n = hiqm_quality(hiqm, x_n, phi, hp)
        es_share = es_energy_share(x_n, y_n, x_sum, y_sum_cluster, E_m_C, E_m_PE)
        E_total = E_n + es_share

        u_n = compute_network_utility(Q_n, T_n, E_total, hp, refs)
        u_q, u_te, _ = decompose_network_utility(Q_n, T_n, E_total, hp, refs)

        q_list.append(Q_n)
        u_list.append(u_n)
        phi_list.append(phi)
        u_q_list.append(u_q)
        u_te_list.append(u_te)
        t_mean_acc.append(T_n)
        e_mean_acc.append(E_total)

    return (
        float(sum(q_list) / len(q_list)),
        float(sum(u_list) / len(u_list)),
        float(sum(phi_list) / len(phi_list)),
        float(sum(u_q_list) / len(u_q_list)),
        float(sum(u_te_list) / len(u_te_list)),
    )


def compute_step_metrics_with_cache(
    embeds: List,
    cluster_assign: List[int],
    cluster_embeds: List,
    hp: HyperParams,
    hw: HardwareParams,
    hiqm: HIQM,
    actions: List[Tuple[float, float]],
    v_max_ES: int,
    v_UE: int,
    refs: UtilityNormRefs,
    x_common_eff: List[float] | None = None,
    return_details: bool = False,
) -> Tuple[float, ...]:
    """One slot with cache-shortened common inference. x_common_eff[m] is the effective common-step count of cluster m."""
    M_eff = len(cluster_embeds) if cluster_embeds else hp.M
    alpha1 = np.ones(M_eff)
    alpha2 = np.ones(M_eff)
    denom = np.sum(np.sqrt(alpha1 + alpha2))
    v_m_ES = v_max_ES * np.sqrt(alpha1 + alpha2) / (denom + 1e-8)

    q_list, u_list, phi_list = [], [], []
    u_q_list, u_te_list = [], []
    latency_list, energy_list = [], []

    for n in range(hp.N):
        x_n, y_n = actions[n]
        z_n = hp.T_max - x_n - y_n
        m = cluster_assign[n]

        x_m_max = max(actions[j][0] for j in range(hp.N) if cluster_assign[j] == m)
        x_common = x_m_max if x_common_eff is None else x_common_eff[m]
        y_sum_cluster = sum(actions[j][1] for j in range(hp.N) if cluster_assign[j] == m) + 1e-8
        x_sum = sum(actions[j][0] for j in range(hp.N) if cluster_assign[j] == m) + 1e-8

        T_n, E_n, E_m_C, E_m_PE = compute_latency_and_energy(
            x_n, y_n, z_n, x_m_max, y_sum_cluster, v_UE, v_m_ES[m], hp, hw,
            x_m_common=x_common,
        )
        phi = compute_phi(embeds[n], cluster_embeds[m])
        Q_n = hiqm_quality(hiqm, x_n, phi, hp)
        es_share = es_energy_share(x_n, y_n, x_sum, y_sum_cluster, E_m_C, E_m_PE)
        E_total = E_n + es_share

        u_n = compute_network_utility(Q_n, T_n, E_total, hp, refs)
        u_q, u_te, _ = decompose_network_utility(Q_n, T_n, E_total, hp, refs)

        q_list.append(Q_n)
        u_list.append(u_n)
        phi_list.append(phi)
        u_q_list.append(u_q)
        u_te_list.append(u_te)
        latency_list.append(T_n)
        energy_list.append(E_total)

    result = (
        float(sum(q_list) / len(q_list)),
        float(sum(u_list) / len(u_list)),
        float(sum(phi_list) / len(phi_list)),
        float(sum(u_q_list) / len(u_q_list)),
        float(sum(u_te_list) / len(u_te_list)),
    )
    if return_details:
        return result + (
            float(sum(latency_list) / len(latency_list)),
            float(sum(energy_list) / len(energy_list)),
        )
    return result
