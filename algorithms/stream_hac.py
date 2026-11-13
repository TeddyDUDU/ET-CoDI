"""StreamHAC: stream_move and HAC recluster."""

from typing import List, Tuple

import numpy as np

from core.clustering import compute_phi, hac_clustering
from core.degradation import cluster_centroid
from core.params import RECLUSTER_TIME_FIXED


def _loo_centroid(members: List[int], embeds: List[np.ndarray], exclude: int) -> np.ndarray:
    others = [j for j in members if j != exclude]
    if not others:
        return embeds[exclude].copy()
    return cluster_centroid(others, embeds)


def stream_move(
    cluster_assign: List[int],
    embeds: List[np.ndarray],
    M: int,
    tau_move: float = 0.05,
    max_moves: int = 10,
) -> Tuple[List[int], int]:
    """Leave-one-out centroid comparison. At most max_moves users move per step."""
    N = len(embeds)
    new_assign = list(cluster_assign)
    cluster_members = [[] for _ in range(M)]
    for n in range(N):
        if new_assign[n] < M:
            cluster_members[new_assign[n]].append(n)

    candidates = []
    for n in range(N):
        m_cur = new_assign[n]
        if m_cur >= M:
            continue
        mem = cluster_members[m_cur]
        if not mem:
            continue
        sim_cur = compute_phi(embeds[n], _loo_centroid(mem, embeds, n))
        m_star, sim_star = m_cur, sim_cur
        for m in range(M):
            if m == m_cur or not cluster_members[m]:
                continue
            sim_m = compute_phi(
                embeds[n],
                _loo_centroid(cluster_members[m], embeds, n),
            )
            if sim_m > sim_star:
                m_star, sim_star = m, sim_m
        gain = sim_star - sim_cur
        if m_star != m_cur and gain > tau_move:
            candidates.append((gain, n, m_cur, m_star))

    candidates.sort(key=lambda x: x[0], reverse=True)
    moved = 0
    for _gain, n, m_cur, m_star in candidates[:max_moves]:
        if new_assign[n] != m_cur:
            continue
        cluster_members[m_cur].remove(n)
        cluster_members[m_star].append(n)
        new_assign[n] = m_star
        moved += 1

    return new_assign, moved


def hac_recluster(embeds: List[np.ndarray], M: int) -> Tuple[List[int], float]:
    """HAC recluster. The time cost is RECLUSTER_TIME_FIXED."""
    assign = hac_clustering(embeds, M)
    return assign, RECLUSTER_TIME_FIXED
