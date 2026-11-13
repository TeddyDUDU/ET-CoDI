"""Cluster degradation D_m, public-state staleness S_m^pub, system degradation D_t, and frozen public-prompt vectors."""

from typing import List

import numpy as np

from core.clustering import compute_phi


def cluster_centroid(members: List[int], embeds: List[np.ndarray]) -> np.ndarray:
    """L2-normalized centroid of the embeddings in a cluster."""
    mu = np.mean([embeds[n] for n in members], axis=0)
    return mu / (np.linalg.norm(mu) + 1e-8)


def cluster_degradation(members: List[int], embeds: List[np.ndarray]) -> float:
    """Per-cluster geometric degradation D_m^geom = 1 - mean(cosine similarity to the current centroid mu)."""
    if not members:
        return 0.0
    mu = cluster_centroid(members, embeds)
    sims = [compute_phi(embeds[n], mu) for n in members]
    return 1.0 - float(np.mean(sims))


def cluster_prompt_degradation(
    members: List[int],
    embeds: List[np.ndarray],
    pub_embed: np.ndarray,
) -> float:
    """Per-cluster service degradation D_m^pub = 1 - mean(cosine similarity to the public prompt actually in use)."""
    if not members:
        return 0.0
    sims = [compute_phi(embeds[n], pub_embed) for n in members]
    return 1.0 - float(np.mean(sims))


def cluster_pub_staleness(
    members: List[int],
    embeds: List[np.ndarray],
    pub_embed: np.ndarray,
) -> float:
    """Per-cluster public-state staleness S_m^pub = 1 - Sim(mu_{m,t}, e_{m,tau}^pub).

    Unlike D_m^pub, this compares only the centroid with the prompt in use and ignores intra-cluster spread.
    It rises under co-movement and does not rise under dispersion, so it can gate refresh by itself.
    """
    if not members:
        return 0.0
    mu = cluster_centroid(members, embeds)
    return 1.0 - compute_phi(mu, pub_embed)


def system_pub_staleness(
    cluster_assign: List[int],
    embeds: List[np.ndarray],
    cluster_embeds: List[np.ndarray],
    M: int,
) -> float:
    """System public-state staleness S_t^pub: size-weighted average of S_m^pub."""
    N = len(embeds)
    s_pub = 0.0
    for m in range(M):
        members = [n for n in range(N) if cluster_assign[n] == m]
        if not members:
            continue
        s_pub += (len(members) / N) * cluster_pub_staleness(members, embeds, cluster_embeds[m])
    return s_pub


def system_degradation(cluster_assign: List[int], embeds: List[np.ndarray], M: int) -> float:
    """System geometric degradation D_t^geom: intra-cluster mismatch of the partition against the current embeddings. Used to trigger clustering."""
    N = len(embeds)
    d_t = 0.0
    for m in range(M):
        members = [n for n in range(N) if cluster_assign[n] == m]
        if not members:
            continue
        d_m = cluster_degradation(members, embeds)
        d_t += (len(members) / N) * d_m
    return d_t


def system_prompt_degradation(
    cluster_assign: List[int],
    embeds: List[np.ndarray],
    cluster_embeds: List[np.ndarray],
    M: int,
) -> float:
    """System service degradation D_t^pub = 1 - mean(phi_n).

    cluster_embeds are the public prompts. Static-HIS uses the vectors frozen at t=0; a dynamic policy uses the current active prompt.
    """
    N = len(embeds)
    d_pub = 0.0
    for m in range(M):
        members = [n for n in range(N) if cluster_assign[n] == m]
        if not members:
            continue
        d_m = cluster_prompt_degradation(members, embeds, cluster_embeds[m])
        d_pub += (len(members) / N) * d_m
    return d_pub


def build_cluster_embeds(
    cluster_assign: List[int],
    embeds: List[np.ndarray],
    M: int,
    dim: int,
    rng: np.random.RandomState,
) -> List[np.ndarray]:
    """Build a public-prompt embedding for each cluster. An empty cluster gets a random unit vector."""
    N = len(embeds)
    cluster_embeds = []
    for m in range(M):
        members = [n for n in range(N) if cluster_assign[n] == m]
        if members:
            cluster_embeds.append(cluster_centroid(members, embeds))
        else:
            empty = rng.randn(dim)
            cluster_embeds.append(empty / (np.linalg.norm(empty) + 1e-8))
    return cluster_embeds
