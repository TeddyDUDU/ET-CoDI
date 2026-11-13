"""Clustering helpers: cosine similarity phi and hierarchical HAC."""

import numpy as np


def compute_phi(prompt_embed_n: np.ndarray, cluster_embed_m: np.ndarray) -> float:
    """Cosine similarity phi in [0, 1] between a user embedding and the cluster public prompt."""
    num = np.dot(prompt_embed_n, cluster_embed_m)
    den = np.linalg.norm(prompt_embed_n) * np.linalg.norm(cluster_embed_m) + 1e-8
    return float(np.clip(num / den, 0.0, 1.0))


def hac_clustering(prompt_embeds, M):
    """Hierarchical clustering: merge N user embeddings into M clusters (average-linkage similarity)."""
    N = len(prompt_embeds)
    sim_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            sim_matrix[i][j] = compute_phi(prompt_embeds[i], prompt_embeds[j])

    clusters = [[i] for i in range(N)]

    while len(clusters) > M:
        best_sim = -1.0
        best_pair = (0, 1)

        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                members_a = clusters[a]
                members_b = clusters[b]
                sim_sum = 0.0
                for i in members_a:
                    for j in members_b:
                        sim_sum += sim_matrix[i][j]
                avg_sim = sim_sum / (len(members_a) * len(members_b))

                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_pair = (a, b)

        a, b = best_pair
        clusters[a].extend(clusters[b])
        del clusters[b]

    cluster_assign = [0] * N
    for label, members in enumerate(clusters):
        for user in members:
            cluster_assign[user] = label

    return cluster_assign
