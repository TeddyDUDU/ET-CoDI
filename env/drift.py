"""Gradual embedding drift: a pre-generated trajectory plus a per-step L2-normalized perturbation."""

from typing import List, Tuple

import numpy as np


def normalize_embed(e: np.ndarray) -> np.ndarray:
    return e / (np.linalg.norm(e) + 1e-8)


def make_drift_trajectory(
    rng: np.random.RandomState,
    T: int,
    N: int,
    dim: int,
    p_drift: float,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-generate T steps of drift masks and noise. Every policy shares the same trajectory."""
    masks = rng.rand(T, N) < p_drift
    noise = rng.randn(T, N, dim) * sigma
    return masks, noise


def apply_drift_step(
    embeds: List[np.ndarray],
    drift_mask: np.ndarray,
    drift_noise: np.ndarray,
) -> List[np.ndarray]:
    """One-slot gradual drift: e_{n,t+1} = normalize(e_{n,t} + epsilon)."""
    new_embeds = []
    for n, e in enumerate(embeds):
        if drift_mask[n]:
            e = normalize_embed(e + drift_noise[n])
        new_embeds.append(e.copy())
    return new_embeds


def make_comove_trajectory(
    rng: np.random.RandomState,
    T: int,
    N: int,
    dim: int,
    groups: np.ndarray,
    p_group: float,
    sigma_v: float,
    sigma_eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Co-movement trajectory: a group vector v_g appears with probability p_group, and individual noise is added every step.

    groups are fixed at t=0 and do not follow online cluster labels. Returns (shared[T,G,dim], eps[T,N,dim]).
    """
    labels = np.asarray(groups, dtype=np.int64)
    gcount = int(labels.max()) + 1 if labels.size else 1
    active = rng.rand(T, gcount) < p_group
    shared = rng.randn(T, gcount, dim) * float(sigma_v)
    shared *= active[..., None]
    eps = rng.randn(T, N, dim) * float(sigma_eps)
    return shared, eps


def apply_comove_step(
    embeds: List[np.ndarray],
    groups: np.ndarray,
    shared_t: np.ndarray,
    eps_t: np.ndarray,
) -> List[np.ndarray]:
    """e_{n,t+1} = normalize(e_{n,t} + v_{g(n),t} + epsilon_{n,t})。"""
    new_embeds = []
    for n, e in enumerate(embeds):
        new_embeds.append(normalize_embed(e + shared_t[int(groups[n])] + eps_t[n]))
    return new_embeds
