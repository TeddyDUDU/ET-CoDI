"""Background compute resources: a pre-generated per-slot trajectory. ET and the baselines share one seed channel."""

from typing import Tuple

import numpy as np

from core.params import HyperParams

# Resource RNG offset, aligned with init=seed and drift=seed+10000.
RES_SEED_OFFSET = 20000


def make_resource_trajectory(
    rng: np.random.RandomState,
    T: int,
    hp: HyperParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-generate length-T (v_max_ES[t], v_UE[t]). Every policy reads the same trajectory step by step."""
    v_es = rng.randint(hp.v_max_ES_range[0], hp.v_max_ES_range[1] + 1, size=T)
    v_ue = rng.randint(hp.v_UE_range[0], hp.v_UE_range[1] + 1, size=T)
    return v_es.astype(np.int64), v_ue.astype(np.int64)


def resource_rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed + RES_SEED_OFFSET)
