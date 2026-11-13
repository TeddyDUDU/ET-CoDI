"""Cluster baselines: Static / Always-move / Always-recluster."""

from dataclasses import dataclass
from typing import List, Literal, Protocol, Tuple

import numpy as np

from algorithms.stream_hac import hac_recluster, stream_move

PromptRefresh = Literal["frozen", "on_change", "every_step"]

BASELINE_ORDER = ("Static-HIS", "Always-move", "Always-recluster")


@dataclass
class PolicyStepInfo:
    recluster: bool = False
    moved: int = 0
    recluster_time: float = 0.0
    refresh_prompt: bool = False
    move_scan: bool = False
    affected_clusters: List[int] | None = None
    refresh_clusters: List[int] | None = None


def _affected_clusters(old_assign: List[int], new_assign: List[int]) -> List[int]:
    return sorted({
        a for a0, a1 in zip(old_assign, new_assign) for a in (a0, a1) if a0 != a1
    })


class ClusterPolicy(Protocol):
    name: str
    prompt_refresh: PromptRefresh

    def reset(self, cluster_assign: List[int]) -> None: ...

    def update(
        self,
        t: int,
        embeds: List[np.ndarray],
        cluster_assign: List[int],
        d_t: float,
        M: int,
    ) -> Tuple[List[int], PolicyStepInfo]: ...


class StaticHISPolicy:
    name = "Static-HIS"
    prompt_refresh: PromptRefresh = "frozen"

    def reset(self, cluster_assign: List[int]) -> None:
        pass

    def update(self, t, embeds, cluster_assign, d_t, M):
        return list(cluster_assign), PolicyStepInfo()


class AlwaysReclusterPolicy:
    name = "Always-recluster"
    prompt_refresh: PromptRefresh = "on_change"

    def reset(self, cluster_assign: List[int]) -> None:
        pass

    def update(self, t, embeds, cluster_assign, d_t, M):
        assign, elapsed = hac_recluster(embeds, M)
        return assign, PolicyStepInfo(
            recluster=True, recluster_time=elapsed, refresh_prompt=True,
        )


class AlwaysMovePolicy:
    """Run one stream_move scan every step. Always-move ignores D_t and recluster thresholds."""

    name = "Always-move"
    prompt_refresh: PromptRefresh = "on_change"

    def __init__(self, tau_move: float = 0.05, max_moves: int = 10):
        self.tau_move = tau_move
        self.max_moves = max_moves

    def reset(self, cluster_assign: List[int]) -> None:
        pass

    def update(self, t, embeds, cluster_assign, d_t, M):
        old_assign = list(cluster_assign)
        assign, moved = stream_move(
            cluster_assign, embeds, M,
            tau_move=self.tau_move, max_moves=self.max_moves,
        )
        affected = _affected_clusters(old_assign, assign) if moved > 0 else None
        return assign, PolicyStepInfo(
            moved=moved,
            refresh_prompt=moved > 0,
            move_scan=True,
            affected_clusters=affected,
        )


def build_baseline_policies() -> List[ClusterPolicy]:
    return [
        StaticHISPolicy(),
        AlwaysMovePolicy(),
        AlwaysReclusterPolicy(),
    ]
