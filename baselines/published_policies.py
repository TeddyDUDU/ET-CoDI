"""Published baselines: Static-HIS / IIS."""

from dataclasses import dataclass
from typing import List, Literal, Protocol, Tuple

import numpy as np

PromptRefresh = Literal["frozen", "on_change", "every_step"]


@dataclass
class PolicyStepInfo:
    recluster: bool = False
    moved: int = 0
    recluster_time: float = 0.0
    refresh_prompt: bool = False
    schedule: bool = False
    move_scan: bool = False


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


class IISPolicy:
    """Independent Inference Scheme: no shared inference phase."""

    name = "IIS"
    prompt_refresh: PromptRefresh = "frozen"

    def reset(self, cluster_assign: List[int]) -> None:
        pass

    def update(self, t, embeds, cluster_assign, d_t, M):
        return list(cluster_assign), PolicyStepInfo()
