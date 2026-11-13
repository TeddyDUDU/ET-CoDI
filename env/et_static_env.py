"""Static-HIS single-step simulation. Supports drift, a cluster policy, and public-prompt refresh modes."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

PromptRefresh = Literal["frozen", "on_change", "every_step"]

import numpy as np
import torch

from core.clustering import compute_phi
from core.degradation import build_cluster_embeds, system_prompt_degradation
from core.hiqm import HIQM
from core.ippo import IPPOTrainer, normalize_obs
from core.params import HardwareParams, HyperParams, UtilityNormRefs
from core.seed import device
from core.utility import compute_step_metrics as _compute_step_metrics
from env.drift import apply_drift_step

EMBED_DIM = 768


def _as_resource_traj(values, T: int | None = None) -> np.ndarray:
    """Accept a scalar or a length-T array. A scalar is held constant for the episode (the degenerate case)."""
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim == 0:
        if T is None:
            return arr.reshape(1)
        return np.full(T, int(arr), dtype=np.int64)
    return arr


def init_embeds(rng: np.random.RandomState, N: int, M: int, intra_sigma: float = 0.08) -> List[np.ndarray]:
    centers = []
    for _ in range(M):
        c = rng.randn(EMBED_DIM)
        centers.append(c / (np.linalg.norm(c) + 1e-8))

    embeds = []
    for n in range(N):
        m = n % M
        e = centers[m] + rng.randn(EMBED_DIM) * intra_sigma
        embeds.append(e / (np.linalg.norm(e) + 1e-8))
    return embeds


def select_fixed_actions(hp: HyperParams) -> List[Tuple[float, float]]:
    x_n, y_n = 10.0, 5.0
    if x_n + y_n > hp.T_max:
        y_n = max(hp.y_min, hp.T_max - x_n)
    return [(x_n, y_n) for _ in range(hp.N)]


def select_ippo_actions(
    trainer: IPPOTrainer,
    embeds: List[np.ndarray],
    cluster_assign: List[int],
    cluster_embeds: List[np.ndarray],
    hp: HyperParams,
    v_max_ES: int,
    v_UE: int,
) -> List[Tuple[float, float]]:
    actions = []
    for n in range(hp.N):
        m = cluster_assign[n]
        phi = compute_phi(embeds[n], cluster_embeds[m])
        obs = normalize_obs(phi, v_max_ES, v_UE, hp)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            logits = trainer.actors[n](obs_t)
            act = logits.cpu().numpy()
        x_n = float(np.clip(round(act[0]), 0, hp.x_max))
        y_n = float(np.clip(round(act[1]), hp.y_min, hp.y_max))
        if x_n + y_n > hp.T_max:
            y_n = max(hp.y_min, hp.T_max - x_n)
        actions.append((x_n, y_n))
    return actions


def compute_step_metrics(
    embeds: List[np.ndarray],
    cluster_assign: List[int],
    cluster_embeds: List[np.ndarray],
    hp: HyperParams,
    hw: HardwareParams,
    hiqm: HIQM,
    actions: List[Tuple[float, float]],
    v_max_ES: int,
    v_UE: int,
    refs: UtilityNormRefs | None = None,
) -> Tuple[float, float, float]:
    if refs is None:
        refs = UtilityNormRefs()
    q, u, phi, _, _ = _compute_step_metrics(
        embeds, cluster_assign, cluster_embeds, hp, hw, hiqm, actions, v_max_ES, v_UE, refs,
    )
    return q, u, phi


@dataclass
class ClusterStats:
    move_ops: int = 0
    recluster_ops: int = 0


@dataclass
class EpisodeResult:
    Q: np.ndarray = field(default_factory=lambda: np.array([]))
    U: np.ndarray = field(default_factory=lambda: np.array([]))
    U_qos: np.ndarray = field(default_factory=lambda: np.array([]))
    phi: np.ndarray = field(default_factory=lambda: np.array([]))
    D_t_pub: np.ndarray = field(default_factory=lambda: np.array([]))
    cum_U: np.ndarray = field(default_factory=lambda: np.array([]))
    cum_U_qos: np.ndarray = field(default_factory=lambda: np.array([]))
    final_D_t: float = 0.0
    stats: ClusterStats = field(default_factory=ClusterStats)
    S_t_pub: np.ndarray = field(default_factory=lambda: np.array([]))
    hit_ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    refresh: np.ndarray = field(default_factory=lambda: np.array([]))


class ETStaticEnv:
    """Drift simulation. Supports a fixed or dynamic public prompt and a cluster-update policy."""

    def __init__(
        self,
        hp: HyperParams,
        hw: HardwareParams,
        hiqm: HIQM,
        initial_embeds: List[np.ndarray],
        cluster_assign: List[int],
        drift_masks: np.ndarray,
        drift_noise: np.ndarray,
        v_max_ES,
        v_UE,
        rng: np.random.RandomState,
        prompt_refresh: PromptRefresh = "frozen",
        ippo_trainer: Optional[IPPOTrainer] = None,
    ):
        self.hp = hp
        self.hw = hw
        self.hiqm = hiqm
        self.embeds = [e.copy() for e in initial_embeds]
        self.cluster_assign = list(cluster_assign)
        self.drift_masks = drift_masks
        self.drift_noise = drift_noise
        T = int(len(drift_masks))
        self.v_es_traj = _as_resource_traj(v_max_ES, T)
        self.v_ue_traj = _as_resource_traj(v_UE, T)
        if len(self.v_es_traj) != T or len(self.v_ue_traj) != T:
            raise ValueError(
                f"resource traj length must equal T={T}, "
                f"got v_ES={len(self.v_es_traj)}, v_UE={len(self.v_ue_traj)}"
            )
        self.v_max_ES = int(self.v_es_traj[0])
        self.v_UE = int(self.v_ue_traj[0])
        self.rng = rng
        self.prompt_refresh = prompt_refresh
        self.ippo_trainer = ippo_trainer
        self.frozen_cluster_embeds = build_cluster_embeds(
            self.cluster_assign, initial_embeds, hp.M, EMBED_DIM, rng
        )
        self.active_cluster_embeds = [e.copy() for e in self.frozen_cluster_embeds]

    def refresh_public_prompts(self):
        self.active_cluster_embeds = build_cluster_embeds(
            self.cluster_assign, self.embeds, self.hp.M, EMBED_DIM, self.rng,
        )

    def refresh_selected_prompts(self, cluster_ids: List[int]) -> None:
        """Refresh the active public prompt of affected clusters only (aligned with the ET stream path)."""
        from core.degradation import cluster_centroid

        if not cluster_ids:
            return
        for m in cluster_ids:
            if m < 0 or m >= len(self.active_cluster_embeds):
                continue
            members = [n for n in range(self.hp.N) if self.cluster_assign[n] == m]
            if members:
                self.active_cluster_embeds[m] = cluster_centroid(members, self.embeds)

    def service_cluster_embeds(self) -> List[np.ndarray]:
        """Public-prompt vector e_m^pub, used to compute HIQM phi."""
        if self.prompt_refresh == "frozen":
            return self.frozen_cluster_embeds
        if self.prompt_refresh == "on_change":
            return self.active_cluster_embeds
        return build_cluster_embeds(
            self.cluster_assign, self.embeds, self.hp.M, EMBED_DIM, self.rng,
        )

    def service_prompt_degradation(self) -> float:
        """D_t^pub: mismatch between user embeddings and the prompt currently served (= 1 - mean phi)."""
        return system_prompt_degradation(
            self.cluster_assign,
            self.embeds,
            self.service_cluster_embeds(),
            self.hp.M,
        )

    def step_metrics(self) -> Tuple[float, float, float]:
        cluster_embeds = self.service_cluster_embeds()
        if self.ippo_trainer is not None:
            actions = select_ippo_actions(
                self.ippo_trainer, self.embeds, self.cluster_assign, cluster_embeds,
                self.hp, self.v_max_ES, self.v_UE,
            )
        else:
            actions = select_fixed_actions(self.hp)
        q, u, phi = compute_step_metrics(
            self.embeds, self.cluster_assign, cluster_embeds,
            self.hp, self.hw, self.hiqm, actions, self.v_max_ES, self.v_UE,
        )
        return q, u, phi

    def apply_drift(self, t: int):
        self.embeds = apply_drift_step(self.embeds, self.drift_masks[t], self.drift_noise[t])
        self.v_max_ES = int(self.v_es_traj[t])
        self.v_UE = int(self.v_ue_traj[t])
