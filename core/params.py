from dataclasses import dataclass
from typing import Tuple

# Fixed HAC recluster time T_recluster (seconds), charged in C_t^upd.
RECLUSTER_TIME_FIXED: float = 0.025


@dataclass
class HyperParams:
    # DRL hyperparameters (Table II).
    actor_lr: float = 0.005
    critic_lr: float = 0.005
    clip_eps: float = 0.2
    critic_clip_eps: float = 0.2
    entropy_coef: float = 0.002
    hidden_layers: Tuple[int, int, int] = (256, 256, 256)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    train_epochs: int = 10
    episodes: int = 500
    decision_points: int = 50

    # Environment parameters (Table II).
    N: int = 50
    M: int = 3
    x_max: int = 16
    y_min: int = 0
    y_max: int = 11
    T_max: int = 28

    # Available ES / UE SM ranges (Table II).
    v_max_ES_range: Tuple[int, int] = (1026, 1674)
    v_UE_range: Tuple[int, int] = (64, 84)

    # Data and transmission parameters (Table II).
    n_h: int = 128
    n_w: int = 128
    n_c: int = 16
    rho: int = 16
    tau: int = 2

    # Paper Eq. (23) weights varpi_1, varpi_2, varpi_3 (Fig. 18 defaults 0.5, 0.25, 0.25).
    w_Q: float = 0.5
    w_T: float = 0.25
    w_E: float = 0.25

    # IPPO training reward scales r_w_Q, r_w_T, r_w_E.
    r_w_Q: float = 1.0
    r_w_T: float = 0.1
    r_w_E: float = 0.01

    # Utility sensitivity (optional ablation).
    phi_q_exponent: float = 1.0

    # HIQM: paper sigmoid bins 0.2-0.7. Embed phi is power-mapped, then looked up with low-end extrapolation.
    hiqm_phi_mode: str = "legacy"
    hiqm_phi_ref: float = 0.7
    hiqm_phi_map: str = "power"  # none | linear | power
    hiqm_phi_min: float = 0.05   # Lower end of the HIQM table; sigmoid parameters are extrapolated below it.
    hiqm_phi_max: float = 0.7
    hiqm_map_phi_lo: float = 0.015   # Lower embed-phi map bound (drift steady state).
    hiqm_map_phi_hi: float = 0.468   # Upper embed-phi map bound (no drift).
    hiqm_map_gamma: float = 2.0        # Power-map exponent.
    hiqm_drift_q_blend: float = 0.18   # Blend of Q and the map progress t.

    # Observation normalization references (linear normalization).
    v_max_ES_ref: float = 1674.0
    v_UE_ref: float = 84.0


@dataclass
class UtilityNormRefs:
    """T/E normalization references in Eq. (23), on the per-user mean scale."""

    T_ref: float = 2.0
    E_ref: float = 7000.0


@dataclass
class CacheThresholds:
    """Thresholds for admitting a reusable public inference state."""

    tau_p: float = 0.9
    tau_mu: float = 0.85
    delta_x: float = 2.0


@dataclass
class CacheOverheadRefs:
    """Cache cost C_t^cache = lambda_miss * n_miss + lambda_reuse * n_hit, counted per cluster."""

    lambda_miss: float = 0.002
    lambda_reuse: float = 0.0002


@dataclass
class CMParams:
    """Upper-level MC-PPO: keep / stream_move / recluster.

    Evaluation uses net u_t. The training reward is same-slot degradation-reduction shaping:
    r^c = u_t + gamma * Phi(s') - Phi(s), Phi = -lambda_D * D (geometric D only).
    Deterministic prompt refresh is gated by tau_stale / refresh_min_interval and a centroid-improvement check.
    It runs before the CM decision, so it does not depend on that slot's structural action.
    """

    gamma: float = 0.97
    pbrs_lambda: float = 0.25
    critic_clip_eps: float = 0.0  # 0 = do not clip the critic update.
    actor_lr: float = 0.003
    critic_lr: float = 0.003
    eval_temperature: float = 0.0
    entropy_coef_start: float = 0.08
    entropy_coef_end: float = 0.02
    adv_center: bool = True
    move_explore_floor: float = 0.08
    move_explore_floor_end: float = 0.0
    move_explore_floor_end_ep: int = 100
    recluster_explore_floor: float = 0.03
    recluster_explore_floor_end: float = 0.0
    recluster_loss_weight: float = 1.0
    update_every_ep: int = 5
    actor_warmup_ep: int = 5
    move_loss_weight: float = 1.0
    recluster_min_interval: int = 5
    min_steps_between_move: int = 1
    refresh_min_interval: int = 5
    tau_move: float = 0.05
    tau_stale: float = 0.15  # tau_s: lower threshold on S^pub.
    max_moves: int = 10
    enable_rule_refresh: bool = True

    def entropy_coef_at(self, ep: int, curriculum_total: int) -> float:
        t = min(1.0, ep / max(curriculum_total - 1, 1))
        return self.entropy_coef_start + t * (
            self.entropy_coef_end - self.entropy_coef_start
        )

    def move_explore_floor_at(self, phase_ep: int) -> float:
        if self.move_explore_floor <= 0:
            return 0.0
        end = max(self.move_explore_floor_end_ep, 1)
        t = min(1.0, phase_ep / end)
        return self.move_explore_floor + t * (
            self.move_explore_floor_end - self.move_explore_floor
        )

    def recluster_explore_floor_at(self, phase_ep: int) -> float:
        if self.recluster_explore_floor <= 0:
            return 0.0
        end = max(self.move_explore_floor_end_ep, 1)
        t = min(1.0, phase_ep / end)
        return self.recluster_explore_floor + t * (
            self.recluster_explore_floor_end - self.recluster_explore_floor
        )


@dataclass
class ClusterOverheadRefs:
    """
    Cluster-update cost C_t^upd, subtracted from the per-step network utility:

    C_t^upd = lambda_f * I(recluster) + lambda_t * recluster_time
            + lambda_move_op * I(move event) + lambda_m * moved
            + lambda_refresh * I(rule refresh)
    """

    lambda_f: float = 0.01
    lambda_t: float = 0.1
    lambda_move_op: float = 0.01
    lambda_m: float = 0.00005
    lambda_refresh: float = 0.002


@dataclass
class HardwareParams:
    xi_ES: float = 12e12
    F_ES: float = 1410e6
    D_ES: float = 2048
    kappa_ES: float = 1e-27
    mu1_ES: float = 4
    mu2_ES: float = 135

    xi_UE: float = 12e12
    F_UE: float = 1860e6
    D_UE: float = 1024
    kappa_UE: float = 1e-27
    mu1_UE: float = 4
    mu2_UE: float = 135
