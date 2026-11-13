"""Latency and energy model (paper Table II parameters)."""

import math
from typing import Tuple

from core.params import HardwareParams, HyperParams


def compute_latency_and_energy(
    x_n: float, y_n: float, z_n: float,
    x_m_max: float,
    y_sum_cluster: float,
    v_n_UE: float, v_m_ES: float,
    hp: HyperParams, hw: HardwareParams,
    b_n: float = 20e6,
    p_n: float = 1.0,
    g_n: float = 1.0,
    sigma2_0: float = 7.96e-14,
    rho_ES: float = 0.6,
    rho_UE: float = 0.7,
    x_m_common: float | None = None,
) -> Tuple[float, float, float, float]:
    """Per-user latency T_n, energy E_n, and cluster-level common/private inference energy."""
    f_m_ES = v_m_ES * hw.F_ES * hw.D_ES * rho_ES
    x_common = x_m_max if x_m_common is None else x_m_common
    T_m_C = (x_common * hw.xi_ES) / f_m_ES
    P_m_ES = hw.kappa_ES * (hw.mu1_ES * v_m_ES + hw.mu2_ES) * (hw.F_ES ** 3)
    E_m_C_total = P_m_ES * T_m_C

    T_n_PE = (y_n * hw.xi_ES) / f_m_ES
    E_m_PE_total = P_m_ES * ((y_sum_cluster * hw.xi_ES) / f_m_ES)

    sinr = (p_n * g_n) / (sigma2_0 * b_n)
    r_n = b_n * math.log2(1.0 + sinr)
    T_n_T = (hp.n_h * hp.n_w * hp.n_c * hp.rho) / (r_n * hp.tau)
    E_n_T = p_n * T_n_T

    f_n_UE = v_n_UE * hw.F_UE * hw.D_UE * rho_UE
    T_n_PU = (z_n * hw.xi_UE) / f_n_UE
    P_n_UE = hw.kappa_UE * (hw.mu1_UE * v_n_UE + hw.mu2_UE) * (hw.F_UE ** 3)
    E_n_PU = P_n_UE * T_n_PU

    T_n = T_m_C + T_n_PE + T_n_T + T_n_PU
    E_n = E_n_T + E_n_PU

    return T_n, E_n, E_m_C_total, E_m_PE_total
