"""HIQM: sigmoid quality model (similarity bins 0.2-0.7).

Embed cosine phi_sim is power-mapped into [phi_min, phi_max]. Below phi_min the paper curve is extrapolated.
"""

from __future__ import annotations

import numpy as np

from core.params import HyperParams

# Sigmoid parameters (similarity bins 0.2-0.7, fitted on CLIP).
PAPER_HIQM_PARAMS: dict[float, tuple[float, float, float, float]] = {
    0.2: (0.268386932, 0.271233986, 16.362286268, 0.057222735),
    0.3: (0.254154904, 0.268183574, 16.377773504, 0.070610329),
    0.4: (0.216310286, 0.258260065, 16.443448425, 0.103935081),
    0.5: (0.184652628, 0.236055261, 16.526390454, 0.136547140),
    0.6: (0.151463116, 0.216391715, 16.548755467, 0.171208139),
    0.7: (0.129200920, 0.193314464, 16.496653718, 0.195974136),
}

PAPER_PHI_LEVELS = tuple(PAPER_HIQM_PARAMS.keys())
PAPER_PHI_FLOOR = PAPER_PHI_LEVELS[0]


def _map_t(phi_sim: float, hp: HyperParams) -> float:
    lo, hi = hp.hiqm_map_phi_lo, hp.hiqm_map_phi_hi
    if hi <= lo:
        return 0.0
    return float(np.clip((phi_sim - lo) / (hi - lo), 0.0, 1.0))


def map_embed_phi_to_hiqm(phi_sim: float, hp: HyperParams) -> float:
    """Map embed phi_sim to the HIQM lookup phi.

    power (default): phi_hiqm = phi_min + (phi_max - phi_min) * t^gamma
      t = (phi_sim - lo) / (hi - lo). gamma > 1 pulls the drift region toward phi_min (a steeper Q curve).
    linear: the special case gamma = 1.
    """
    mode = hp.hiqm_phi_map
    if mode == "none":
        return float(phi_sim)

    phi_sim = float(np.clip(phi_sim, 0.0, 1.0))
    out_lo, out_hi = hp.hiqm_phi_min, hp.hiqm_phi_max
    t = _map_t(phi_sim, hp)

    if mode == "linear":
        return out_lo + (out_hi - out_lo) * t
    if mode == "power":
        gamma = max(hp.hiqm_map_gamma, 1e-6)
        return out_lo + (out_hi - out_lo) * (t ** gamma)

    raise ValueError(f"Unknown hiqm_phi_map: {mode!r}")


class HIQM:
    """Corrected sigmoid HIQM. phi is a paper similarity bin (0.2-0.7) or a low-end extrapolation."""

    def __init__(
        self,
        phi_mode: str = "legacy",
        phi_ref: float = 0.7,
        params: dict[float, tuple[float, float, float, float]] | None = None,
        phi_extrap_min: float = 0.05,
    ):
        self.phi_mode = phi_mode
        self.phi_ref = max(phi_ref, 1e-6)
        self.params = dict(params or PAPER_HIQM_PARAMS)
        self._phi_keys = sorted(self.params.keys())
        self.phi_extrap_min = phi_extrap_min

    @staticmethod
    def _sigmoid(x, L, k, x0, C):
        return L / (1.0 + np.exp(k * (x - x0))) + C

    def _interpolate_params(self, phi: float) -> tuple[float, float, float, float]:
        keys = self._phi_keys
        if phi >= keys[-1]:
            return self.params[keys[-1]]
        if phi >= keys[0]:
            for i in range(len(keys) - 1):
                lo, hi = keys[i], keys[i + 1]
                if lo <= phi <= hi:
                    w = (hi - phi) / (hi - lo)
                    a, b = self.params[lo], self.params[hi]
                    return tuple(w * a[j] + (1.0 - w) * b[j] for j in range(4))
            return self.params[keys[-1]]

        # phi < 0.2: linearly extrapolate the parameters along (0.2, 0.3).
        lo, hi = keys[0], keys[1]
        p_lo, p_hi = self.params[lo], self.params[hi]
        span = hi - lo
        w = (hi - phi) / span
        return tuple(w * p_lo[j] + (1.0 - w) * p_hi[j] for j in range(4))

    def _raw_q(self, x: float, phi_hiqm: float) -> float:
        phi_hiqm = max(phi_hiqm, self.phi_extrap_min)
        L, k, x0, C = self._interpolate_params(phi_hiqm)
        return float(L / (1.0 + np.exp(k * (x - x0))) + C)

    def _apply_phi_mode(self, raw_q: float, phi_hiqm: float) -> float:
        if self.phi_mode == "legacy":
            return raw_q
        if self.phi_mode == "multiply":
            return raw_q * (phi_hiqm / self.phi_ref)
        if self.phi_mode == "cap":
            return min(raw_q, 0.1 + 0.35 * phi_hiqm)
        raise ValueError(f"Unknown phi_mode: {self.phi_mode}")

    def __call__(self, x: float, phi_hiqm: float) -> float:
        phi_hiqm = float(np.clip(phi_hiqm, 0.0, 1.0))
        return self._apply_phi_mode(self._raw_q(x, phi_hiqm), phi_hiqm)


def make_hiqm(hp: HyperParams | None = None) -> HIQM:
    if hp is None:
        return HIQM()
    return HIQM(
        phi_mode=hp.hiqm_phi_mode,
        phi_ref=hp.hiqm_phi_ref,
        phi_extrap_min=hp.hiqm_phi_min,
    )
