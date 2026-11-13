"""Public inference-state cache (SP3): search the whole history pool; unbind on recluster; refresh forces a miss."""

from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np

from core.clustering import compute_phi
from core.degradation import cluster_centroid
from core.params import CacheThresholds


@dataclass
class CacheEntry:
    cluster_id: int  # -1 = unbound historical entry
    e_pub: np.ndarray
    mu: np.ndarray
    x_cache: float
    t_created: int
    last_used: int = 0


@dataclass
class ClusterLookupResult:
    hit: bool
    x_saved: float = 0.0
    entry: Optional[CacheEntry] = None


class ClusterCacheManager:
    """Reusable public inference states in a history pool. enabled=False means no cache.

    b_max=None keeps an unlimited pool and appends once per cluster per step.
    A finite capacity evicts by last_used (LRU) after each write.
    """

    def __init__(
        self,
        enabled: bool,
        thresholds: CacheThresholds,
        b_max: Optional[int] = None,
    ):
        self.enabled = enabled
        self.thresholds = thresholds
        self.b_max = b_max
        self.entries: List[CacheEntry] = []
        self._force_miss: Set[int] = set()
        self.lookup_calls = 0
        self.lookup_comparisons = 0

    def reset(self) -> None:
        self.entries.clear()
        self._force_miss.clear()
        self.lookup_calls = 0
        self.lookup_comparisons = 0

    def _evict_if_needed(self) -> None:
        if self.b_max is None:
            return
        if self.b_max <= 0:
            self.entries.clear()
            return
        while len(self.entries) > self.b_max:
            idx = min(
                range(len(self.entries)),
                key=lambda i: (
                    self.entries[i].last_used,
                    self.entries[i].t_created,
                    i,
                ),
            )
            del self.entries[idx]

    def force_miss_clusters(self, cluster_ids: List[int]) -> None:
        """After a rule refresh, this slot is a forced miss until store."""
        if not self.enabled or not cluster_ids:
            return
        self._force_miss.update(cluster_ids)

    def lookup(
        self,
        m: int,
        e_pub: np.ndarray,
        mu: np.ndarray,
        x_m: float,
        t: int,
    ) -> ClusterLookupResult:
        self.lookup_calls += 1
        if not self.enabled or m in self._force_miss:
            return ClusterLookupResult(hit=False)

        best: Optional[CacheEntry] = None
        best_score = -1.0
        # Search the whole pool so a recluster label swap is not ambiguous.
        for entry in self.entries:
            self.lookup_comparisons += 1
            sim_pub = compute_phi(e_pub, entry.e_pub)
            sim_mu = compute_phi(mu, entry.mu)
            if (
                sim_pub >= self.thresholds.tau_p
                and sim_mu >= self.thresholds.tau_mu
                and abs(x_m - entry.x_cache) <= self.thresholds.delta_x
            ):
                score = sim_pub + sim_mu
                if score > best_score:
                    best_score = score
                    best = entry

        if best is None:
            return ClusterLookupResult(hit=False)

        best.last_used = t
        x_saved = min(best.x_cache, x_m)
        return ClusterLookupResult(hit=True, x_saved=x_saved, entry=best)

    def store(
        self,
        m: int,
        e_pub: np.ndarray,
        mu: np.ndarray,
        x_m: float,
        t: int,
    ) -> None:
        if not self.enabled:
            return

        self.entries.append(
            CacheEntry(
                cluster_id=m,
                e_pub=e_pub.copy(),
                mu=mu.copy(),
                x_cache=x_m,
                t_created=t,
                last_used=t,
            )
        )
        self._force_miss.discard(m)
        self._evict_if_needed()

    def invalidate_on_recluster(self) -> None:
        """Unbind the cluster index and keep the physical entry in the history pool."""
        for entry in self.entries:
            entry.cluster_id = -1
        self._force_miss.clear()

    def invalidate_clusters(self, cluster_ids: List[int]) -> None:
        """After a move, drop bindings of affected clusters; entries stay unbound."""
        if not self.enabled or not cluster_ids:
            return
        blocked = set(cluster_ids)
        for entry in self.entries:
            if entry.cluster_id in blocked:
                entry.cluster_id = -1
        self._force_miss.update(blocked)

    def inherit_on_move(
        self,
        old_assign: List[int],
        new_assign: List[int],
        cluster_embeds: List[np.ndarray],
        embeds: List[np.ndarray],
        t: int,
    ) -> None:
        """When a user changes cluster, copy the source cache entry to the target and refresh e_pub/mu."""
        if not self.enabled:
            return
        pairs: set[tuple[int, int]] = set()
        for a0, a1 in zip(old_assign, new_assign):
            if a0 != a1:
                pairs.add((a0, a1))
        for src, dst in pairs:
            src_entries = [e for e in self.entries if e.cluster_id == src]
            if not src_entries:
                continue
            donor = max(src_entries, key=lambda e: e.last_used)
            members = [n for n, m in enumerate(new_assign) if m == dst]
            if not members or dst >= len(cluster_embeds):
                continue
            mu = cluster_centroid(members, embeds)
            self.entries.append(
                CacheEntry(
                    cluster_id=dst,
                    e_pub=cluster_embeds[dst].copy(),
                    mu=mu.copy(),
                    x_cache=donor.x_cache,
                    t_created=t,
                    last_used=t,
                )
            )
        self._evict_if_needed()
