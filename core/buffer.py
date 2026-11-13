"""PPO rollout buffer and per-agent reward normalization."""

import math


class Buffer:
    """Generic on-policy trajectory buffer (obs / act / logp / rew / next_obs / done)."""

    def __init__(self):
        self.obs, self.act, self.logp, self.rew, self.next_obs, self.done = [], [], [], [], [], []

    def store(self, o, a, logp, r, o2, d):
        self.obs.append(o)
        self.act.append(a)
        self.logp.append(logp)
        self.rew.append(r)
        self.next_obs.append(o2)
        self.done.append(d)

    def clear(self):
        self.__init__()


class RewardNormalizer:
    """Online Welford mean and variance, used to scale IPPO rewards."""

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def normalize(self, r):
        self.count += 1.0
        delta = r - self.mean
        self.mean += delta / self.count
        delta2 = r - self.mean
        self.var += delta * delta2
        std = math.sqrt(self.var / self.count)
        return (r - self.mean) / (std + 1e-8)
