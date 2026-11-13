"""Actor/Critic networks: continuous-action IPPO and discrete-action CM-PPO."""

from typing import Tuple

import torch.nn as nn


class Actor(nn.Module):
    """Continuous-action policy head. Outputs the mean inference steps (x, y)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DiscreteActor(nn.Module):
    """Discrete-action policy (CM-PPO)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """State-value network V(s). IPPO and CM-PPO share this structure."""

    def __init__(self, obs_dim: int, hidden: Tuple[int, ...]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
