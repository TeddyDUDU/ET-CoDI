"""Parameter-shared IPPO: lower-level step allocation (x, y), 3-D observation."""

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.buffer import Buffer, RewardNormalizer
from core.ippo import normalize_obs
from core.network import Actor, Critic
from core.params import HyperParams
from core.seed import device

# One shared Actor/Critic for all users. The checkpoint is a single state_dict, not an average over agents.
normalize_shared_obs = normalize_obs


class SharedIPPOTrainer:
    def __init__(self, hp: HyperParams, actor_lr: Optional[float] = None, critic_lr: Optional[float] = None):
        self.hp = hp
        self.obs_dim = 3
        self.act_dim = 2
        alr = actor_lr if actor_lr is not None else hp.actor_lr
        clr = critic_lr if critic_lr is not None else hp.critic_lr

        self.actor = Actor(self.obs_dim, self.act_dim, hp.hidden_layers).to(device)
        self.critic = Critic(self.obs_dim, hp.hidden_layers).to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=alr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=clr)

        self.buffers = [Buffer() for _ in range(hp.N)]
        self.rnorms = [RewardNormalizer() for _ in range(hp.N)]
        self._frozen = False

    def set_frozen(self, frozen: bool) -> None:
        self._frozen = frozen
        for p in self.actor.parameters():
            p.requires_grad = not frozen
        for p in self.critic.parameters():
            p.requires_grad = not frozen
        if frozen:
            self.actor.eval()
            self.critic.eval()
        else:
            self.actor.train()
            self.critic.train()

    def init_from_exp0(self, ippo_path: str, agent_id: Optional[int] = None) -> int:
        """
        Initialize the shared network from one agent's Actor/Critic in a per-agent IPPO checkpoint.

        If agent_id is None, pick the agent whose action is closest to (x, y) = (10, 5) on a typical observation.
        Returns the agent_id actually used.
        """
        if not os.path.isfile(ippo_path):
            return -1
        state = torch.load(ippo_path, map_location=device, weights_only=False)

        if agent_id is None:
            probe = normalize_obs(0.28, 1386.0, 75.0, self.hp)
            probe_t = torch.tensor(probe, dtype=torch.float32, device=device)
            best_dist, agent_id = float("inf"), 0
            probe_actor = Actor(self.obs_dim, self.act_dim, self.hp.hidden_layers).to(device)
            for n in range(self.hp.N):
                probe_actor.load_state_dict(state[f"actor_{n}"])
                probe_actor.eval()
                with torch.no_grad():
                    mean = probe_actor(probe_t).cpu().numpy()
                dist = (mean[0] - 10.0) ** 2 + (mean[1] - 5.0) ** 2
                if dist < best_dist:
                    best_dist, agent_id = dist, n

        self.actor.load_state_dict(state[f"actor_{agent_id}"])
        self.critic.load_state_dict(state[f"critic_{agent_id}"])
        return agent_id

    def select_action(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = self.actor(obs_t)
            mean = logits
            std = torch.ones_like(mean) * 0.5
            dist = torch.distributions.Normal(mean, std)
            if self._frozen:
                act = mean
            else:
                act = dist.sample()
            logp = dist.log_prob(act).sum(dim=-1)
        return act.detach().cpu().numpy(), float(logp.item())

    def compute_gae(self, rewards, values, next_values, dones):
        adv = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.hp.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.hp.gamma * self.hp.gae_lambda * (1 - dones[t]) * gae
            adv.insert(0, gae)
        return np.array(adv, dtype=np.float32)

    def update(self) -> None:
        if self._frozen:
            for buf in self.buffers:
                buf.clear()
            return

        obs_chunks, act_chunks, logp_chunks, adv_chunks, ret_chunks = [], [], [], [], []
        for buf in self.buffers:
            if len(buf.obs) == 0:
                continue

            obs = torch.tensor(np.array(buf.obs), dtype=torch.float32, device=device)
            act = torch.tensor(np.array(buf.act), dtype=torch.float32, device=device)
            old_logp = torch.tensor(np.array(buf.logp), dtype=torch.float32, device=device)

            with torch.no_grad():
                values = self.critic(obs).squeeze(-1).cpu().numpy()
                next_obs = torch.tensor(np.array(buf.next_obs), dtype=torch.float32, device=device)
                next_values = self.critic(next_obs).squeeze(-1).cpu().numpy()

            rewards = np.array(buf.rew, dtype=np.float32)
            dones = np.array(buf.done, dtype=np.float32)
            adv = self.compute_gae(rewards, values, next_values, dones)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            returns = adv + values

            obs_chunks.append(obs)
            act_chunks.append(act)
            logp_chunks.append(old_logp)
            adv_chunks.append(torch.tensor(adv, dtype=torch.float32, device=device))
            ret_chunks.append(torch.tensor(returns, dtype=torch.float32, device=device))

        for buf in self.buffers:
            buf.clear()

        if not obs_chunks:
            return

        obs = torch.cat(obs_chunks, dim=0)
        act = torch.cat(act_chunks, dim=0)
        old_logp = torch.cat(logp_chunks, dim=0)
        adv_t = torch.cat(adv_chunks, dim=0)
        ret_t = torch.cat(ret_chunks, dim=0)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        with torch.no_grad():
            values_old_t = self.critic(obs).squeeze(-1)

        for _ in range(self.hp.train_epochs):
            logits = self.actor(obs)
            mean = logits
            std = torch.ones_like(mean) * 0.5
            dist = torch.distributions.Normal(mean, std)
            logp = dist.log_prob(act).sum(dim=-1)

            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.hp.clip_eps, 1 + self.hp.clip_eps) * adv_t
            actor_loss = -(torch.min(surr1, surr2)).mean()
            entropy = dist.entropy().sum(dim=-1).mean()
            actor_loss -= self.hp.entropy_coef * entropy

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()

            values_pred = self.critic(obs).squeeze(-1)
            values_clipped = values_old_t + torch.clamp(
                values_pred - values_old_t,
                -self.hp.critic_clip_eps,
                self.hp.critic_clip_eps,
            )
            critic_loss = torch.max(
                (values_pred - ret_t).pow(2),
                (values_clipped - ret_t).pow(2),
            ).mean()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_opt.step()

    def save(self, path: str) -> None:
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
            path,
        )

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=device, weights_only=False)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
