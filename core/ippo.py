"""Per-agent IPPO trainer (Static-HIS pretraining)."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.buffer import Buffer, RewardNormalizer
from core.network import Actor, Critic
from core.params import HyperParams
from core.seed import device


def normalize_obs(phi: float, v_max_ES: float, v_UE: float, hp: HyperParams) -> np.ndarray:
    """Lower-level observation [phi, v_ES/v_ref, v_UE/v_ref], 3 dimensions."""
    return np.array([
        phi,
        v_max_ES / hp.v_max_ES_ref,
        v_UE / hp.v_UE_ref,
    ], dtype=np.float32)


class IPPOTrainer:
    def __init__(self, hp: HyperParams):
        self.hp = hp
        self.obs_dim = 3
        self.act_dim = 2

        self.actors = [Actor(self.obs_dim, self.act_dim, hp.hidden_layers).to(device) for _ in range(hp.N)]
        self.critics = [Critic(self.obs_dim, hp.hidden_layers).to(device) for _ in range(hp.N)]
        self.actor_opts = [optim.Adam(a.parameters(), lr=hp.actor_lr) for a in self.actors]
        self.critic_opts = [optim.Adam(c.parameters(), lr=hp.critic_lr) for c in self.critics]

        self.buffers = [Buffer() for _ in range(hp.N)]
        self.rnorms = [RewardNormalizer() for _ in range(hp.N)]
        self._frozen = False

    def set_frozen(self, frozen: bool) -> None:
        self._frozen = frozen
        for actor, critic in zip(self.actors, self.critics):
            for p in actor.parameters():
                p.requires_grad = not frozen
            for p in critic.parameters():
                p.requires_grad = not frozen
            if frozen:
                actor.eval()
                critic.eval()
            else:
                actor.train()
                critic.train()

    def select_action(self, n, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = self.actors[n](obs_t)
            mean = logits
            std = torch.ones_like(mean) * 0.5
            dist = torch.distributions.Normal(mean, std)
            act = mean if self._frozen else dist.sample()
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

    def update(self):
        if self._frozen:
            for buf in self.buffers:
                buf.clear()
            return
        for n in range(self.hp.N):
            buf = self.buffers[n]
            if len(buf.obs) == 0:
                continue

            obs = torch.tensor(np.array(buf.obs), dtype=torch.float32, device=device)
            act = torch.tensor(np.array(buf.act), dtype=torch.float32, device=device)
            old_logp = torch.tensor(np.array(buf.logp), dtype=torch.float32, device=device)

            with torch.no_grad():
                values = self.critics[n](obs).squeeze(-1).cpu().numpy()
                next_obs = torch.tensor(np.array(buf.next_obs), dtype=torch.float32, device=device)
                next_values = self.critics[n](next_obs).squeeze(-1).cpu().numpy()

            rewards = np.array(buf.rew, dtype=np.float32)
            dones = np.array(buf.done, dtype=np.float32)

            adv = self.compute_gae(rewards, values, next_values, dones)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            returns = adv + values

            adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
            ret_t = torch.tensor(returns, dtype=torch.float32, device=device)

            with torch.no_grad():
                values_old_t = self.critics[n](obs).squeeze(-1)

            for _ in range(self.hp.train_epochs):
                logits = self.actors[n](obs)
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

                self.actor_opts[n].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[n].parameters(), 0.5)
                self.actor_opts[n].step()

                values_pred = self.critics[n](obs).squeeze(-1)
                values_clipped = values_old_t + torch.clamp(
                    values_pred - values_old_t,
                    -self.hp.critic_clip_eps,
                    self.hp.critic_clip_eps
                )
                critic_loss = torch.max(
                    (values_pred - ret_t).pow(2),
                    (values_clipped - ret_t).pow(2)
                ).mean()

                self.critic_opts[n].zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critics[n].parameters(), 0.5)
                self.critic_opts[n].step()

            buf.clear()

    def save(self, path: str) -> None:
        state = {f"actor_{n}": a.state_dict() for n, a in enumerate(self.actors)}
        state.update({f"critic_{n}": c.state_dict() for n, c in enumerate(self.critics)})
        torch.save(state, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=device, weights_only=False)
        for n in range(self.hp.N):
            self.actors[n].load_state_dict(state[f"actor_{n}"])
            self.critics[n].load_state_dict(state[f"critic_{n}"])
        if self._frozen:
            for actor, critic in zip(self.actors, self.critics):
                actor.eval()
                critic.eval()
