from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rl.config import DQNConfig
from rl.network import QNetwork
from rl.replay_buffer import ReplayBuffer


class DQNAgent:
    """Vanilla DQN (Mnih et al. 2015): online net, target net, replay buffer.

    Kept deliberately minimal and readable — one file, no inheritance —
    so the training mechanics are easy to follow and instrument.
    """

    def __init__(self, obs_dim: int, n_actions: int, config: DQNConfig):
        self.config = config
        self.n_actions = n_actions
        self.device = torch.device(config.device)

        self.q_net = QNetwork(obs_dim, n_actions, config.hidden_sizes).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, config.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_size, obs_dim, seed=config.seed)
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------ act

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy over Q-values; epsilon=0 gives the greedy policy."""
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.n_actions))
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            q = self.q_net(t.unsqueeze(0))
        return int(q.argmax(dim=1).item())

    # ---------------------------------------------------------------- learn

    def update(self) -> dict:
        """One gradient step on a sampled batch. Returns diagnostics."""
        batch = self.buffer.sample(self.config.batch_size)
        obs      = torch.as_tensor(batch["obs"], device=self.device)
        actions  = torch.as_tensor(batch["actions"], device=self.device)
        rewards  = torch.as_tensor(batch["rewards"], device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device)
        dones    = torch.as_tensor(batch["dones"], device=self.device)

        q_pred = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_obs).max(dim=1).values
            target = rewards + self.config.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(q_pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return {
            "loss":   float(loss.item()),
            "mean_q": float(q_pred.mean().item()),
            "max_q":  float(q_pred.max().item()),
        }

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------ persist

    def save(self, path: Path, global_step: int) -> None:
        torch.save(
            {
                "global_step": global_step,
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: Path) -> int:
        """Restore weights/optimizer; returns the checkpoint's global step."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        return int(ckpt["global_step"])
