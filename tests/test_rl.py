"""Comprehensive tests for the RL module components.

Tests cover: ReplayBuffer, QNetwork, DQNAgent, RunLogger, and NormalizeObservation.
Run via: python -m pytest tests/test_rl.py -v
"""
import os
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest

from rl.config import DQNConfig
from rl.network import QNetwork
from rl.replay_buffer import ReplayBuffer
from rl.agent import DQNAgent
from rl.logger import RunLogger, CsvLog
from sailing_env.env import SailingEnv
from sailing_env.wrappers import NormalizeObservation


# =============================================================================
# REPLAY BUFFER TESTS
# =============================================================================

class TestReplayBuffer:
    """Test the fixed-size ring buffer."""

    def test_len_grows_to_capacity(self):
        """Length grows until capacity is reached, then stays constant."""
        capacity = 5
        buf = ReplayBuffer(capacity=capacity, obs_dim=2, seed=42)

        assert len(buf) == 0

        for i in range(capacity + 3):
            obs = np.array([float(i), float(i+1)], dtype=np.float32)
            action = i % 3
            reward = float(i)
            next_obs = np.array([float(i+1), float(i+2)], dtype=np.float32)
            done = False
            buf.add(obs, action, reward, next_obs, done)

        # After adding capacity items, length should be capacity
        # After adding more, length stays at capacity
        assert len(buf) == capacity

    def test_wraparound_overwrites_oldest(self):
        """Ring buffer wraps around and overwrites oldest entries."""
        capacity = 5
        buf = ReplayBuffer(capacity=capacity, obs_dim=2, seed=42)

        # Fill buffer with transitions 0-4
        for i in range(capacity):
            obs = np.array([float(i), float(i)], dtype=np.float32)
            action = i
            reward = float(i * 10)
            next_obs = np.array([float(i+1), float(i+1)], dtype=np.float32)
            done = False
            buf.add(obs, action, reward, next_obs, done)

        # Add one more (should overwrite position 0 with value 5)
        obs = np.array([5.0, 5.0], dtype=np.float32)
        action = 5
        reward = 50.0
        next_obs = np.array([6.0, 6.0], dtype=np.float32)
        done = False
        buf.add(obs, action, reward, next_obs, done)

        # All entries should have actions 1, 2, 3, 4, 5
        # (the initial 0 was overwritten)
        assert len(buf) == capacity

        # Verify by sampling many times and checking that
        # we see actions 1-5 but not 0
        all_actions_seen = set()
        for _ in range(100):
            sample = buf.sample(1)
            all_actions_seen.update(sample["actions"])

        # Action 0 should never appear (it was overwritten)
        assert 0 not in all_actions_seen
        # Actions 1-5 should appear
        assert all(a in all_actions_seen for a in [1, 2, 3, 4, 5])

    def test_sample_shapes_and_dtypes(self):
        """Sample returns correct shapes and dtypes."""
        capacity = 10
        obs_dim = 4
        batch_size = 3
        buf = ReplayBuffer(capacity=capacity, obs_dim=obs_dim, seed=42)

        # Fill buffer
        for i in range(capacity):
            obs = np.random.randn(obs_dim).astype(np.float32)
            action = i % 3
            reward = float(np.random.randn())
            next_obs = np.random.randn(obs_dim).astype(np.float32)
            done = i % 5 == 4  # Some episodes done
            buf.add(obs, action, reward, next_obs, done)

        sample = buf.sample(batch_size)

        # Check keys
        assert set(sample.keys()) == {"obs", "actions", "rewards", "next_obs", "dones"}

        # Check shapes
        assert sample["obs"].shape == (batch_size, obs_dim)
        assert sample["actions"].shape == (batch_size,)
        assert sample["rewards"].shape == (batch_size,)
        assert sample["next_obs"].shape == (batch_size, obs_dim)
        assert sample["dones"].shape == (batch_size,)

        # Check dtypes
        assert sample["obs"].dtype == np.float32
        assert sample["actions"].dtype == np.int64
        assert sample["rewards"].dtype == np.float32
        assert sample["next_obs"].dtype == np.float32
        assert sample["dones"].dtype == np.float32

    def test_sample_only_contains_added_transitions(self):
        """Sample only returns transitions that were added."""
        capacity = 10
        obs_dim = 2
        buf = ReplayBuffer(capacity=capacity, obs_dim=obs_dim, seed=42)

        added_actions = []
        for i in range(5):  # Only add 5 transitions
            obs = np.array([float(i), 0.0], dtype=np.float32)
            action = i * 10  # Distinctive action values
            reward = float(i)
            next_obs = np.array([float(i+1), 0.0], dtype=np.float32)
            done = False
            buf.add(obs, action, reward, next_obs, done)
            added_actions.append(action)

        # Sample multiple times and verify only added actions appear
        for _ in range(10):
            sample = buf.sample(5)
            sampled_actions = set(sample["actions"])
            assert sampled_actions.issubset(set(added_actions))


# =============================================================================
# QNETWORK TESTS
# =============================================================================

class TestQNetwork:
    """Test the Q-network."""

    def test_output_shape(self):
        """Output shape is (batch_size, n_actions)."""
        obs_dim = 8
        n_actions = 3
        batch_size = 4

        net = QNetwork(obs_dim=obs_dim, n_actions=n_actions)

        obs = torch.randn(batch_size, obs_dim)
        output = net(obs)

        assert output.shape == (batch_size, n_actions)

    def test_single_observation_batch(self):
        """Works for a batch of 1 observation."""
        obs_dim = 8
        n_actions = 3

        net = QNetwork(obs_dim=obs_dim, n_actions=n_actions)

        obs = torch.randn(1, obs_dim)
        output = net(obs)

        assert output.shape == (1, n_actions)

    def test_output_is_differentiable(self):
        """Output can be used in a loss function (is differentiable)."""
        obs_dim = 8
        n_actions = 3

        net = QNetwork(obs_dim=obs_dim, n_actions=n_actions)

        obs = torch.randn(2, obs_dim, requires_grad=True)
        output = net(obs)

        # Compute a simple loss
        loss = output.sum()
        loss.backward()

        # Network parameters should have gradients
        for param in net.parameters():
            if param.requires_grad:
                assert param.grad is not None


# =============================================================================
# DQNAGENT TESTS
# =============================================================================

class TestDQNAgent:
    """Test the DQN agent."""

    def _make_agent(self, buffer_size=100, batch_size=8, obs_dim=8, n_actions=3):
        """Helper to create an agent with test config."""
        config = DQNConfig(
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_starts=10,  # Low threshold for testing
            eps_start=1.0,
            eps_end=0.05,
            eps_decay_steps=1000,
            seed=42,
            device="cpu",
            hidden_sizes=(64, 64),
        )
        agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, config=config)
        return agent

    def test_select_action_deterministic_with_zero_epsilon(self):
        """With epsilon=0, select_action is deterministic and greedy."""
        agent = self._make_agent()
        obs = np.random.randn(8).astype(np.float32)

        # Multiple calls with epsilon=0 should give same action
        actions = [agent.select_action(obs, epsilon=0.0) for _ in range(5)]
        assert len(set(actions)) == 1  # All same
        assert actions[0] in [0, 1, 2]  # Valid action range

    def test_select_action_explores_with_epsilon_one(self):
        """With epsilon=1, select_action explores (random)."""
        agent = self._make_agent()
        obs = np.random.randn(8).astype(np.float32)

        # Many calls with epsilon=1 should give varied actions
        actions = [agent.select_action(obs, epsilon=1.0) for _ in range(30)]

        # Should see multiple different actions
        unique_actions = set(actions)
        assert len(unique_actions) > 1  # Should explore
        assert all(a in [0, 1, 2] for a in unique_actions)

    def test_select_action_in_valid_range(self):
        """select_action always returns action in [0, n_actions)."""
        agent = self._make_agent(n_actions=3)
        obs = np.random.randn(8).astype(np.float32)

        for epsilon in [0.0, 0.5, 1.0]:
            for _ in range(10):
                action = agent.select_action(obs, epsilon=epsilon)
                assert 0 <= action < 3

    def test_update_returns_finite_metrics(self):
        """update() returns dict with finite float metrics."""
        agent = self._make_agent()

        # Fill buffer with random transitions
        for _ in range(20):
            obs = np.random.randn(8).astype(np.float32)
            action = np.random.randint(3)
            reward = np.random.randn()
            next_obs = np.random.randn(8).astype(np.float32)
            done = np.random.rand() < 0.2
            agent.buffer.add(obs, action, reward, next_obs, done)

        # Call update
        metrics = agent.update()

        # Check structure
        assert isinstance(metrics, dict)
        assert set(metrics.keys()) == {"loss", "mean_q", "max_q"}

        # Check values are finite floats
        for key, value in metrics.items():
            assert isinstance(value, float)
            assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_update_changes_parameters(self):
        """After several updates, network parameters change."""
        agent = self._make_agent()

        # Get initial parameter values
        initial_params = [p.clone() for p in agent.q_net.parameters()]

        # Fill buffer and do several updates
        for _ in range(30):
            obs = np.random.randn(8).astype(np.float32)
            action = np.random.randint(3)
            reward = np.random.randn()
            next_obs = np.random.randn(8).astype(np.float32)
            done = np.random.rand() < 0.2
            agent.buffer.add(obs, action, reward, next_obs, done)

        for _ in range(5):
            agent.update()

        # Check that at least one parameter changed
        final_params = list(agent.q_net.parameters())
        params_changed = False
        for initial, final in zip(initial_params, final_params):
            if not torch.allclose(initial, final):
                params_changed = True
                break

        assert params_changed, "No parameters changed after updates"

    def test_sync_target_copies_weights(self):
        """sync_target() copies online weights to target."""
        agent = self._make_agent()

        # Weights should initially be equal (set in __init__)
        for qp, tp in zip(agent.q_net.parameters(), agent.target_net.parameters()):
            assert torch.allclose(qp, tp)

        # Manually modify first layer weight to be different
        agent.q_net.net[0].weight.data += 0.1

        # Now they should be different
        different = False
        for qp, tp in zip(agent.q_net.parameters(), agent.target_net.parameters()):
            if not torch.allclose(qp, tp):
                different = True
                break
        assert different

        # After sync, they should match again
        agent.sync_target()
        for qp, tp in zip(agent.q_net.parameters(), agent.target_net.parameters()):
            assert torch.allclose(qp, tp)

    def test_save_and_load_round_trip(self, tmp_path):
        """Save/load preserves weights and global_step."""
        agent = self._make_agent()

        # Get initial weights
        initial_weights = [p.clone() for p in agent.q_net.parameters()]

        # Do a few updates to change weights
        for _ in range(20):
            obs = np.random.randn(8).astype(np.float32)
            action = np.random.randint(3)
            reward = np.random.randn()
            next_obs = np.random.randn(8).astype(np.float32)
            done = False
            agent.buffer.add(obs, action, reward, next_obs, done)

        for _ in range(3):
            agent.update()

        # Get updated weights
        updated_weights = [p.clone() for p in agent.q_net.parameters()]

        # Verify weights did change
        assert not all(torch.allclose(iw, uw) for iw, uw in zip(initial_weights, updated_weights))

        # Save
        ckpt_path = tmp_path / "agent.pt"
        global_step = 12345
        agent.save(ckpt_path, global_step=global_step)

        # Create a new agent and load
        agent2 = self._make_agent()
        loaded_step = agent2.load(ckpt_path)

        # Check global_step
        assert loaded_step == global_step

        # Check weights are identical
        for p1, p2 in zip(agent.q_net.parameters(), agent2.q_net.parameters()):
            assert torch.allclose(p1, p2)


# =============================================================================
# RUNLOGGER AND CSVLOG TESTS
# =============================================================================

class TestCsvLog:
    """Test CSV logging."""

    def test_header_written_once(self, tmp_path):
        """Header is written once on creation."""
        csv_path = tmp_path / "test.csv"
        columns = ["step", "loss", "reward"]

        log = CsvLog(csv_path, columns)

        # File should exist with header
        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert lines[0] == "step,loss,reward"
        assert len(lines) == 1  # Only header

    def test_append_adds_row(self, tmp_path):
        """append() adds a row in column order."""
        csv_path = tmp_path / "test.csv"
        columns = ["step", "loss", "reward"]

        log = CsvLog(csv_path, columns)
        log.append({"step": 1, "loss": 0.5, "reward": 10.0})
        log.append({"step": 2, "loss": 0.4, "reward": 20.0})

        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 3  # Header + 2 rows
        assert lines[1] == "1,0.5,10.0"
        assert lines[2] == "2,0.4,20.0"

    def test_reopening_does_not_rewrite_header(self, tmp_path):
        """Reopening an existing log doesn't rewrite header."""
        csv_path = tmp_path / "test.csv"
        columns = ["step", "loss"]

        log1 = CsvLog(csv_path, columns)
        log1.append({"step": 1, "loss": 0.5})

        log2 = CsvLog(csv_path, columns)
        log2.append({"step": 2, "loss": 0.4})

        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 3  # Header + 2 rows
        assert lines[0] == "step,loss"


class TestRunLogger:
    """Test the run logger."""

    def test_creates_directory_structure(self, tmp_path):
        """Creates eval/ and checkpoints/ directories."""
        run_dir = tmp_path / "my_run"
        logger = RunLogger(run_dir)

        assert (run_dir / "eval").exists()
        assert (run_dir / "checkpoints").exists()

        # CSV files should be created
        assert (run_dir / "episodes.csv").exists()
        assert (run_dir / "training.csv").exists()
        assert (run_dir / "evals.csv").exists()

    def test_save_trajectory_creates_json(self, tmp_path):
        """save_trajectory() writes parseable JSON."""
        run_dir = tmp_path / "my_run"
        logger = RunLogger(run_dir)

        traj = {
            "observations": [[1.0, 2.0], [3.0, 4.0]],
            "actions": [0, 1],
            "rewards": [10.0, 20.0],
        }

        path = logger.save_trajectory(global_step=1000, episode=5, traj=traj)

        # Check filename and location
        assert path.parent == run_dir / "eval"
        assert path.name == "traj_step1000_ep5.json"

        # Verify it's valid JSON
        saved_traj = json.loads(path.read_text())
        assert saved_traj == traj

    def test_checkpoint_path_format(self, tmp_path):
        """checkpoint_path() returns expected path."""
        run_dir = tmp_path / "my_run"
        logger = RunLogger(run_dir)

        path = logger.checkpoint_path(global_step=5000)

        assert path.parent == run_dir / "checkpoints"
        assert path.name == "ckpt_5000.pt"


# =============================================================================
# NORMALIZE OBSERVATION WRAPPER TESTS
# =============================================================================

class TestNormalizeObservationWrapper:
    """Test the observation normalization wrapper."""

    def test_observation_space_is_normalized(self):
        """Wrapped observation_space is [-1, 1]."""
        env = SailingEnv()
        wrapped = NormalizeObservation(env)

        space = wrapped.observation_space
        assert space.low.min() >= -1.0
        assert space.high.max() <= 1.0
        assert np.allclose(space.low, -1.0)
        assert np.allclose(space.high, 1.0)

    def test_reset_returns_normalized_obs(self):
        """reset() returns observation in [-1, 1] with dtype float32."""
        env = SailingEnv()
        wrapped = NormalizeObservation(env)

        obs, info = wrapped.reset(seed=42)

        assert obs.dtype == np.float32
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)

    def test_step_returns_normalized_obs(self):
        """step() returns observation in [-1, 1] with dtype float32."""
        env = SailingEnv()
        wrapped = NormalizeObservation(env)

        wrapped.reset(seed=42)

        for _ in range(10):
            obs, reward, terminated, truncated, info = wrapped.step(wrapped.action_space.sample())

            assert obs.dtype == np.float32
            assert np.all(obs >= -1.0)
            assert np.all(obs <= 1.0)

            if terminated or truncated:
                break

    def test_normalization_formula(self):
        """Normalization maps input bounds to [-1, 1]."""
        env = SailingEnv()
        wrapped = NormalizeObservation(env)

        # Manually check the normalization formula
        # obs_normalized = 2.0 * (obs - low) / span - 1.0
        # At low: 2.0 * (low - low) / span - 1.0 = -1.0
        # At high: 2.0 * (high - low) / span - 1.0 = 2.0 * span / span - 1.0 = 1.0

        env_obs_space = env.observation_space
        low = env_obs_space.low.astype(np.float32)
        span = (env_obs_space.high - env_obs_space.low).astype(np.float32)

        # Test at the bounds
        obs_at_low = low
        expected_normalized_low = 2.0 * (obs_at_low - low) / span - 1.0
        assert np.allclose(expected_normalized_low, -1.0)

        obs_at_high = env_obs_space.high.astype(np.float32)
        expected_normalized_high = 2.0 * (obs_at_high - low) / span - 1.0
        assert np.allclose(expected_normalized_high, 1.0)

    def test_wrapper_preserves_env_interface(self):
        """Wrapped env still has proper Gym interface."""
        env = SailingEnv()
        wrapped = NormalizeObservation(env)

        # Should have action and observation spaces
        assert hasattr(wrapped, "action_space")
        assert hasattr(wrapped, "observation_space")

        # Should be able to reset and step
        obs, info = wrapped.reset()
        assert obs is not None

        obs, reward, terminated, truncated, info = wrapped.step(wrapped.action_space.sample())
        assert obs is not None
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_agent_with_normalized_env(self):
        """Agent can work with wrapped environment."""
        env = SailingEnv()
        wrapped = NormalizeObservation(env)

        obs, info = wrapped.reset(seed=42)
        obs_dim = obs.shape[0]
        n_actions = wrapped.action_space.n

        config = DQNConfig(
            buffer_size=50,
            batch_size=8,
            learning_starts=5,
            seed=42,
            device="cpu",
        )
        agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, config=config)

        # Play a few steps
        for _ in range(20):
            action = agent.select_action(obs, epsilon=0.5)
            next_obs, reward, terminated, truncated, info = wrapped.step(action)
            done = terminated or truncated

            agent.buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs

            if len(agent.buffer) >= config.batch_size:
                metrics = agent.update()
                assert all(np.isfinite(v) for v in metrics.values())

            if done:
                obs, info = wrapped.reset()

    def test_config_save_and_load(self, tmp_path):
        """DQNConfig save/load round-trip."""
        config = DQNConfig(
            buffer_size=500,
            batch_size=32,
            learning_starts=100,
            lr=5e-5,
            gamma=0.99,
            hidden_sizes=(256, 256),
            seed=123,
        )

        config_path = tmp_path / "config.json"
        config.save(config_path)

        loaded = DQNConfig.load(config_path)

        assert loaded.buffer_size == config.buffer_size
        assert loaded.batch_size == config.batch_size
        assert loaded.learning_starts == config.learning_starts
        assert loaded.lr == config.lr
        assert loaded.gamma == config.gamma
        assert loaded.hidden_sizes == config.hidden_sizes
        assert loaded.seed == config.seed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
