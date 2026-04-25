"""DQN training entry point — stub, wire up once physics are implemented."""

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from sailing_env import SailingEnv


def main():
    env = SailingEnv(render_mode=None)
    check_env(env, warn=True)   # validates spaces and API compliance

    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        # TODO: tune these hyperparameters
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
    )

    model.learn(total_timesteps=500_000)
    model.save("sailing_dqn")
    print("Model saved to sailing_dqn.zip")


if __name__ == "__main__":
    main()
