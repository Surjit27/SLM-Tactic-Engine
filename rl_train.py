"""RL training loop for tactical action refinement."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

import config
from rl_env import TacticalRLEnv
from rl_policy import SoftmaxPolicy


@dataclass
class TrainResult:
    mean_reward_last_20: float
    episodes: int
    model_path: str


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    out = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    # Normalize for stability.
    arr = np.array(out, dtype=np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return arr.tolist()


def train_tactical_policy(
    observations: np.ndarray,
    tactical_targets: np.ndarray,
    episodes: int = config.EPISODES,
    steps_per_episode: int = config.STEPS_PER_EPISODE,
    learning_rate: float = config.LEARNING_RATE,
    gamma: float = config.GAMMA,
    output_dir: str = "artifacts",
) -> TrainResult:
    env = TacticalRLEnv(observations, tactical_targets, max_steps=steps_per_episode)
    policy = SoftmaxPolicy(obs_dim=config.OBS_SIZE, n_actions=config.TACTICAL_ACTIONS, seed=config.RANDOM_SEED)
    rng = np.random.default_rng(config.RANDOM_SEED)

    rewards_log: list[float] = []
    os.makedirs(output_dir, exist_ok=True)

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False

        obs_batch: list[np.ndarray] = []
        act_batch: list[int] = []
        rew_batch: list[float] = []
        ep_reward = 0.0

        while not done:
            action, _ = policy.sample_action(obs, rng)
            next_obs, reward, done, _ = env.step(action)
            obs_batch.append(obs)
            act_batch.append(action)
            rew_batch.append(reward)
            ep_reward += reward
            obs = next_obs

        returns = discounted_returns(rew_batch, gamma=gamma)
        loss = policy.reinforce_update(obs_batch, act_batch, returns, learning_rate=learning_rate)
        rewards_log.append(ep_reward)

        if ep % 20 == 0:
            mean_20 = float(np.mean(rewards_log[-20:]))
            print(f"[episode {ep:03d}] mean_reward(last20)={mean_20:.3f} loss={loss:.4f}")

    model_path = os.path.join(output_dir, "tactical_policy.npz")
    np.savez_compressed(model_path, w=policy.w, b=policy.b)

    mean_reward_last_20 = float(np.mean(rewards_log[-20:])) if len(rewards_log) >= 20 else float(np.mean(rewards_log))
    return TrainResult(
        mean_reward_last_20=mean_reward_last_20,
        episodes=episodes,
        model_path=model_path,
    )
