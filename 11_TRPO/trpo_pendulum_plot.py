#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import time

import gymnasium as gym
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from trpo_pendulum import TRPOContinuous


def main():
    env_name = "Pendulum-v1"
    env = gym.make(env_name, render_mode="human")
    device = torch.device("cpu")

    hidden_dim = 128
    lmbda = 0.90
    kl_constraint = 0.00005
    alpha = 0.5
    critic_lr = 1e-2
    gamma = 0.90

    agent = TRPOContinuous(
        hidden_dim,
        env.observation_space,
        env.action_space,
        lmbda,
        kl_constraint,
        alpha,
        critic_lr,
        gamma,
        device,
    )
    agent.load(SCRIPT_DIR / "trpo_pendulum.pth")

    state, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        env.render()
        state_tensor = torch.tensor(
            np.array([state]),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            mu, std = agent.actor(state_tensor)
        action = float(np.clip(
            mu.item(),
            env.action_space.low[0],
            env.action_space.high[0],
        ))
        action = [action]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        time.sleep(0.02)

    print(f"总回报: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
