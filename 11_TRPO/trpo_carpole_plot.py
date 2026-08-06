#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch

from trpo_carpole import TRPO


def take_greedy_action(agent, state):
    state = torch.tensor(np.array([state]), dtype=torch.float).to(agent.device)
    with torch.no_grad():
        probs = agent.actor(state)
    return probs.argmax(dim=1).item()


def main():
    env_name = "CartPole-v1"
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "trpo_cartpole.pth"

    env = gym.make(env_name, render_mode="human", max_episode_steps=200)
    device = torch.device("cpu")

    hidden_dim = 128
    lmbda = 0.95
    kl_constraint = 0.0005
    alpha = 0.5
    critic_lr = 1e-2
    gamma = 0.98

    agent = TRPO(
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
    agent.load(model_path)

    state, info = env.reset(seed=0)
    done = False
    total_reward = 0

    while not done:
        action = take_greedy_action(agent, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        time.sleep(0.02)

    print(f"总回报: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()



