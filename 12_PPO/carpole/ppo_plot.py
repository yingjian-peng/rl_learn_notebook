#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import time

import gymnasium as gym
import torch

from ppo_train import PPO



def main():
    env_name = "CartPole-v1"
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "ppo_cartpole.pth"

    env = gym.make(env_name, render_mode="human", max_episode_steps=200)
    device = torch.device("cpu")

    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.n
    actor_lr = 0.001
    critic_lr = 1e-2
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    gamma = 0.98

    agent = PPO(
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        lmbda,
        epochs,
        eps,
        gamma,
        device,
    )
    agent.load(model_path)

    state, info = env.reset(seed=0)
    done = False
    total_reward = 0

    while not done:
        state = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            probs = agent.actor(state)
            action = probs.argmax(dim=1).item()
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        time.sleep(0.02)

    print(f"总回报: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()

