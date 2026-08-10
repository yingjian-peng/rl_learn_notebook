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

from ppo_train import PPOContinuous



def main():
    env_name = "Pendulum-v1"
    env = gym.make(env_name, render_mode="human")
    device = torch.device("cpu")

    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.shape[0]
    actor_lr = 0.0001
    critic_lr = 0.005
    lmbda = 0.9
    epochs = 10
    eps = 0.2
    gamma = 0.9

    agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
    agent.load(SCRIPT_DIR / "ppo_pendulum.pth")

    state, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        env.render()
        state_tensor = torch.as_tensor(np.asarray(state, dtype=np.float32),device=device,).unsqueeze(0)
        with torch.no_grad():
            mu, _ = agent.actor(state_tensor)

        # PPO 连续策略使用均值作为确定性动作，并限制在环境动作范围内。
        action = np.clip(mu.squeeze(0).cpu().numpy(), env.action_space.low, env.action_space.high).astype(np.float32)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        time.sleep(0.02)

    print(f"总回报: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()





