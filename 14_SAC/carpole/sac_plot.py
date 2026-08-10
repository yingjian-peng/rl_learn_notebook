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

from sac_train import SAC



def main():
    env_name = "CartPole-v1"
    env = gym.make(env_name, render_mode="human")


    # 构建SAC算法智能体
    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.n
    actor_lr = 0.001
    critic_lr = 0.01
    alpha_lr = 0.01
    target_entropy = -1
    tau = 0.005                                  # 软更新参数
    gamma = 0.98
    device = torch.device("cpu")

    agent = SAC(state_dim, hidden_dim, action_dim, actor_lr, 
        critic_lr, alpha_lr, target_entropy, tau, gamma, device)
    agent.load(SCRIPT_DIR / "sac_cartpole.pth")

    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        env.render()
        state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_probs = agent.actor(state)
            action = action_probs.argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        time.sleep(0.02)
    
    print(f"Episode return:{total_reward:.3f}")
    env.close()




if __name__ == "__main__":
    main()












