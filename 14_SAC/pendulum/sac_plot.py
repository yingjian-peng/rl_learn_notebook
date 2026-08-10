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

from sac_train import SACContinuous



def main():
    env_name = "Pendulum-v1"
    env = gym.make(env_name, render_mode="human")


    # 构建SAC算法智能体
    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]     # 动作最大值
    actor_lr = 0.0003
    critic_lr = 0.003
    alpha_lr = 0.0003
    target_entropy = -env.action_space.shape[0] 
    tau = 0.005                                  # 软更新参数
    gamma = 0.99
    device = torch.device("cpu")

    agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound, actor_lr, 
        critic_lr, alpha_lr, target_entropy, tau, gamma, device)
    agent.load(SCRIPT_DIR / "sac_pendulum.pth")

    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        env.render()
        state = torch.as_tensor(np.asarray(state, dtype=np.float32), device=device).unsqueeze(0)
        with torch.no_grad():
            action, _ = agent.actor(state)
            action = action.squeeze(0).cpu().numpy()

        action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        time.sleep(0.02)
    
    print(f"Episode return:{total_reward:.3f}")
    env.close()




if __name__ == "__main__":
    main()













