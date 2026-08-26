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

from dqn_train import DQN



#---------------------------------------------------------------------------------------
# 主函数
#---------------------------------------------------------------------------------------
def main():
    env_name = "CartPole-v1"
    env = gym.make(env_name, render_mode="human")


    state_dim = env.observation_space.shape[0]              # 状态维度
    hidden_dim = 128                                        # 隐藏层维度
    action_dim = env.action_space.n                         # 动作维度
    learning_rate = 1e-3                                    # 学习率
    gamma = 0.98                                            # 折扣因子
    epsilon = 0.1                                           # epsilon-贪婪策略中的参数
    target_update = 100                                     # 目标网络更新频率
    device = torch.device("cpu")
    agent = DQN(state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device)

    agent.load(SCRIPT_DIR / "dqn_cartpole.pth")

    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        env.render()
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        time.sleep(0.02)

    print(f"Episode return: {total_reward:.3f}")
    env.close()



if __name__ == "__main__":
    main()


