#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
import collections
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import gymnasium as gym



#---------------------------------------------------------------------------------------
# 经验回放池
#---------------------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)        # 队列,先进先出

    def add(self, state, action, reward, next_state, done):     # 将数据加入buffer中
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):                               # 从buffer中采样数据
        transitions = random.sample(self.buffer, batch_size)    # 随机采样batch_size个数据
        state, action, reward, next_state, done = zip(*transitions)  # 解压数据
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):                                           # 返回buffer中数据的个数
        return len(self.buffer)



#---------------------------------------------------------------------------------------
# 定义Q网络(1层隐藏层)
#---------------------------------------------------------------------------------------
class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)



#---------------------------------------------------------------------------------------
# DQN算法
#---------------------------------------------------------------------------------------
class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)         # Q网络
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)  # 目标Q网络
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)  # Adam优化器

        self.gamma = gamma                                  # 折扣因子
        self.epsilon = epsilon                              # epsilon-贪婪策略
        self.target_update = target_update                  # 目标网络更新频率
        self.device = device                                # 设备类型

        self.count = 0                                      # 计数器,用于判断是否更新目标网络 

    # epsilon-贪婪算法
    def take_action(self, state):                           # 选取下一步的操作,具体实现为epsilon-贪婪
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)     # 随机采样1个动作
        else:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)  # 转换为tensor
            action = self.q_net(state_tensor).argmax().item()     # 使用动作价值最大的那个动作
        return action

    # 单步策略更新
    def update(self, transition_dict):                       # 更新Q网络
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.long).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)    # Q(s,a)值

        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)  # max Q(s',a')值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        
        self.optimizer.zero_grad()                              # 梯度清零
        dqn_loss.backward()                                     # 反向传播
        self.optimizer.step()                                   # 更新参数

        if self.count % self.target_update == 0:                  # 每target_update步更新一次目标网络
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络参数
        self.count += 1

    # 保存策略
    def save(self, path):
        torch.save(self.q_net.state_dict(), path)
        print(f"模型已保存到: {path}")

    # 加载策略
    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.q_net.eval()
        print(f"模型已加载自: {path}")



#---------------------------------------------------------------------------------------
# 离策略训练智能体
#---------------------------------------------------------------------------------------
def train_off_policy_agent(env, agent, num_episodes, replay_buffer, batch_size):
    return_list = []                                                        # 记录每1条序列的回报
    for i in range(10):                                                     # 显示10个进度条
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:         # tqdm的进度条功能
            for i_episode in range(int(num_episodes / 10)):                 # 每个进度条的序列数
                episode_return = 0                                          # 单条轨迹的回报值
                state, _ = env.reset()                                      # 重置环境
                done = False
                while not done:                                             # 本次轨迹未结束
                    action = agent.take_action(state)                       # 根据当前状态选择动作
                    next_state, reward, terminated, truncated, _ = env.step(action)     # 执行动作后返回环境状态等信息
                    done = terminated or truncated
                    
                    replay_buffer.add(state, action, reward, next_state, done)  # 将数据加入经验回放池
                    state = next_state
                    episode_return += reward                                # 累加本动作奖励至本轨迹回报
                    
                    if replay_buffer.size() > 500:                          # 当经验回放池中数据量大于500时,才进行训练
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)         # 从经验回放池中采样数据
                        transition_dict = {'states': b_s, 'actions': b_a, 
                            'rewards': b_r, 'next_states': b_ns, 'dones': b_d}
                        agent.update(transition_dict)                       # 更新Q网络

                return_list.append(episode_return)                          # 插入本轨迹的回报值
                if (i_episode + 1) % 10 == 0:                               # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({'episode':'%d' % (int(num_episodes / 10) * i + i_episode + 1), 
                        'return':'%.3f' % np.mean(return_list[-10:]) })
                pbar.update(1)
    return return_list                                                      # 返回: 每条序列的回报



#---------------------------------------------------------------------------------------
# 主函数
#---------------------------------------------------------------------------------------
def main():
    env = gym.make('CartPole-v1')                           # 创建环境
    env.reset(seed=0)                                       # 环境随机种子
    torch.manual_seed(0)

    replay_buffer = ReplayBuffer(10000)                     # 创建经验回放池

    state_dim = env.observation_space.shape[0]              # 状态维度
    hidden_dim = 128                                        # 隐藏层维度
    action_dim = env.action_space.n                         # 动作维度
    learning_rate = 1e-3                                    # 学习率
    gamma = 0.98                                            # 折扣因子
    epsilon = 0.1                                           # epsilon-贪婪策略中的参数
    target_update = 100                                     # 目标网络更新频率
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = DQN(state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device)

    batch_size = 64
    num_episodes = 1000
    return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, batch_size)

    SCRIPT_DIR = Path(__file__).resolve().parent            # 获取当前脚本所在目录
    model_path = SCRIPT_DIR / "dqn_cartpole.pth"            # 拼接出完整路径
    agent.save(model_path)                                  # 传入正确路径

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format('CartPole-v1'))
    plt.show()



if __name__ == "__main__":
    main()




