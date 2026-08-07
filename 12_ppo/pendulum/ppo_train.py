#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



#===============================================================================
# 策略网络
#===============================================================================
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std                                          # 均值 标准差


#===============================================================================
# 价值网络
#===============================================================================
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)                          # 当前状态价值估计,标量



#===============================================================================
# PPO算法
#===============================================================================
class PPOContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)                             # 输出连续动作的高斯分布参数
        action_dist = torch.distributions.Normal(mu, sigma)       # 
        action = action_dist.sample()                           # 根据概率分布进行1次随机采样,返回1个张量,形状为(1,)
        return [action.item()]                                  # Pendulum 期望形状为 (1,) 的连续动作


    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        rewards = (rewards + 8.0) / 8.0  # 对奖励进行修改,方便训练
        
        # 计算TD target 和 TD error
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)   # TD_target = r + !Qn+1(s,a)
        td_delta = td_target - self.critic(states)                                  # TD_error = TD_target - Qn(s,a)
        
        # 使用 均方误差损失(MSE) 计算 优势函数(advantage)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(actions)
        
        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()                                           # 价值网络清零梯度
            
            actor_loss.backward()
            critic_loss.backward()                                                      # 价值网络反向传播,计算梯度
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()                                                # 价值网络梯度更新参数
            

    # 保存训练的模型
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path)
        print(f"模型已保存到: {path}")

    # 加载训练的模型
    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        self.actor.eval()
        print(f"模型已加载自: {path}")



#===============================================================================
# 优势函数(广义优势估计)
#===============================================================================
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()                                # TD_error
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:                                        # 逆序遍历TD误差（从后往前）
        advantage = gamma * lmbda * advantage + delta                   # GAE递推公式,逆向递推
        advantage_list.append(advantage)
    advantage_list.reverse()                                            # 恢复正向顺序
    return torch.tensor(np.array(advantage_list), dtype=torch.float)    # 转回Tensor


#===============================================================================
# 在线策略训练智能体
#===============================================================================
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    episodes_per_iteration = int(num_episodes / 10)

    for i in range(10):
        with tqdm(total=episodes_per_iteration, desc='Iteration %d' % i) as pbar:
            for i_episode in range(episodes_per_iteration):
                global_episode = i * episodes_per_iteration + i_episode + 1
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, info = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)

                    state = next_state
                    episode_return += reward

                return_list.append(episode_return)

                agent.update(transition_dict)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % global_episode, 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list



#===============================================================================
# 主函数
#===============================================================================
def main():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)

    env.reset(seed=0)
    torch.manual_seed(0)
    env.action_space.seed(0)

    state_dim = env.observation_space.shape[0]
    hidden_dim = 128                                # 隐藏层数
    action_dim = env.action_space.shape[0]
    actor_lr = 0.0001
    critic_lr = 0.005
    lmbda = 0.9
    epochs = 10
    eps = 0.2
    gamma = 0.9
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

    num_episodes = 2000
    return_list = train_on_policy_agent(env, agent, num_episodes)
    
    model_path = Path(__file__).resolve().parent / "ppo_pendulum.pth"
    agent.save(model_path)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list,return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()










if __name__ == "__main__":
    main()




