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
# TRPO算法
#===============================================================================
class TRPOContinuous:
    def __init__(self, hidden_dim, state_space, action_space, lmbda, kl_constraint, alpha, critic_lr, gamma, device):
        state_dim = state_space.shape[0]
        action_dim = action_space.shape[0]

        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.lmbda = lmbda                                      # GAE参数
        self.kl_constraint = kl_constraint                      # kl距离最大限制
        self.alpha = alpha                                      # 线性搜索参数
        self.device = device
        

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        mu, std = self.actor(state)                             # 输出连续动作的高斯分布参数
        action_dist = torch.distributions.Normal(mu, std)       # 
        action = action_dist.sample()                           # 根据概率分布进行1次随机采样,返回1个张量,形状为(1,)
        return [action.item()]                                  # Pendulum 期望形状为 (1,) 的连续动作
    

    # 计算黑塞矩阵和一个向量的乘积
    def hessian_matrix_vector_product(self, states, old_action_dists, vector, damping=0.1):
        mu, std = self.actor(states)
        new_action_dists = torch.distributions.Normal(mu, std)
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists)) # 计算平均 KL 距离
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.contiguous().view(-1) for grad in grad2])

        return grad2_vector + damping * vector


    # 共轭梯度法求解方程(求解线性系统 Hx=g)
    def conjugate_gradient(self, grad, states, old_action_dists): 
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):                                                     # 共轭梯度主循环
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = rdotr / (torch.dot(p, Hp) + 1e-8)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    # 计算策略目标
    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor): 
        mu, std = actor(states)
        action_dists = torch.distributions.Normal(mu, std)
        log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)                            # e**(当前动作概率 - 上一动作概率)
        return torch.mean(ratio * advantage)                                    # 平均值(概率比值 * 优势函数)


    # 线性搜索
    def line_search(self, states, actions, advantage, old_log_probs, old_action_dists, max_vec):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        for i in range(15):                                                     # 线性搜索主循环
            coef = self.alpha ** i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
            mu, std = new_actor(states)
            new_action_dists = torch.distributions.Normal(mu, std)
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, new_actor)
            if new_obj.item() > old_obj.item() and kl_div.item() < self.kl_constraint:
                return new_para
        return old_para


    # 更新策略函数
    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())         # 计算代理函数关于 actor 网络所有参数的梯度
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()            # 获取梯度值g
        descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists) # 用共轭梯度法计算 x = H^(-1)g
        Hd = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)        # 黑塞矩阵向量积
        max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8))   # 计算缩放因子
        new_para = self.line_search(states, actions, advantage, old_log_probs, old_action_dists, descent_direction * max_coef) # 线性搜索
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters()) # 用线性搜索后的参数更新策略

    # 新策略参数 = 就策略参数 + KLK范围内的更新
    #----------------------------------------------------------------------------------------------
    # 更新policy网络参数: TRPO 的特殊流程更新
    # 更新critic网络参数: 普通 优化器(Adam) 和 均方误差损失(MSE) 更新
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
        
        # 记录策略网络数据
        mu, std = self.actor(states)
        old_action_dists = torch.distributions.Normal(mu.detach(), std.detach())     # 创建离散分类分布
        old_log_probs = old_action_dists.log_prob(actions)
        
        # 价值网络计算与更新
        critic_loss = F.mse_loss(self.critic(states), td_target.detach())           # 计算价值网络的均方误差(MSE)
        self.critic_optimizer.zero_grad()                                           # 价值网络清零梯度
        critic_loss.backward()                                                      # 价值网络反向传播,计算梯度
        self.critic_optimizer.step()                                                # 价值网络梯度更新参数
        
        # 策略网络计算与更新
        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)

    # 保存训练的模型
    def save(self, path):
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

    hidden_dim = 128                                # 隐藏层数
    lmbda = 0.90                                    # 工程中一般就用这个值(控制bias与variance的折中)
    kl_constraint = 0.00005                         # KL散度参数
    alpha = 0.5                                     # 线性搜索缩放系数
    critic_lr = 1e-2                                # 价值网络学习率
    gamma = 0.90                                    # 折扣因子
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = TRPOContinuous(hidden_dim, env.observation_space, env.action_space, lmbda, kl_constraint, alpha, critic_lr, gamma, device)

    model_path = Path(__file__).resolve().parent / "trpo_pendulum.pth"
    num_episodes = 4000
    return_list = train_on_policy_agent(env, agent, num_episodes)
    agent.save(model_path)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list,return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.show()










if __name__ == "__main__":
    main()




