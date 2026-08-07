import random
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path
import collections

import matplotlib.pyplot as plt



#---------------------------------------------------------------------------------------
# 策略网络 
# 作用: 根据当前状态s输出1个具体的动作值a,并且改值会被限制在环境允许范围内
#---------------------------------------------------------------------------------------
class PolicyNet(torch.nn.Module):                           # 定义继承(torch.nn.Module)的类,即PyTorch中的神经网络模块
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):    # 4参数(状态维度, 隐藏数量, 动作维度, 最大动作值)
        super(PolicyNet, self).__init__()                   # 调用父类构造函数
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)   # 状态维度到隐藏维度映射
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # 隐藏维度到动作维度映射
        self.action_bound = action_bound                    # action_bound 是环境接受动作最大值(动作边界)

    def forward(self, x):                                   # 前向传播过程
        x = F.relu(self.fc1(x))                             # ReLU激活函数
        return torch.tanh(self.fc2(x)) * self.action_bound  # 正切函数限制在[-1,1]并进行缩放



#---------------------------------------------------------------------------------------
# Q值网络 - Critic
# 作用: 估计给定状态s下执行某动作a所能获得的未来累计奖励的期望值(即Q值)
#---------------------------------------------------------------------------------------
class QValueNet(torch.nn.Module):                           # 定义继承(torch.nn.Module)的类
    def __init__(self, state_dim, hidden_dim, action_dim):  # 3参数(状态维度, 隐藏数量, 动作维度)
        super(QValueNet, self).__init__()                   # 调用父类构造函数
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)  # 状态,动作拼接在一起作为输入
        self.fc2 = torch.nn.Linear(hidden_dim, 1)           # 最后输出为单值,即Q值

    def forward(self, x, a):                                # 前向传播过程
        cat = torch.cat([x, a], dim=1)                      # 先拼接状态s和动作a
        x = F.relu(self.fc1(cat))                           # ReLU激活函数
        return self.fc2(x)                                  # 输出标量Q值,形状[batch, 1]



#---------------------------------------------------------------------------------------
# 两隐藏层的全连接网络 - 通用神经网络模块
# 作用: fc1 → 激活 → fc2 → 激活 → fc3 → out_fn
#---------------------------------------------------------------------------------------
class TwoLayerFC(torch.nn.Module):
    # 5参数(输入维度, 输出维度, 隐藏层神经元数量, 隐藏层激活函数, 输出层额外应用函数)
    def __init__(self, num_in, num_out, hidden_dim, activation=F.relu, out_fn=lambda x:x):
        super().__init__()
        self.fc1 = nn.Linear(num_in, hidden_dim)            # 输入层 --> 隐藏层1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)        # 隐藏层1 --> 隐藏层2
        self.fc3 = nn.Linear(hidden_dim, num_out)           # 隐藏层2 --> 输出层

        self.activation = activation                        # 隐藏层激活函数
        self.out_fn = out_fn                                # 输出层后处理函数

    def forward(self, x):                                   # 前向传播过程
        x = self.activation(self.fc1(x))                    # 进行1次 输入->隐藏层1
        x = self.activation(self.fc2(x))                    # 进行1次 隐藏层2->隐藏层1
        x = self.out_fn(self.fc3(x))                        # 输出层后处理(当前原样映射)
        return x



#---------------------------------------------------------------------------------------
# DDPG(Deep Deterministic Policy Gradient)
# Actor(策略网络):根据状态输出确定性的动作。
# Critic(Q 值网络):评估在给定状态下采取某个动作的价值
# 目标网络(Target Networks):用于稳定训练,通过软更新缓慢跟踪在线网络
# 状态维度, 隐藏层数, 动作维度, 动作最大值, 探索噪声标准差, 学习率, 软更新系数, 折扣因子, 设备
#---------------------------------------------------------------------------------------
class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, 
        critic_lr, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)  # actor网络
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)               # critic网络

        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)   # 目标actor网络
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)        # 目标critic网络

        self.target_actor.load_state_dict(self.actor.state_dict())              # 将在线网络参数直接复制给目标网络
        self.target_critic.load_state_dict(self.critic.state_dict())            # 将在线网络参数直接复制给目标网络

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)       # Adam优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)    # Adam优化器

        self.action_dim = action_dim                                            # 动作维度
        self.sigma = sigma                                                      # 探索噪声标准差
        self.tau = tau                                                          # 软更新系数
        self.gamma = gamma                                                      # 折扣因子
        self.device = device                                                    # 设备


    # 根据当前状态选择1个动作(带探索噪声)
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).item()                                       # 通过actor获得动作,维度[1, state_dim]
        action = action + self.sigma * np.random.randn(self.action_dim)         # 添加高斯噪声(标准差为 sigma)
        return action                                                           # 返回动作(标量/数组)
    

    # 软更新(将目标网络的参数向在线网络缓慢靠拢)
    def soft_update(self, net, target_net):                                     # 在线网络, 目标网络
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # θtarget <- τ * θonline ​+ (1−τ) * θtarget​
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
    

    # 更新主流程
    def update(self, transition_dict):
        # 数据转为张量并移至指定设备
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)
        
        # 计算在线 critic 损失
        next_q_values = self.target_critic(next_states, self.target_actor(next_states)) # 下一时间Q值 = 目标critic(下一时间状态, 目标actor(下一时间状态))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)                  # TD 目标
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))   # critic 损失 = 均方误差(当前critic, TD目标)

        # 更新 critic 网络参数
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 计算在线 actor 损失
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))               # 在线 actor 损失 = -(在线critic价值)
        
        # 更新 actor 网络参数
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标actor和critic网络参数
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)


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





class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)



def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)

                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
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
    random.seed(0)
    np.random.seed(0)

    state_dim = env.observation_space.shape[0]
    hidden_dim = 64                                 # 隐藏层数
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]         # 动作最大值
    sigma = 0.01                                    # 高斯噪声标准差
    actor_lr = 0.0003
    critic_lr = 0.003
    tau = 0.005                                     # 软更新参数
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)

    num_episodes = 200
    replay_buffer = ReplayBuffer(10000)
    minimal_size = 1000
    batch_size = 64
    return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)
    
    model_path = Path(__file__).resolve().parent / "ddpg_pendulum.pth"
    agent.save(model_path)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list,return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG on {}'.format(env_name))
    plt.show()





if __name__ == "__main__":
    main()


















