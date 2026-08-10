import random
import gymnasium as gym
import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions import Normal

from pathlib import Path
import collections

from tqdm import tqdm
import matplotlib.pyplot as plt



#---------------------------------------------------------------------------------------
# 经验回放缓冲区(Replay Buffer)
#---------------------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):                               # 类定义与初始化
        self.buffer = collections.deque(maxlen=capacity)        # 创建双端队列,并设置最大长度

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size):                               # 从缓冲区随机采样x条经验
        transitions = random.sample(self.buffer, batch_size)    # 从缓存区无放回随机选取x个元素
        state, action, reward, next_state, done = zip(*transitions) # 解压
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)



#---------------------------------------------------------------------------------------
# 离线策略训练智能体
# 智能体, 最大轮数, 缓存区, 开始训练前缓存区最小经验数量, 每次更新时从缓存区采样的批大小
#---------------------------------------------------------------------------------------
def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []                                        # 每回合的总回报
    for i in range(10):                                     # 外层循环
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):   # 内层循环
                episode_return = 0                          # 累计本回合奖励
                state, _ = env.reset()                      # 重置环境
                done = False
                while not done:
                    action = agent.take_action(state)       # actor根据当前状态选择动作a

                    next_state, reward, terminated, truncated, _ = env.step(action) # 执行动作
                    done = terminated or truncated
                    
                    replay_buffer.add(state, action, reward, next_state, done)  # 存入经验回放缓冲区
                    state = next_state
                    episode_return += reward                # 累加奖励
                    if replay_buffer.size() > minimal_size: # 当经验数量超过size时开始训练
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size) # 采样1个批次经验
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)       # 采样结果更新网络
                return_list.append(episode_return)          # 记录本回合回报
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list












#---------------------------------------------------------------------------------------
# 策略网络(连续动作空间)
# 作用: 根据当前状态s输出1个具体的动作值a,并且改值会被限制在环境允许范围内
#---------------------------------------------------------------------------------------
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):    # 4参数(状态维度, 隐藏数量, 动作维度, 最大动作值)
        super(PolicyNetContinuous, self).__init__()                         # 调用父类构造函数
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)                   # 输入状态 → 隐藏层
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)                # 隐藏层 → 均值
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)               # 隐藏层 → 标准差(对数形式)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)                                                  # 均值
        std = F.softplus(self.fc_std(x)) + 1e-5                             # 标准差(始终为正且加小常数)
        dist = Normal(mu, std)                                              # 创建正态分布(高斯分布)
        normal_sample = dist.rsample()                                      # rasmple()重参数化采样
        log_prob = dist.log_prob(normal_sample)                             # 高斯分布下采样值的对数概率密度
        action = torch.tanh(normal_sample)                                  # 将采样值映射到(-1, 1),使动作有界
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)           # 概率校正
        action = action * self.action_bound                                 # 动作线性缩放
        return action, log_prob                                             # 动作,对数概率密度



#---------------------------------------------------------------------------------------
# 价值网络 
# 作用: 估计给定状态s下执行某动作a所能获得的未来累计奖励的期望值(即Q值)
#---------------------------------------------------------------------------------------
class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):                  # 3参数(状态维度, 隐藏数量, 动作维度)
        super(QValueNetContinuous, self).__init__()                         # 调用父类构造函数
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)      # 状态+动作 -> 隐藏层1
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)                  # 隐藏层1 -> 隐藏层2
        self.fc_out = torch.nn.Linear(hidden_dim, 1)                        # 单值

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)                                      # 先拼接状态s和动作a
        x = F.relu(self.fc1(cat))                                           # ReLU激活函数
        x = F.relu(self.fc2(x))                                             # ReLU激活函数
        return self.fc_out(x)                                               # 输出标量Q值,形状[batch, 1]


    
#---------------------------------------------------------------------------------------
# SAC(soft actor critic)
# actor(策略网络):根据状态输出确定性的动作
# critic(价值网络):评估在给定状态下采取某个动作的价值
# 目标网络(Target Networks):用于稳定训练,通过软更新缓慢跟踪在线网络
# 状态维度, 隐藏层数, 动作维度, 动作最大值, 学习率, 目标熵, 软更新系数, 折扣因子, 设备
#---------------------------------------------------------------------------------------
class SACContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, 
        alpha_lr, target_entropy, tau, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, device=device)
        self.log_alpha.requires_grad = True         # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy        # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device


    # 根据当前状态选择1个动作
    def take_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        action = self.actor(state)[0]
        return [action.item()]

    def calc_target(self, rewards, next_states, dones):         # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)


    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)

        reward = (rewards + 8.0) / 8.0

        td_target = self.calc_target(reward, next_states, dones)
        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), td_target.detach()))

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path)
        print(f"模型保存到:{path}")

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        self.actor.eval()
        print("模型已加载自:{path}")







#===============================================================================
# 主函数
#===============================================================================
def main():
# 加载环境与随机种子
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)

    env.reset(seed=0)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound, actor_lr,
        critic_lr, alpha_lr, target_entropy, tau, gamma, device)

    # 进行离线训练
    num_episodes = 100
    replay_buffer = ReplayBuffer(100000)
    minimal_size = 1000
    batch_size = 64
    return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

    # 保存训练好的模型(最后1轮的)
    model_path = Path(__file__).resolve().parent / "sac_pendulum.pth"
    agent.save(model_path)

    # 绘制训练回报图像
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.show()




if __name__ == "__main__":
    main()















