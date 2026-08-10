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
# 策略网络(离散动作空间)
# 作用: 根据当前状态s输出1个具体的动作值a,并且改值会被限制在环境允许范围内
#---------------------------------------------------------------------------------------
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):      # 3参数(状态维度, 隐藏数量, 动作维度)
        super(PolicyNet, self).__init__()                       # 调用父类构造函数
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)       # 输入状态 → 隐藏层
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)      # 隐藏层 → 均值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)



#---------------------------------------------------------------------------------------
# 价值网络 
# 作用: 估计给定状态s下执行某动作a所能获得的未来累计奖励的期望值(即Q值)
#---------------------------------------------------------------------------------------
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):      # 3参数(状态维度, 隐藏数量, 动作维度)
        super(QValueNet, self).__init__()                       # 调用父类构造函数
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)       # 状态+动作 -> 隐藏层1
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)      # 隐藏层1 -> 隐藏层

    def forward(self, x):
        x = F.relu(self.fc1(x))                                 # ReLU激活函数
        return self.fc2(x)


    
#---------------------------------------------------------------------------------------
# SAC(soft actor critic)
# actor(策略网络):根据状态输出确定性的动作
# critic(价值网络):评估在给定状态下采取某个动作的价值
# 目标网络(Target Networks):用于稳定训练,通过软更新缓慢跟踪在线网络
# 状态维度, 隐藏层数, 动作维度, 动作最大值, 学习率, 目标熵, 软更新系数, 折扣因子, 设备
#---------------------------------------------------------------------------------------
class SAC:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, 
        alpha_lr, target_entropy, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, device=device)
        self.log_alpha.requires_grad = True                 # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy                # 熵正则项系数
        self.gamma = gamma                                  # 折扣因子
        self.tau = tau                                      # 软更新参数
        self.device = device                                # 设备


    # 根据当前状态选择1个动作
    def take_action(self, state):
        state = torch.as_tensor(np.asarray(state), dtype=torch.float, device=self.device).unsqueeze(0)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    # 计算目标Q值
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    # 更新目标网络
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    # 主网络循环更新
    def update(self, transition_dict):
        # 类型转换
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.as_tensor(transition_dict['actions'], dtype=torch.long, device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)

        # 计算损失
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))

        # 更新critic网络
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新熵正则项系数
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # 更新目标critic网络
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    # 保存模型参数
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path)
        print(f"模型保存到:{path}")

    # 加载模型参数
    def load(self, path):
        path = Path(path)
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        self.actor.eval()
        print(f"模型已加载自:{path}")





#===============================================================================
# 主函数
#===============================================================================
def main():
# 加载环境与随机种子
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    env.reset(seed=0)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # 构建SAC算法智能体
    state_dim = env.observation_space.shape[0]          # 状态维度
    hidden_dim = 128                                    # 隐藏层维度
    action_dim = env.action_space.n                     # 动作维度
    actor_lr = 0.001                                    # actor学习率
    critic_lr = 0.01                                    # critic学习率
    alpha_lr = 0.01                                     # 学习率
    target_entropy = -1                                 # 熵正则项系数
    tau = 0.005                                         # 软更新参数
    gamma = 0.98                                        # 折扣因子
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = SAC(state_dim, hidden_dim, action_dim, actor_lr,
        critic_lr, alpha_lr, target_entropy, tau, gamma, device)

    # 进行离线训练
    num_episodes = 200                                  # 训练总轮次
    replay_buffer = ReplayBuffer(10000)                 # 缓存区大小
    minimal_size = 500                                  # 最小样本条数
    batch_size = 64                                     # 随机采样单批大小
    return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

    # 保存训练好的模型(最后1轮的)
    model_path = Path(__file__).resolve().parent / "sac_cartpole.pth"
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













