import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x),dim=1)


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr = learning_rate) # 使用 Adam 优化器
        self.gamma = gamma                              # 折扣因子
        self.device = device

    def take_action(self, state):                       # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):     # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = - log_prob * G                       # 每一步的损失函数
            loss.backward()                             # 反向传播计算梯度
        self.optimizer.step()                           # 梯度下降

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"模型已保存到: {path}")

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval()                          # 切换到评估模式
        print(f"模型已加载自: {path}")



env = gym.make('CartPole-v1', render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = REINFORCE(state_dim, 128, action_dim, 1e-3, 0.98, torch.device("cpu"))
agent.load("reinforce_cartpole.pth")


state, info = env.reset(seed=0)
done = False
total_reward = 0
while not done:
    env.render()
    action = agent.take_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = next_state
    total_reward += reward

print(f"总回报: {total_reward}")
env.close()

