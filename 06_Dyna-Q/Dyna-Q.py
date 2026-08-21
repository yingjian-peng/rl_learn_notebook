import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import time

from cliffwalking import CliffWalkingEnv



#---------------------------------------------------------------------------------------
# Dyna-Q算法
#---------------------------------------------------------------------------------------
class DynaQ:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_planning, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])        # 初始化Q(s,a)表格
        self.epsilon = epsilon                                  # epsilon-贪婪策略中的参数
        self.alpha = alpha                                      # 学习率
        self.gamma = gamma                                      # 折扣因子
        self.n_planning = n_planning                            # 执行 Q-planning 的次数
        self.n_action = n_action                                # 动作个数

        self.model = dict()                                     # 环境模型

    # epsilon-贪婪算法
    def take_action(self, state):                               # 选取下一步的操作,具体实现为epsilon-贪婪
        if np.random.random() < self.epsilon:                   # 小于epsilon值
            action = np.random.randint(self.n_action)           # 随机采样1个动作
        else:                                                   # 否则
            action = np.argmax(self.Q_table[state])             # 使用动作价值最大的那个动作
        return action                                           # 返回动作

    # Q-learning算法的TD计算
    def q_learning(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    # 单步策略更新
    def update(self, s0, a0, r, s1):
        self.q_learning(s0, a0, r, s1)                          # 智能体与真实环境交互
        self.model[(s0, a0)] = r, s1                            # 将数据添加到模型中
        for _ in range(self.n_planning):                        # Q-planning循环
            (s, a), (r, s_) = random.choice(list(self.model.items()))   # 随机选择曾经遇到过的状态动作对
            self.q_learning(s, a, r, s_)                        # 智能体与环境模型交互



#---------------------------------------------------------------------------------------
# 离策略训练智能体
#---------------------------------------------------------------------------------------
def train_off_policy_agent(env, agent, num_episodes):
    return_list = []                                        # 记录每1条序列的回报
    for i in range(10):                                     # 显示10个进度条
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:         # tqdm的进度条功能
            for i_episode in range(int(num_episodes / 10)): # 每个进度条的序列数
                episode_return = 0                          # 单条轨迹的回报值
                state = env.reset()                         # 重置环境
                done = False
                while not done:                             # 本次轨迹未结束
                    action = agent.take_action(state)       # 根据当前状态选择动作
                    next_state, reward, done = env.step(action) # 执行动作后返回环境状态等信息
                    agent.update(state, action, reward, next_state)    # 计算本状态的价值
                    episode_return += reward                # 累加本动作奖励至本轨迹回报
                    state = next_state
                return_list.append(episode_return)          # 插入本轨迹的回报值
                if (i_episode + 1) % 10 == 0:               # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({'episode':'%d' % (int(num_episodes / 10) * i + i_episode + 1), 
                        'return':'%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list                                      # 返回: 每条序列的回报



#---------------------------------------------------------------------------------------
# 主函数
#---------------------------------------------------------------------------------------
def main():
    np.random.seed(0)
    random.seed(0)
    
    n_planning_list = [0, 2, 20]
    for n_planning in n_planning_list:
        time.sleep(0.5)
        print('Q-learning步数为:%d' %n_planning)

        env = CliffWalkingEnv(12, 4)
        epsilon = 0.01
        alpha = 0.1
        gamma = 0.9
        agent = DynaQ(12, 4, epsilon, alpha, gamma, n_planning)
        
        num_episodes = 300                                      # 智能体在环境中运行多少条序列
        return_list = train_off_policy_agent(env, agent, num_episodes)

        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list, label=str(n_planning) + 'planning steps')
    
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Dyna-Q on {}'.format('Cliff Walking'))
    plt.show()



if __name__ == "__main__":
    main()






