import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from cliffwalking import CliffWalkingEnv



#---------------------------------------------------------------------------------------
# n步Sarsa算法
# 核心理解: 在线与环境交互,在策略学习,只是将1步回报扩展成n步回报
#---------------------------------------------------------------------------------------
class nstep_Sarsa:
    def __init__(self, n, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])        # 初始化Q(s,a)表格
        self.n = n                                              # 采用 n 步 Sarsa 算法
        self.epsilon = epsilon                                  # epsilon-贪婪策略中的参数
        self.alpha = alpha                                      # 学习率
        self.gamma = gamma                                      # 折扣因子
        self.n_action = n_action                                # 动作个数
        self.state_list = []                                    # 保存n步的状态
        self.action_list = []                                   # 保存n步的动作
        self.reward_list = []                                   # 保存n步的奖励

    # epsilon-贪婪算法
    def take_action(self, state):                               # 选取下一步的操作,具体实现为epsilon-贪婪
        if np.random.random() < self.epsilon:                   # 小于epsilon值
            action = np.random.randint(self.n_action)           # 随机采样1个动作
        else:                                                   # 否则
            action = np.argmax(self.Q_table[state])             # 使用动作价值最大的那个动作
        return action                                           # 返回动作

    # 选择最佳动作
    def best_action(self, state):                               # 打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):                          # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    # 计算状态价值函数
    def update(self, s0, a0, r, s1, a1, done):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)                              # 每个时间步先缓存s0, a0, r
        if len(self.state_list) == self.n:                      # 当缓存长度达到 n 时
            G = self.Q_table[s1, a1]                            # 得到Q{s_{t+n}, a_{t+n}}
            for i in reversed(range(self.n)):                   # 代码从Q(s1,a1)开始倒序累加奖励构造G
                G = self.gamma * G + self.reward_list[i]        # 向前计算每一步的回报
                if done and i > 0:                               # 达到终止状态,最后几步虽然不够n步,也将进行更新
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
            s = self.state_list.pop(0)                          # 将需要更新的状态动作动列表中删除,下次不用更新
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a]) # 常规 n 步更新
        if done:                                                # 若到达终止状态,将开始下一序列,则列表清空
            self.state_list = []
            self.action_list = []
            self.reward_list = []



#---------------------------------------------------------------------------------------
# 在策略训练智能体
#---------------------------------------------------------------------------------------
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []                                        # 记录每1条序列的回报
    for i in range(10):                                     # 显示10个进度条
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:         # tqdm的进度条功能
            for i_episode in range(int(num_episodes / 10)): # 每个进度条的序列数
                episode_return = 0                          # 单条轨迹的回报值
                state = env.reset()                         # 重置环境
                action = agent.take_action(state)           # 根据当前状态选择动作
                done = False
                while not done:                             # 本次轨迹未结束
                    next_state, reward, done = env.step(action) # 执行动作后返回环境状态等信息
                    next_action = agent.take_action(next_state) # 根据下一状态获取下一动作
                    agent.update(state, action, reward, next_state, next_action, done)    # 计算本状态的价值
                    episode_return += reward                # 累加本动作奖励至本轨迹回报
                    state = next_state
                    action = next_action
                return_list.append(episode_return)          # 插入本轨迹的回报值
                if (i_episode + 1) % 10 == 0:               # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({'episode':'%d' % (int(num_episodes / 10) * i + i_episode + 1), 
                        'return':'%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list                                      # 返回: 每条序列的回报



#---------------------------------------------------------------------------------------
# 打印agent决策过程/可视化策略地图
# agent: 智能体;        env: 环境;      action_meaning: 动作编号到箭头的映射
# disaster: 悬崖状态                    end: 终点状态
#---------------------------------------------------------------------------------------
def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:          # 悬崖就打印****
                print('****', end=' ')
            elif (i * env.ncol + j) in end:             # 终点就打印EEEE
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()



#---------------------------------------------------------------------------------------
# 主函数
#---------------------------------------------------------------------------------------
def main():
    env = CliffWalkingEnv(12, 4)
    np.random.seed(0)

    n_step = 5                                          # 5步Sarsa算法
    epsilon = 0.1                                       # epsilon-贪婪策略中的参数
    alpha = 0.1                                         # 学习率
    gamma = 0.9                                         # 折扣因子
    agent = nstep_Sarsa(n_step, 12, 4, epsilon, alpha, gamma)

    num_episodes = 500                                  # 智能体在环境中运行的序列数
    return_list = train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('5-step Sarsa on {}'.format('Cliff Walking'))
    plt.show()

    action_meaning = ['^', 'v', '<', '>']
    print('5步Sarsa算法最终收敛得到的策略为:')
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])



if __name__ == "__main__":
    main()



