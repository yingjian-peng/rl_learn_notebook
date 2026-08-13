import numpy as np
import matplotlib.pyplot as plt
from bernoulli_bandit import BernoulliBandit, Solver



#---------------------------------------------------------------------------------------
# epsilon贪婪算法(继承Solver类)
#---------------------------------------------------------------------------------------
class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)             # 父类初始化多臂老虎机对象
        self.epsilon = epsilon                                  # 采样概率
        self.estimates = np.array([init_prob] * self.bandit.K)  # 初始化拉动所有拉杆的期望奖励估值

    
    def run_one_step(self):
        if np.random.random() < self.epsilon:                   # 当前概率 < 采样概率
            k = np.random.randint(0, self.bandit.K)             # 随机选择1根拉杆
        else:
            k = np.argmax(self.estimates)                       # 选择期望奖励估值最大的拉杆
        
        r = self.bandit.step(k)                                 # 得到本次动作的奖励
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k



#---------------------------------------------------------------------------------------
# 累积懊悔随时间变化的图像
# solvers: 列表,列表中的每个元素是1种特定的策略
# solver_names: 列表,存储每个策略的名称
#---------------------------------------------------------------------------------------
def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' %solvers[0].bandit.K)
    plt.legend()
    plt.show()



#---------------------------------------------------------------------------------------
# epsilon值随时间衰减的rpsilion-贪婪算法
#---------------------------------------------------------------------------------------
class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)  # 初始化拉动所有拉杆的期望奖励估值
        self.total_count = 0                                    # 训练计数器

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:           # epsilon 值随时间衰减
            k = np.random.randint(0, self.bandit.K)                  # 随机选择1根拉杆 
        else:
            k = np.argmax(self.estimates)                       # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)                                 # 得到本次动作的奖励
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k



#---------------------------------------------------------------------------------------
# 主函数
#---------------------------------------------------------------------------------------
def main():
    # # (1)固定采样概率为0.01
    # bandit_10_arm = BernoulliBandit(10)
    # epsilin_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    # epsilin_greedy_solver.run(5000)

    # print('epsilon-贪婪算法的累积懊悔为:', epsilin_greedy_solver.regret)
    # plot_results([epsilin_greedy_solver], ['EpsilonGreedy'])


    # # (2)不用采样概率
    # epsilons = [1e-4, 0.01, 0.25, 0.5]
    # bandit_10_arm = BernoulliBandit(10)
    # epsilons_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons]
    # epsilion_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
    # for solver in epsilons_greedy_solver_list:
    #     solver.run(500)
    
    # plot_results(epsilons_greedy_solver_list, epsilion_greedy_solver_names)


    # (3)采样概率随时间衰减
    bandit_10_arm = BernoulliBandit(10)
    decaying_epsilin_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
    decaying_epsilin_greedy_solver.run(5000)

    print('epsilon值衰减的贪婪算法的累积懊悔为:', decaying_epsilin_greedy_solver.regret)
    plot_results([decaying_epsilin_greedy_solver], ['DecayingEpsilonGreedy'])



if __name__ == "__main__":
    main()





