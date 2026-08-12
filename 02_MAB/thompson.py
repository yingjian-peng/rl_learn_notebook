import numpy as np
import matplotlib.pyplot as plt
from bernoulli_bandit import BernoulliBandit, Solver



#---------------------------------------------------------------------------------------
# 汤普森采样算法
#---------------------------------------------------------------------------------------
class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)                # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)                # 列表,表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)      # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)                          # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)

        self._a[k] += r                                 # 更新beta分布的第1个参数
        self._b[k] += (1-r)                             # 更新beta分布的第2个参数
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
# 主函数
#---------------------------------------------------------------------------------------
def main():
    bandit_10_arm = BernoulliBandit(10)
    thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
    thompson_sampling_solver.run(5000)

    print('汤普森采样算法的累积懊悔为:', thompson_sampling_solver.regret)
    plot_results([thompson_sampling_solver], ["ThompsonSampling"])



if __name__ == "__main__":
    main()


