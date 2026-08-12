import numpy as np
import matplotlib.pyplot as plt
from bernoulli_bandit import BernoulliBandit, Solver



#---------------------------------------------------------------------------------------
# USB算法(继承Solver类)
#---------------------------------------------------------------------------------------
class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef
    
    def run_one_step(self):
        self.total_count += 1
        # 计算上置信界
        ucb = self.estimates+self.coef*np.sqrt(np.log(self.total_count)/(2*(self.counts+1)))
        k = np.argmax(ucb)                                      # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k]+1)*(r-self.estimates[k])
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
    coef = 1                                                # 控制不确定性比重的系数
    bandit_10_arm = BernoulliBandit(10)
    UCB_solver = UCB(bandit_10_arm, coef)
    UCB_solver.run(5000)

    print('上置信界算法的累积懊悔为:', UCB_solver.regret)
    plot_results([UCB_solver], ['ucb'])



if __name__ == "__main__":
    main()


