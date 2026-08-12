import numpy as np
import matplotlib.pyplot as plt



#---------------------------------------------------------------------------------------
# 伯努利多臂老虎机
#---------------------------------------------------------------------------------------
class BernoulliBandit:
    def __init__(self, K):                          # K:拉杆个数
        self.probs = np.random.uniform(size=K)      # 随机生成K个0~1的数,作为获奖概率
        self.best_idx = np.argmax(self.probs)       # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率
        self.K = K
    
    def step(self, k):
        if np.random.rand() < self.probs[k]:        # 当前概率 < 该号拉杆的获奖概率
            return 1                                # 获奖
        else:
            return 0                                # 未获奖



#---------------------------------------------------------------------------------------
# 多臂老虎机算法基本框架
#---------------------------------------------------------------------------------------
class Solver:
    def __init__(self, bandit):
        self.bandit = bandit                         # 保存当前交互的多臂老虎机对象
        self.counts = np.zeros(self.bandit.K)       # 每根拉杆的尝试次数
        self.regret = 0.0                           # 当前步的累积懊悔
        self.actions = []                           # 维护1个列表,记录每1步的动作
        self.regrets = []                           # 维护1个列表,记录每1步的累积懊悔

    # 计算累积懊悔并保存k,k为本次动作选择的拉杆的编号
    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k] # 期望奖励估值
        self.regrets.append(self.regret)            # 记录当前步拉动拉杆对应的奖励值

    # 返回当前动作选择哪根拉杆,由每个具体的策略实现
    def run_one_step(self):
        raise NotImplementedError                   # 拉动1次拉杆返回拉杆编号
    
    # 运行一定的次数,num_steps为总运行次数
    def run(self, num_steps):
        for _ in range(num_steps):                  # for循环 num_steps 次
            k = self.run_one_step()                 # 拉动1次拉杆,返回拉杆编号
            self.counts[k] += 1                     # 该编号拉杆拉动次数+1
            self.actions.append(k)                  # 记录当前步拉动的拉杆编号
            self.update_regret(k)                   # 计算期望奖励并记录



#---------------------------------------------------------------------------------------
# 主函数
#---------------------------------------------------------------------------------------
def main():
    np.random.seed(1)                                   # 设定随机种子
    K = 10
    bandit_10_arm = BernoulliBandit(K)

    print("随机生成一个%d臂伯努利老虎机" %K)
    print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %(bandit_10_arm.best_idx, bandit_10_arm.best_prob))





if __name__ == "__main__":
    main()







