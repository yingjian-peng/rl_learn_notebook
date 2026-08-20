import numpy as np



#---------------------------------------------------------------------------------------
# 悬崖漫步环境
#---------------------------------------------------------------------------------------
class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0                                  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1                      # 记录当前智能体位置的纵坐标

    # 环境交互函数
    def step(self, action):                         # 输入:智能体的动作
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]] # 定义在左上角
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done             # 下一个状态,奖励

    # 初始化函数
    def reset(self):                                # 回归初始状态,坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x
    




