import copy



#---------------------------------------------------------------------------------------
# 悬崖漫步环境
#---------------------------------------------------------------------------------------
class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol                                        # 定义网格世界的列
        self.nrow = nrow                                        # 定义网格世界的行
        self.P = self.createP()

    def createP(self):
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        change = [[0,-1], [0,1], [-1,0], [1,0]]                 # 定义在左上角
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):                              # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:            # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                        P[i*self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    next_x = min(self.ncol -1, max(0, j+change[a][0]))  # 其他位置
                    next_y = min(self.nrow -1, max(0, i+change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    if next_y == self.nrow -1 and next_x > 0:   # 下一个位置在悬崖或者终点
                        done = True
                        if next_x != self.ncol -1:              # 下一个位置在悬崖
                            reward = -100
                    P[i*self.ncol + j][a] = [(1, next_state, reward, done)]
        return P


