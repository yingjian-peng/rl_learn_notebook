import gymnasium as gym                                 # 导入 Gymnasium 库



env = gym.make("FrozenLake-v1", render_mode="human")    # 创建环境,并实时显示画面
env = env.unwrapped                                     # 解封装才能访问状态转移矩阵P
state, info = env.reset()                               # 将环境初始化到1个新的回合
env.render()
print("初始状态:", state)



holes = set()                                           # 创建空集合,用于保存冰洞状态编号
ends = set()                                            # 创建空集合，用于保存目标状态编号
for s in env.P:                                         # 表示从最终状态出发时的所有动作
    for a in env.P[s]:                                  # 遍历当前状态 s 下的所有动作
        for s_ in env.P[s][a]:                          # 遍历执行动作 a 后可能出现的所有转移结果
            if s_[2] == 1.0:                            # 获得奖励为1,代表是目标
                ends.add(s_[1])                         # s_[1] 是下一个状态的编号
            if s_[3] == True:
                holes.add(s_[1])
holes = holes - ends                                    # 最终 holes 中只保留真正的冰洞状态

print("冰洞的索引:", holes)
print("目标的索引:", ends)


for a in env.P[14]:                                     # 保存从状态 14 出发时四个动作的转移信息
    print(env.P[14][a])                                 # (转移概率, 下一个状态, 奖励, 是否结束)


env.close()                                             # 关闭环境并释放相关资源





