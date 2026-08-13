"""
使用 蒙特卡洛方法 计算MDP中的状态价值
"""


import numpy as np



#---------------------------------------------------------------------------------------
# MDP数据
#---------------------------------------------------------------------------------------
S = ["s1", "s2", "s3", "s4", "s5"]              # 状态集合

A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]    # 动作集合

P = {"s1-保持s1-s1":1.0, "s1-前往s2-s2":1.0,    # 状态转移函数
    "s2-前往s1-s1":1.0, "s2-前往s3-s3":1.0,
    "s3-前往s4-s4":1.0, "s3-前往s5-s5":1.0,
    "s4-前往s5-s5":1.0, "s4-概率前往-s2":0.2,
    "s4-概率前往-s3":0.4, "s4-概率前往-s4":0.4,}


R = {"s1-保持s1":-1, "s1-前往s2":0,             # 奖励函数
    "s2-前往s1":-1, "s2-前往s3":-2,
    "s3-前往s4":-2, "s3-前往s5":0,
    "s4-前往s5":10, "s4-概率前往":1,}

gamma = 0.5                                     # 折扣因子

MDP = (S, A, P, R, gamma)



#---------------------------------------------------------------------------------------
# 策略(此处使用一个固定的策略)
#---------------------------------------------------------------------------------------
Pi_1 = {
    "s1-保持s1":0.5, "s1-前往s2":0.5,
    "s2-前往s1":0.5, "s2-前往s3":0.5,
    "s3-前往s4":0.5, "s3-前往s5":0.5,
    "s4-前往s5":0.5, "s4-概率前往":0.5,
}



#---------------------------------------------------------------------------------------
# 采样函数,策略Pi, 限制最长时间步 timestep_max,总共采样序列数number
#---------------------------------------------------------------------------------------
def join(str1, str2):                               # 把输入的两个字符串通过“-”连接,便于使用上述定义的 P、R 变量
    return str1 + '-' + str2


def sample(MDP, Pi, timestep_max, number):
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):                         # 采样次数
        episode = []                                # 单词采样总回报
        timestep = 0                                # 时间步计数器
        s = S[np.random.randint(4)]                 # 随机选择1个除s5以外的状态s作为起点
        while s != "s5" and timestep <= timestep_max:   # 当前状态为终止状态或时间步太长时,1次采样结束
            timestep += 1                           # 时间步+1
            rand, temp = np.random.rand(), 0        # 
            for a_opt in A:                         # 在状态s下根据策略选择动作
                temp += Pi.get(join(s, a_opt), 0)   # 相当于策略执行,返回动作a
                if temp > rand:                     # 执行动作的概率 > 随机值
                    a = a_opt                       # 获取Pi_1里的动作
                    r = R.get(join(s,a), 0)         # 获取对应的奖励
                    break
            rand, temp = np.random.rand(), 0
            s_next = None
            for s_opt in S:                         # 根据状态得到下一个状态s_next
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt                  # 获得下一个状态
                    break
            episode.append((s, a, r, s_next))       # 把(s,a,r,s_next)元组放入序列中
            s = s_next                              # 变成当前状态,开始接下来的循环
        episodes.append(episode)                    # 保存一轮完整流程
    return episodes



#---------------------------------------------------------------------------------------
# 采样5次,每个序列最长不超过20步
#---------------------------------------------------------------------------------------
episodes = sample(MDP, Pi_1, 20, 5)

print('第1条序列\n', episodes[0])
print('第1条序列\n', episodes[1])
print('第1条序列\n', episodes[4])



#---------------------------------------------------------------------------------------
# 计算状态价值
#---------------------------------------------------------------------------------------
def MC(episodes, V, N, gamma):
    for episode in episodes:
        G =0
        for i in range(len(episode)-1, -1, -1):
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1                         # 状态计数器
            V[s] = V[s] + (G - V[s]) / N[s]         # 状态总回报



#---------------------------------------------------------------------------------------
# 采样1000次计算状态价值
#---------------------------------------------------------------------------------------
timestep_max = 20
episodes = sample(MDP, Pi_1, timestep_max, 1000)

gamma = 0.5
V = {"s1":0, "s2":0, "s3":0, "s4":0, "s5":0}
N = {"s1":0, "s2":0, "s3":0, "s4":0, "s5":0}
MC(episodes, V, N, gamma)

print("\n使用蒙特卡洛方法计算MDP的状态价值为\n", V)



#---------------------------------------------------------------------------------------
# 计算状态动作对(s,a)出现的频率,以此来估算策略的占用度量
#---------------------------------------------------------------------------------------
def occupancy(episodes, s, a, timestep_max, gamma): 
    rho = 0 
    total_times = np.zeros(timestep_max)            # 记录每个时间步 t 各被经历过几次
    occur_times = np.zeros(timestep_max)            # 记录(s_t,a_t)=(s,a)的次数
    for episode in episodes: 
        for i in range(len(episode)): 
            (s_opt, a_opt, r, s_next) = episode[i] 
            total_times[i] += 1 
            if s == s_opt and a == a_opt: 
                occur_times[i] += 1 
    for i in reversed(range(timestep_max)): 
        if total_times[i]: 
            rho += gamma**i * occur_times[i] / total_times[i] 
    
    return (1 - gamma) * rho



#---------------------------------------------------------------------------------------
# 计算占用度量
#---------------------------------------------------------------------------------------
Pi_2 = { 
    "s1-保持s1":0.6, "s1-前往s2":0.4, 
    "s2-前往s1":0.3, "s2-前往s3":0.7, 
    "s3-前往s4":0.5, "s3-前往s5":0.5, 
    "s4-前往s5":0.1, "s4-概率前往":0.9, 
}

timestep_max = 1000
episodes_1 = sample(MDP, Pi_1, timestep_max, 1000) 
episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)

gamma = 0.5 
rho_1 = occupancy(episodes_1, "s4", "概率前往", timestep_max, gamma)
rho_2 = occupancy(episodes_2, "s4", "概率前往", timestep_max, gamma)

print("\n", rho_1, rho_2)






















