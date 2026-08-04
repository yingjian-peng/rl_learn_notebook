import gymnasium as gym


# 指定渲染模式('human':弹窗   ;rgb_array:返回像素值)
env = gym.make('CartPole-v1', render_mode='human')


# 观测空间(Observation)
#   形状: (4,)
#   类型: float32
#   范围: 每个维度在 [-4.8, 4.8] 之间（实际无限，但通常在此范围内）
#   含义: [小车位置, 小车速度, 杆的角度, 杆的角速度]

# 动作空间(Action Space)
#   类型: Discrete(2)
#   取值: 0 -> 向左推, 1 -> 向右推


obs, info = env.reset(seed=0)
done = False
while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()

