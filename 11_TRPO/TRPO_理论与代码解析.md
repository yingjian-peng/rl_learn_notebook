# TRPO 理论与代码解析

这份文档面向工程实现理解，不追求完整数学推导。目标是看懂 `trpo.py` 为什么要这样更新策略，以及每段代码在算法流程里负责什么。当前代码可以理解为《动手学强化学习》教材示例迁移到新版 `gymnasium` 后的单文件实现。

## 1. TRPO 想解决什么问题

普通策略梯度直接沿着梯度更新策略：

```text
新策略参数 = 旧策略参数 + 学习率 * 梯度
```

问题是策略网络很敏感。学习率稍大，动作概率分布可能突然变化，采样数据就不再可信，训练回报会剧烈波动甚至崩掉。

TRPO 的核心想法是：

```text
每次尽量提升策略目标，但限制新旧策略的 KL 距离不要太大。
```

可以把它理解为“带安全半径的策略更新”。不是问“梯度往哪里最大”，而是问：

```text
在策略变化不超过一个小范围的前提下，往哪里走收益提升最多？
```

代码里这个限制由 `kl_constraint` 控制。

## 2. 这份代码的整体流程

一次 episode 结束后，`train_on_policy_agent()` 收集一条轨迹：

```text
state, action, reward, next_state, done
```

然后调用：

```python
agent.update(transition_dict)
```

`update()` 里分两件事：

1. 更新 critic，也就是价值网络 `ValueNet`
2. 更新 actor，也就是策略网络 `PolicyNet`

critic 使用普通 Adam 和 MSE 损失更新，比较像 Actor-Critic。actor 使用 TRPO 的特殊流程更新，没有直接使用 Adam。

## 3. 网络分别代表什么

### `PolicyNet`

输入状态，输出每个离散动作的概率：

```python
return F.softmax(self.fc2(x), dim=1)
```

在 CartPole 里动作只有两个，所以输出类似：

```text
[向左概率, 向右概率]
```

`take_action()` 用 `Categorical(probs)` 按概率采样动作。

### `ValueNet`

输入状态，输出当前状态的价值估计：

```text
V(s)
```

它用来判断“这一步动作之后，比预期好还是差”。

## 4. advantage 是什么

策略更新不直接用 reward，而是用 advantage：

```text
advantage = 实际表现 - 预期表现
```

直觉上：

```text
advantage > 0：这个动作比预期好，以后提高它的概率
advantage < 0：这个动作比预期差，以后降低它的概率
```

代码里先算 TD 误差：

```python
td_target = rewards + gamma * self.critic(next_states) * (1 - dones)
td_delta = td_target - self.critic(states)
```

再用 GAE 计算 advantage：

```python
advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
```

`lmbda` 是 GAE 参数。它控制 bias 和 variance 的折中。工程上一般先用 `0.95`，不用太纠结。

## 5. TRPO 的 actor 更新在做什么

TRPO 不直接写：

```python
optimizer.step()
```

而是走这几个步骤：

```text
1. 保存旧策略下的动作概率
2. 构造 surrogate objective
3. 计算目标函数梯度 g
4. 用共轭梯度近似求 H^(-1)g
5. 根据 KL 约束缩放步长
6. 用线性搜索确认真的变好且 KL 没超
7. 写回 actor 参数
```

对应代码入口是：

```python
self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)
```

## 6. surrogate objective 是什么

代码：

```python
ratio = torch.exp(log_probs - old_log_probs)
return torch.mean(ratio * advantage)
```

这里的 `ratio` 是：

```text
新策略选择这个动作的概率 / 旧策略选择这个动作的概率
```

如果某个动作 advantage 为正，TRPO 希望新策略提高它的概率；如果 advantage 为负，希望降低它的概率。

这个目标函数是“用旧数据估计新策略是否更好”的近似目标。因为数据来自旧策略，所以要用 `ratio` 修正。

## 7. KL 约束为什么重要

KL 距离衡量两个动作概率分布差多远：

```python
kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
```

例如旧策略在某个状态下是：

```text
[0.50, 0.50]
```

新策略如果变成：

```text
[0.52, 0.48]
```

变化很小，KL 小。  
如果突然变成：

```text
[0.95, 0.05]
```

变化很大，KL 大。

TRPO 允许策略更新，但不允许一步跨太远。

## 8. Hessian-vector product 是什么

TRPO 理论里需要用 KL 的 Hessian 矩阵。但神经网络参数很多，直接构造 Hessian 很贵。

所以代码只计算：

```text
H * vector
```

而不是完整的 `H`。

对应函数：

```python
hessian_matrix_vector_product()
```

它通过 PyTorch 的二阶自动求导得到 Hessian-vector product。工程上可以把它当成“告诉共轭梯度法 KL 曲面形状”的工具函数。

## 9. 共轭梯度在这里干什么

TRPO 的理想更新方向近似是：

```text
H^(-1) * g
```

其中：

```text
g：surrogate objective 的梯度
H：KL 对参数的二阶曲率
```

直接求逆很贵，所以用共轭梯度法解线性方程：

```text
H x = g
```

求出来的 `x` 就近似是 `H^(-1)g`。

对应代码：

```python
descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)
```

变量名叫 `descent_direction`，但这里优化目标是最大化 surrogate objective，更准确地说它是策略提升方向。

## 10. 线性搜索在这里干什么

即使用 KL 约束算出了理论步长，实际神经网络还是可能出现：

```text
目标没变好
KL 超过限制
```

所以 `line_search()` 会从最大步长开始试：

```text
1.0, alpha, alpha^2, alpha^3, ...
```

直到找到一个满足条件的新参数：

```python
if new_obj.item() > old_obj.item() and kl_div.item() < self.kl_constraint:
    return new_para
```

如果找不到，就保持旧参数不变。

## 11. 工程调参建议

这份 CartPole 示例可以先保持默认：

```python
lmbda = 0.95
kl_constraint = 0.0005
alpha = 0.95
critic_lr = 1e-2
gamma = 0.98
```

如果训练很慢，可以略微增大 `kl_constraint`，例如 `0.001`。  
如果训练不稳定，可以减小 `kl_constraint`，例如 `0.0002`。

`alpha` 是线性搜索缩放系数，越小回退越快；一般 `0.8` 到 `0.95` 都常见。

## 12. 看代码时抓住这条主线

你不需要先掌握完整推导，只要记住这条线：

```text
采样一条轨迹
-> critic 估计状态价值
-> GAE 算每个动作好坏
-> actor 想提高好动作概率
-> 但用 KL 限制新旧策略距离
-> 共轭梯度负责找受 KL 曲率修正后的方向
-> 线性搜索负责最后把关
```

这就是工程实现里 TRPO 的主体。
