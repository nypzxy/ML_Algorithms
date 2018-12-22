# HMM隐马尔科夫前向算法
import numpy as np


def forward_calc(pi, A, B, Q):
    """
    :type pi:1*n 的初始状态的概率分布
    :type A: n*n 的状态转移矩阵
    :type B: n*m 的观测概率矩阵 n表示n个状态  m表示每个状态下的m种取值的概率 这个是离散的情况
    :type Q: 1*T 长度为T的观测序列
    :return alpha: alpha矩阵  代表T个时刻的alpha值 t*n n个状态 不同状态对应的alpha值不同 比如t=1时刻 n=1 n=2 n=3 分别对应不同的概率值
    """
    # 1. 获取信息
    n = np.shape(A)[0]  # n个状态
    T = np.shape(Q)[0]  # t个时刻

    alpha = np.zeros((T, n))

    # 2. 更新t=1时刻的前向概率
    for i in range(n):
        alpha[0][i] = pi[i] * B[i][Q[0]]

    # 3. 更新t=2...T时刻的前向概率
    for t in range(1, T):

        for j in range(n):
            alpha[t][j] = np.dot(alpha[t - 1], np.transpose(A[:, j])) * B[j][Q[t]]

    return alpha


if __name__ == "__main__":
    pi = np.array([0.2, 0.5, 0.3])
    A = np.array([
        [0.5, 0.4, 0.1],
        [0.2, 0.2, 0.6],
        [0.2, 0.5, 0.3],
    ])
    B = np.array([
        [0.4, 0.6],
        [0.8, 0.2],
        [0.5, 0.5]
    ])
    Q = np.array([0, 1, 0, 0, 1])  # "白黑白白黑"

    alpha = forward_calc(pi, A, B, Q)
    print(alpha)
    # 计算最终概率值：
    p = 0
    for i in alpha[-1]:
        p += i
    print(Q, end="->出现的概率为:")
    print(p)
