# GMM 高斯混合聚类
import numpy as np
from scipy.stats import multivariate_normal  # 导入scikit中协方差库


def train(X, max_iterator=100):
    # 1. 获取样本的数量m以及特征维度n
    m, n = np.shape(X)

    # 初始化u1 u2 sigma1 sigma2
    # 这里考虑到 特征属性x1 x2 x3 x4 ... xn 之间可能有某种联系  所以用协方差矩阵来进行计算 有n个特征  那么miu和sigma矩阵就是n维的
    mu1 = X.min(axis=0)
    mu2 = X.max(axis=0)
    sigma1 = np.identity(n)
    sigma2 = np.identity(n)
    pi = 0.5

    for i in range(max_iterator):
        mu1, mu2, sigma1, sigma2, pi = expectation_maximization(X, mu1, mu2, sigma1, sigma2, pi)

    print("第一个类别的相关参数:")
    print(mu1)
    print(sigma1)
    print("第二个类别的相关参数:")
    print(mu2)
    print(sigma2)

    print("预测样本属于那个类别(概率越大就是那个类别)：")
    norm1 = multivariate_normal(mu1, sigma1)
    norm2 = multivariate_normal(mu2, sigma2)
    x = np.array([3, 2, 2])
    print(pi * norm1.pdf(x))  # 属于类别1的概率为:0.0275  => 0.989
    print((1 - pi) * norm2.pdf(x))  # 属于类别2的概率为:0.0003 => 0.011


# EM算法
def expectation_maximization(X, mu1, mu2, sigma1, sigma2, pi):
    # 1.定义两个类别高斯分布
    norm1 = multivariate_normal(mu1, sigma1)
    norm2 = multivariate_normal(mu2, sigma2)

    m = X.shape[0]
    # 2. Expectation:  这步算的是Q(z)分布 关于z的分布是什么？ 是p(z|x) 给定x情况下  属于z类别的概率
    # a. 计算所有数据属于 norm1 与 norm2 中的概率
    p1 = pi * norm1.pdf(X)
    p2 = (1 - pi) * norm2.pdf(X)

    # b. 归一化操作 计算得到Q(z)分布
    w = p1 / (p1 + p2)

    # Maximization: 这步是根据最大似然 以及Expectation中求得的分布情况 计算目标函数最大的值时的mu1, mu2, sigma1, sigma2
    mu1 = np.dot(w, X) / np.sum(w)
    mu2 = np.dot(1 - w, X) / np.sum(1 - w)
    sigma1 = np.dot(w * (X - mu1).T, (X - mu1)) / np.sum(w)
    sigma2 = np.dot((1 - w) * (X - mu2).T, (X - mu2)) / np.sum(1 - w)
    pi = np.sum(w) / m

    return mu1, mu2, sigma1, sigma2, pi


if __name__ == '__main__':
    np.random.seed(28)
    # 产生一个服从多元高斯分布的数据（标准正态分布的多元高斯数据）
    mean1 = (0, 0, 0)  # x1\x2\x3的数据分布都是服从正态分布的，同时均值均为0
    cov1 = np.identity(3)
    print(cov1)
    data1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=500)

    # 产生一个数据分布不均衡
    mean2 = (2, 2, 3)
    cov2 = np.array([[1, 1, 3], [1, 2, 1], [0, 0, 1]])
    data2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=200)

    # 合并两个数据
    data = np.vstack((data1, data2))

    train(data, 100)
