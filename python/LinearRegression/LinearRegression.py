from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import threading
import time

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 解决windows环境下画图汉字乱码问题


# 加载txt和csv文件
def load_data(fileName, split, dataType):
    print(u"加载数据...\n")
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)


# 特征值归一化 为什么要特征归一化呢  其实很简单  只有使得各特征之间的大小范围一致，才能使用距离度量等算法 加速梯度下降算法的收敛
def featureNormalization(X):
    X_norm = np.array(X)  # 将X转化为numpy数组对象，才可以进行矩阵的运算

    # 归一化是利用 标准差 来计算的  所以需要均值(mu) 来计算标准差(sigma) 然后在归一化
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))

    mu = np.mean(X_norm, 0)  # 求每一列的平均值（0指定为列，1代表行）
    sigma = np.std(X_norm, 0)  # 求每一列的标准差

    for i in range(X.shape[1]):  # 遍历列
        X_norm[:, i] = (X_norm[:, i] - mu[i]) / sigma[i]  # 归一化

    return X_norm, mu, sigma


# 画二维图
def plot_X1_X2(X):
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


# 梯度下降法
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)  # 47
    n = len(theta)  # 3 theta是列向量  注意了  3,1

    # 创建一个temp容器  保存每次迭代的theta值  比如300次迭代 那么这个矩阵为3 * 300的
    temp = np.matrix(np.zeros((n, num_iters)))

    # 记录每次迭代计算的代价值
    J_loss = np.zeros((num_iters, 1))

    # 主要算法部分
    for i in range(num_iters):
        h = np.dot(X, theta)  # 这个hypothesis假设函数为θX
        # 接着就是梯度的计算 我们使用的是BGD算法
        temp[:, i] = theta - (alpha * (1 / m) * (np.dot(np.transpose(X), h - y)))
        theta = temp[:, i]
        J_loss[i] = calc_cost(X, y, theta)

    return theta, J_loss


def calc_cost(X, y, theta):
    m = len(y)
    J = (np.transpose(X * theta - y)) * (X * theta - y) / (2 * m)  # 计算代价J
    return J


# 画每次迭代代价的变化图
def plotJ(J_history, num_iters):
    x = np.arange(1, num_iters + 1)
    plt.plot(x, J_history)
    plt.xlabel(u"迭代次数", fontproperties=font)  # 注意指定字体，要不然出现乱码问题
    plt.ylabel(u"代价值", fontproperties=font)
    plt.title(u"代价随迭代次数的变化", fontproperties=font)
    plt.show()


def linearRegression(alpha=0.01, num_iters=400):
    data = load_data("data.txt", ",", np.float64)
    X = data[:, 0:-1]  # X特征值
    y = data[:, -1]  # Y值
    # 这里要注意一下  numpy的操作是行优先的 所以读取的数据X.shape = 47,3  y.shape = 47,
    col = data.shape[1]  # 列数 2行3列的矩阵 data.shape = (2,3) 获取到这里的3列
    m = len(y)  # 总的数据条数

    # 然后我们需要做一下归一化操作: 为了防止某个特征值太大
    X, mu, sigma = featureNormalization(X)
    plot_X1_X2(X)

    # 由于本实例中有两个特征值x1 x2 但是同时我们也需要截距项 即θ0 x0 使x0全为1即可
    # 添加一行特征 使该特征值全部为1
    X = np.hstack((np.ones((m, 1)), X))

    print(u"\n执行梯度下降算法....\n")
    # 我们要求的是最后的theta值 而这个theta是一个向量的形式 所以定义一个2行1列的矩阵
    theta = np.zeros((col, 1))  # theta.shape = 3,1 为列向量

    # 将行向量转换成列向量 这是因为numpy的操作是行优先的 y.shape = (47,)  我们要将他转换成列向量 才可以进行计算 因为后续的计算是列来做的
    # 如(θx - y) θ是3,1 每条样本对应的x是行向量1*3 一共47条 为47*3
    y = y.reshape(-1, 1)
    theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

    plotJ(J_history, num_iters)

    return mu, sigma, theta  # 返回均值mu,标准差sigma,和学习的结果theta


# 测试linearRegression函数
def testLinearRegression():
    linearRegression(0.01, 400)
    # print u"\n计算的theta值为：\n",theta
    # print u"\n预测结果为：%f"%predict(mu, sigma, theta)


# 测试学习效果（预测）
def predict(mu, sigma, theta):
    result = 0
    # 注意归一化
    predict = np.array([1650, 3])
    norm_predict = (predict - mu) / sigma
    final_predict = np.hstack((np.ones((1)), norm_predict))

    result = np.dot(final_predict, theta)  # 预测结果
    return result


# 梯度下降算法
if __name__ == "__main__":
    testLinearRegression()
