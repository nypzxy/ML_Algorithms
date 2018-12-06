import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from scipy import optimize

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 解决windows环境下画图汉字乱码问题


# 逻辑回归是基于线性回归的分类问题 利用回归的思想 构造分类的边界
def load_data(filename, split, dataType):
    return np.loadtxt(filename, delimiter=split, dtype=dataType)


# 对原来的特征量扩维
def mapFeature(X1, X2):
    degree = 2  # 映射的最高次方
    out = np.ones((X1.shape[0], 1))  # 映射后的结果数组（取代X）
    '''
    这里以degree=2为例，映射为x1x2,x1^2,x1,x2,x2^2
    当i=1 j=0时 X1
    当i=1 j=1时 X2
    
    当i=2 j=0时 X1二次方
    当i=2 j=1时 X1X2
    当i=2 j=2时 X2的二次方
    '''
    for i in np.arange(1, degree + 1):
        for j in range(i + 1):
            temp = X1 ** (i - j) * (X2 ** j)  # 矩阵直接乘相当于matlab中的点乘.*
            print()
            out = np.hstack((out, temp.reshape(-1, 1)))
    return out


def sigmoid(init_theta, X):
    z = np.dot(X, init_theta)  # theta行向量 6，1  X 矩阵 118，6
    hypothesis_func = 1 / (1 + np.exp(-z))
    return hypothesis_func


def cost_function(init_theta, X, y, init_lambda):
    m = len(y)
    J = 0

    hypothesis_func = sigmoid(init_theta, X)
    theta1 = init_theta.copy()
    theta1[0] = 0  # 因为正则不需要对截距项进行计算

    temp = np.dot(np.transpose(theta1), theta1)
    J = (-np.dot(y, np.log(hypothesis_func)) - np.dot(1 - y, np.log(1 - hypothesis_func)) + init_lambda * temp / 2) / m

    return J


def gradient_descent(initial_theta, X, y, init_lambda):
    m = len(y)
    hypothesis_func = sigmoid(initial_theta, X)

    theta1 = initial_theta.copy()
    theta1[0] = 0

    grad = np.dot(hypothesis_func - y, X) / m + init_lambda / m * theta1  # 正则化的梯度
    return grad


# 预测
def predict(X, theta):
    m = X.shape[0]
    p = sigmoid(theta, X)  # 预测的结果，是个概率值

    for i in range(m):
        if p[i] > 0.5:  # 概率大于0.5预测为1，否则预测为0
            p[i] = 1
        else:
            p[i] = 0
    return p


def logisticRegression():
    # 1 加载数据
    data = load_data("data2.txt", ",", np.float64)
    X = data[:, 0:-1]
    y = data[:, -1]

    plot_data(X, y)  # 作图 有了图我们就不难发现 这个图是无法分割的 故而只能通过多项式扩维
    X = mapFeature(X[:, 0], X[:, 1])

    # print(X)  # 看一下映射之后的变量 2维变成6维

    # 2. 逻辑回归之损失函数 + L2正则
    # 先初始化一下θ向量 X.shape[1] = 6列 = 6个特征 同样需要6个theta
    init_theta = np.zeros((X.shape[1], 1))  # 此时为行向量
    init_lambda = 0.01  # 初始化正则项

    J = cost_function(init_theta, X, y, init_lambda)  # 有了所有的参数之后就可以写出我们的cost 损失函数了

    print(J)  # 输出一下计算的值，应该为0.693147 到这里损失函数就定义完了  下面我们就开始

    # 3. 逻辑回归之梯度下降 - 这里使用sklearn中的optimize包下的你牛顿法
    result = optimize.fmin_bfgs(cost_function, init_theta, fprime=gradient_descent, args=(X, y, init_lambda))
    p = predict(X, result)  # 预测
    print(u'在训练集上的准确度为%f%%' % np.mean(np.float64(p == y) * 100))  # 与真实值比较，p==y返回True，转化为float


# 显示二维图形
def plot_data(X, y):
    pos = np.where(y == 1)  # 找到y==1的坐标位置
    neg = np.where(y == 0)  # 找到y==0的坐标位置

    plt.figure(figsize=(15, 12))
    plt.plot(X[pos, 0], X[pos, 1], 'ro')  # red o
    plt.plot(X[neg, 0], X[neg, 1], 'bo')  # blue o
    plt.title(u"两个类别散点图", fontproperties=font)
    plt.show()


if __name__ == "__main__":
    logisticRegression()
