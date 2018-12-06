# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 引入归一化的包


def loadtxtAndcsv_data(filename, split, dtype):
    return np.loadtxt(filename, delimiter=split, dtype=dtype)


def logisticRegression():
    print(u"加载数据...\n")
    data = loadtxtAndcsv_data("data1.txt", ",", np.float32)
    X = np.array(data[:, 0:-1], dtype=np.float32)
    y = np.array(data[:, -1], dtype=np.float32)

    # 划分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 进行归一化操作
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)  # fit相当于训练  transform相当于归一化
    x_test = ss.transform(x_test)  # test集只需要归一化 不用于训练

    # 逻辑回归模型
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # 预测结果
    result = model.predict(x_test)
    right = sum(result == y_test)

    result2 = np.hstack((result.reshape(-1, 1), y_test.reshape(-1, 1)))  # 将预测值和真实值放在一块，好观察
    print(result2)
    print('测试集准确率：%f%%' % (right * 100.0 / result.shape[0]))  # 计算在测试集上的准确度


# 加载txt和csv文件
def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)


if __name__ == "__main__":
    logisticRegression()
