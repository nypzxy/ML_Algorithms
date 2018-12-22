# Boosting思想主要方法 - AdaBoost
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.ensemble import AdaBoostClassifier  # adaboost引入方法
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles  # 造数据


def create_data():
    # 创建符合高斯分布的数据集
    X1, y1 = make_gaussian_quantiles(cov=2.,
                                     n_samples=200, n_features=2,
                                     n_classes=2, random_state=1)
    X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                     n_samples=300, n_features=2,
                                     n_classes=2, random_state=1)

    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, -y2 + 1))

    return X, y


def adaboost():
    X, y = create_data()
    # 构建adaboost
    abt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=200)
    abt.fit(X, y)

    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    # 预测
    Z = abt.predict(np.c_[xx.ravel(), yy.ravel()])
    # 设置维度
    Z = Z.reshape(xx.shape)


if __name__ == "__main__":
    adaboost()
