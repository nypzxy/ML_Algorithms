# Bagging思想主要方法 - RandomForest 随机森林 “三个臭皮匠顶个诸葛亮”
# 案例 利用随机森林实现宫颈癌预测

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import label_binarize
from sklearn import metrics


def random_forest():
    data = read_csv()
    ## 模型存在多个需要预测的y值，如果是这种情况下，简单来讲可以直接模型构建，在模型内部会单独的处理每个需要预测的y值，相当于对每个y创建一个模型
    X = data[0:-4]
    Y = data[-4:]
    # 特征处理 ?=>NaN 之后将缺省值(NaN)转换成列均值
    X = X.replace("?", np.nan)
    imputer = Imputer(missing_values="NaN")
    X = imputer.fit_transform(X, Y)
    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    # 特征处理 标准化
    ss = MinMaxScaler()  # 分类模型，经常使用的是minmaxscaler归一化，回归模型经常用standardscaler
    x_train, y_train = ss.fit_transform(x_train, y_train)
    x_test = ss.transform(x_test)

    # 特征处理 降维
    pca = PCA(n_components=2)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    # 随机森林模型
    RandomForestClassifier(n_estimators=100, max_depth=1, random_state=0)


def read_csv():
    path = "data/risk_factors_cervical_cancer.csv"
    return pd.read_csv(path)


if __name__ == "__main__":
    random_forest()
