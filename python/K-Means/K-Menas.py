# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy import io as spio
from scipy import misc  # 图片操作
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 解决windows环境下画图汉字乱码问题


def k_means():
    '''二维数据聚类过程演示'''
    print(u'聚类过程展示...\n')
    data = spio.loadmat("data.mat")


if __name__ == "__main__":
    k_means()
