# -*- coding: utf-8 -*-
# @Time : 2021/12/14 16:25
# @Author :
# @Site : 
# @File : load_data.py
# @Software: PyCharm

import pandas as pd
import numpy as np

def load_seq_data(way):
    file_name = './data/miRNA_' + way + '.csv'
    feature_data = pd.read_csv(file_name, header=None)

    label_data = pd.read_csv("data/miRNA_原始.csv", header=None)

    data = feature_data.iloc[1:, 0:38].values
    target = label_data.iloc[1:, 38:].values
    target_new = []
    for i in target:
        target_new.append(i[0])
    target = np.array(target_new)
    print(file_name, "加载完毕！")

    return data, target

def load_data(seq, way, ngram):
    if way == '原始':
        file_name = '../dataset/original/process/miRNA_38.csv'
    else:
        file_name = '../output/ebd_' + seq + '_seq/fusion/ngram' + str(ngram) + '_' + way + '.csv'
    feature_data = pd.read_csv(file_name, header=None)

    label_data = pd.read_csv('../dataset/original/process/miRNA_38.csv', header=None)

    if way == '拼接':
        data = feature_data.iloc[1:, 0:76].values
    else:
        data = feature_data.iloc[1:, 0:38].values
    target = label_data.iloc[1:, 38:].values
    target_new = []
    for i in target:
        target_new.append(i[0])
    target = np.array(target_new)
    print(file_name, "加载完毕！")

    return data, target
