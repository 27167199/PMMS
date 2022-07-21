# -*- coding: utf-8 -*-
# @Time : 2022/5/14 16:25
# @Author :
# @Site : 
# @File : load_data.py
# @Software: PyCharm

import pandas as pd
import numpy as np

def load_data(seq, way, ngram):
    file_name = '../output/ebd_' + seq + '_seq/fusion/ngram' + str(ngram) + '_' + way + '.csv'
    feature_data = pd.read_csv(file_name, header=None)

    label_data = pd.read_csv('../dataset/miRNA/miRNA_38.csv', header=None)

    data = feature_data.iloc[1:, 0:76].values
    target = label_data.iloc[1:, 38:].values
    target_new = []
    for i in target:
        target_new.append(i[0])
    target = np.array(target_new)
    print(file_name, "Loading completed!")

    return data, target
