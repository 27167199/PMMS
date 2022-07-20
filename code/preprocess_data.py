# -*- coding: utf-8 -*-
# @Time : 2022/3/21 9:39
# @Author :
# @Site :
# @File : preprocess_data.py
# @Software: PyCharm

from collections import defaultdict
import os
import pickle
import sys
from config import CONFIG

import numpy as np
import pandas as pd

def get_row_vec():
    data = np.load('../dataset/original/dataset.npz')

    feature = data['X'].tolist()
    label = data['Y'].tolist()

    print("before:", len(feature[0]))
    for i in range(len(feature)):
        feature[i].append(label[i])
    print("after:", len(feature[0]))

    new_data = pd.DataFrame(feature)

    new_data.to_csv('../dataset/original/process/miRNA_38.csv', index=False)
    print("原始特征已读取保存！")

def save_file(filename, seq_label):
    with open(filename, 'w') as f:
        for i in range(len(seq_label)):
            print(seq_label[i][0] + ' ' + seq_label[i][1] + ' ' + str(seq_label[i][2]) + "\r")
            f.write(seq_label[i][0] + ' ' + seq_label[i][1] + ' ' + str(seq_label[i][2]) + "\r")
        f.close()
    print(filename, "has been generated！")

def get_seq(way):
    # 读取实验154个miRNA名称
    file1 = open("../dataset/original/正负样本名称.txt")
    list = file1.readlines()
    miRNA_name = []
    interaction = []
    for item in list:
        item = item.split()
        miRNA_name.append(item[0])
        interaction.append(item[1])
    # print(miRNA_name)

    # 读取所有miRNA序列信息
    file2 = open("../dataset/original/mmu_mir_data.txt")
    list = file2.readlines()
    lists = []
    for item in list:
        item = item.split()
        lists.append(item)

    #  获取154个miRNA的短序列，存入字典
    miRNA_seq = {}
    if way == 'short':
        col = 2
    elif way == 'long':
        col = 1
    for i in range(len(miRNA_name)):
        for j in range(len(lists)):
            if miRNA_name[i] == lists[j][0]:
                miRNA_seq[miRNA_name[i]] = lists[j][col]

    # 寻找序列最大长度
    seq = []
    for k,v in miRNA_seq.items():
        seq.append(v)
        # print(len(v))
    max_len_seq = max(seq, key=len, default='')
    max_len = len(max_len_seq)
    print("max_len:", max_len)

    seq_label = []
    for i in range(len(interaction)):
        cord = []
        cord.append(miRNA_name[i])
        cord.append(seq[i])
        cord.append(int(interaction[i]))
        seq_label.append(cord)

    filename = '../dataset/original/process/miRNA_' + way + '_seq.txt'
    save_file(filename, seq_label)

    return seq_label

def save_ls_seq(short_seq, long_seq):
    long_short_seq = []
    for i in range(len(short_seq)):
        cord = []
        cord.append(short_seq[i][0])
        long = long_seq[i][1]
        short = short_seq[i][1]
        long_short = long + short
        cord.append(long_short)
        cord.append(short_seq[i][2])
        long_short_seq.append(cord)

    save_file("../dataset/original/process/miRNA_ls_seq.txt", long_short_seq)

def load_seq_data():
    file_name = '../dataset/original/process/miRNA_38.csv'
    feature_data = pd.read_csv(file_name, header=None)

    data = feature_data.iloc[1:, 0:38].values

    return data

def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)

if __name__ == "__main__":

    # DATASET, radius, ngram = sys.argv[1:]
    cfg = CONFIG()
    DATASET = cfg.dataset
    seq = cfg.seq
    ngram = cfg.ngram

    '''
    Load miRNA row data
    '''
    '''
    short_seq = get_seq('short')
    long_seq = get_seq('long')
    save_ls_seq(short_seq, long_seq)
    '''

    with open('../dataset/original/process/miRNA_' + seq + '_seq.txt', 'r') as f:
        data_list = f.read().strip().split('\n')

    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)

    word_dict = defaultdict(lambda: len(word_dict))

    proteins, interactions = [], []

    for no, data in enumerate(data_list):

        print('/'.join(map(str, [no+1, N])))

        name, sequence, interaction = data.strip().split()

        words = split_sequence(sequence, ngram)
        proteins.append(words)

        interactions.append(np.array([float(interaction)]))

    dir_input = ('../dataset/' + DATASET + '/' + seq + '/ngram' + str(ngram) + '/')
    os.makedirs(dir_input, exist_ok=True)

    row_feature = load_seq_data()
    np.save(dir_input + 'row_feature', row_feature)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'interactions', interactions)
    dump_dictionary(word_dict, dir_input + 'word_dict.pickle')

    print('The preprocess of ' + DATASET + ' dataset has finished!')
