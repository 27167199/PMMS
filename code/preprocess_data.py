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

def load_seq_data():
    file_name = '../dataset/miRNA/miRNA_38.csv'
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

    cfg = CONFIG()
    DATASET = cfg.dataset
    seq = cfg.seq
    ngram = cfg.ngram

    with open('../dataset/miRNA/miRNA_' + seq + '_seq.txt', 'r') as f:
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
