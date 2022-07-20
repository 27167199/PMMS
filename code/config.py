# -*- coding: utf-8 -*-
# @Time : 2022/5/14 16:34
# @Author :
# @Site : 
# @File : config.py
# @Software: PyCharm

class CONFIG(object):
    """docstring for CONFIG"""
    def __init__(self):
        super(CONFIG, self).__init__()
        self.dataset = 'miRNA'
        self.seq = 'long'   #  ['long', 'short', 'ls']
        self.ngram = 4
        self.dim = 128    # BiLSTM hidden number
        self.dim_out = 38
        self.num_heads = 8
        self.layer_rnn = 2
        self.layer_output = 3
        self.lr = 1e-3
        self.lr_decay = 0.5
        self.decay_interval = 10
        self.weight_decay = 1e-6
        self.iteration = 100  # BiLSTM epoch
        self.num_hidden1 = 10  # MLP hidden1
        self.num_hidden2 = 10  # MLP hidden2
        self.mlp_epoch = 200
