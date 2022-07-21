# -*- coding: utf-8 -*-
# @Time : 2022/5/23 15:34
# @Author :
# @Site : 
# @File : MultiHead_BiLSTM.py
# @Software: PyCharm

import pickle
import sys
import timeit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from config import CONFIG


class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_bilstm = nn.LSTM(input_size=dim, hidden_size=int(dim / 2), num_layers=layer_lstm, dropout=0.1, bidirectional=True)
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads)
        self.W_attention = nn.Linear(dim_out, dim_out)
        self.hidden = nn.Linear(dim, dim_out)
        self.W_out = nn.ModuleList([nn.Linear(2*dim_out, 2*dim_out)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2*dim_out, 2)

    def attention_rnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of RNN."""
        xs = torch.unsqueeze(xs, 0)
        xs, (hidden, c) = self.W_bilstm(xs)

        query = xs.clone()
        key = query
        value = query
        xs, attn_output_weights = self.multihead_attn(query, key, value)

        xs = torch.relu(xs)
        xs = torch.squeeze(xs, 0)
        xs = torch.relu(self.hidden(xs))

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))

        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):

        row_38, words = inputs

        """Protein vector with multi-head attention-RNN."""
        compound_vector = row_38[None, :]
        word_vectors = self.embed_word(words)

        protein_vector = self.attention_rnn(compound_vector, word_vectors, layer_lstm)

        num = protein_vector.detach().numpy()
        num = num.flatten()

        with open('../output/ebd_' + seq + '_seq/ebd--' + setting + '.txt', 'a') as f:
            f.write('\t'.join(map(str, num)) + '\n')

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction

    def __call__(self, data, train=True):
        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)
        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        N = len(dataset)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def save_Loss(Loss, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, Loss)) + '\n')

def get_feature(seq, ngram):
    data = pd.read_csv("../dataset/original/process/miRNA_38.csv", header=None)
    row_feature = data.iloc[1:, 0:38].values
    miRNA_num = len(row_feature)

    rnn_file = '../output/ebd_' + seq + '_seq/ebd--miRNA--' + seq + '--ngram' + str(ngram) +'--dim128--layer_lstm2--layer_output3--lr0.001--lr_decay0.5--decay_interval10--weight_decay1e-06--iteration100.txt'

    feature = []
    with open(rnn_file, 'r') as f:
        lines = f.readlines()
        lines = lines[miRNA_num*(-1):]
        for i in lines:
            num = [float(j) for j in i.split()]
            num = np.array(num)
            feature.append(num)

    rnn_feature = np.array(feature)
    print("Feature loading complete")

    return row_feature, rnn_feature

def dot(a, b):
    new_vec = []
    for i in range(len(a)):
        sum = 0
        for j in range(len(b)):
            sum += a[i]*b[j]
        new_vec.append(sum)
    return new_vec

def feature_fussion(row_feature, rnn_feature, seq, way, ngram):
    miRNA_vec = []
    for i in range(len(row_feature)):
        row_vec = row_feature[i]
        rnn_vec = rnn_feature[i]
        fussion_vec = np.hstack((row_vec, rnn_vec))
        miRNA_vec.append(fussion_vec)

    file_name = '../output/ebd_' + seq + '_seq/fusion/ngram' + str(ngram) + '_' + way + '.csv'
    new_vec = pd.DataFrame(miRNA_vec)
    new_vec.to_csv(file_name, index=False)
    print(file_name, "has been generated！")

if __name__ == "__main__":

    """Hyperparameters."""
    cfg = CONFIG()
    seq = cfg.seq
    DATASET = cfg.dataset
    ngram = cfg.ngram
    dim = cfg.dim
    dim_out = cfg.dim_out
    num_heads = cfg.num_heads
    layer_lstm = cfg.layer_rnn
    layer_output = cfg.layer_output
    lr = cfg.lr
    lr_decay = cfg.lr_decay
    decay_interval = cfg.decay_interval
    weight_decay = cfg.weight_decay
    iteration = cfg.iteration
    setting = DATASET + "--" + seq + "--ngram" + str(ngram) + '--dim' + str(dim) + "--layer_lstm" + str(layer_lstm) + "--layer_output" + str(layer_output) + "--lr" + str(lr) + "--lr_decay" + str(lr_decay) + \
              "--decay_interval" + str(decay_interval) + "--weight_decay" + str(weight_decay) + "--iteration" + str(iteration)

    (dim, dim_out, num_heads, layer_lstm, layer_output, decay_interval,
     iteration) = map(int, [dim, dim_out, num_heads, layer_lstm, layer_output,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('../dataset/' + DATASET + '/' + seq + '/ngram' + str(ngram) + '/')
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)

    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)
    row_feature = load_tensor(dir_input + 'row_feature', torch.FloatTensor)
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    n_word = len(word_dict)

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(row_feature, proteins, interactions))
    dataset_shuffle = shuffle_dataset(dataset, 1234)
    """Set a model."""
    torch.manual_seed(1234)
    model = CompoundProteinInteractionPrediction().to(device)
    trainer = Trainer(model)

    """Output files."""
    file_Loss = '../output/BiLSTM_Result/Loss--' + setting + '.txt'
    file_model = '../output/model/' + setting
    Loss = ('Epoch\tTime(sec)\tLoss_train')
    with open(file_Loss, 'w') as f:
        f.write(Loss + '\n')
    with open('../output/ebd_' + seq + '_seq/ebd--' + setting + '.txt', 'w') as f:
        pass

    """Start training."""
    print('Training...')
    print(Loss)
    start = timeit.default_timer()

    for epoch in range(0, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_shuffle)

        end = timeit.default_timer()
        time = end - start
        Loss = [epoch+1, time, loss_train]
        print('\t'.join(map(str, Loss)))
        save_Loss(Loss, file_Loss)

    print("ebd start")
    start = timeit.default_timer()
    Loss = trainer.train(dataset)
    end = timeit.default_timer()
    time = end - start
    print("ebd complete", "time:", time)

    #-------------------------Feature Fusion-------------------------
    row_feature, rnn_feature = get_feature(seq, ngram)
    print(rnn_feature)

    feature_fussion(row_feature, rnn_feature, seq, 'cancat', ngram)