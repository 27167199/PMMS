# -*- coding: utf-8 -*-
# @Time : 2022/3/18 16:40
# @Author :
# @Site :
# @File : MLP_Fusion.py
# @Software: PyCharm

from load_data import load_data
import random
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
from config import CONFIG

cfg = CONFIG()
seq = cfg.seq
ngram = cfg.ngram
num_hidden1 = cfg.num_hidden1
num_hidden2 = cfg.num_hidden2
num_epochs = cfg.mlp_epoch

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_random_seed(1)

x, y = load_data(seq, "concat", ngram)
y = to_categorical(y, num_classes=None)

kf = KFold(n_splits=5, shuffle=True, random_state=2022)

n_inputs = x.shape[1]
n_outputs = 2

class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, num_hidden1, num_hidden2):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(n_inputs, num_hidden1)
        self.act1 = nn.Sigmoid()
        self.hidden2 = nn.Linear(num_hidden1, num_hidden2)
        self.act2 = nn.Sigmoid()
        self.interaction = nn.Linear(num_hidden1+num_hidden2, n_outputs)

    def forward(self, x):
        x1 = self.hidden1(x)
        x1 = self.act1(x1)
        x2 = self.hidden2(x1)
        x2 = self.act2(x2)

        cat_vector = torch.cat((x1, x2), 1)
        outputs = self.interaction(cat_vector)
        outputs = torch.sigmoid(outputs)

        return outputs

def evaluate_accuracy(X, y, model, epoch):
    prob_all = []
    lable_all = []
    prob_max = []

    pred = model(X)

    pred1 = pred.detach().numpy()
    prob_all.extend(np.argmax(pred1, axis=1))
    prob_max.extend(pred[:,1].detach().numpy())

    lable_all.extend(np.argmax(y, axis=1))

    if epoch == 199:
        auc = []
        for i in range(len(prob_max)):
            value = []
            value.append(prob_max[i])
            value.append(lable_all[i])
            auc.append(value)

        file = open('../output/cv/1-' + str(k+1) + '.txt', 'w')
        for row in auc:
            rowtext = '{}\t{}'.format(row[0], row[1])
            file.write(rowtext)
            file.write('\n')
        file.close()

    return accuracy_score(lable_all, prob_all), f1_score(lable_all, prob_all), roc_auc_score(lable_all, prob_max)


def train(X_train, X_test, y_train, y_test, model, loss, num_epochs, optimizer):

    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    train_data = [(a, b) for a, b in zip(X_train, y_train)]
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)

    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_loader:
            pred = model(X)
            l = loss(pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()

            pred_train = []
            pred_train.extend(np.argmax(pred.detach().numpy(), axis=1))
            pred_lable = []
            pred_lable.extend(np.argmax(y.detach().numpy(), axis=1))

            batch_count += 1
        test_acc, f1, auc = evaluate_accuracy(X_test, y_test, model, epoch)
        print(
            'epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
            % (epoch + 1, train_l_sum / batch_count, accuracy_score(pred_lable, pred_train),
               test_acc))

    return test_acc, f1, auc


model = MLP(n_inputs, n_outputs, num_hidden1, num_hidden2)
print(model)

loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

test_acc_score = []
test_f1_score = []
test_auc_score = []
result = []

for k, (train_set, test_set) in enumerate(kf.split(x, y)):
    print("%s-th round of validation:" % (k + 1))
    X_train, X_test, y_train, y_test = x[train_set], x[test_set], y[train_set], y[test_set]
    print("train_split_rate:", len(X_train) / len(x))

    acc, f1, auc = train(X_train, X_test, y_train, y_test, model, loss, num_epochs, optimizer)
    test_acc_score.append(acc)
    test_f1_score.append(f1)
    test_auc_score.append(auc)


result.append(np.mean(test_acc_score))
result.append(np.mean(test_f1_score))
result.append(np.mean(test_auc_score))
for i in result:
    print(i)