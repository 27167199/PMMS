# -*- coding: utf-8 -*-
# @Time : 2022/5/3 9:39
# @Author :
# @Site : 
# @File : Dimension reduction.py
# @Software: PyCharm

from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_data

def t_SNE_2D():
    target = []  # 保存标签

    E, y = load_data('long', "拼接", 4)  # print(target)
    for i in y:
        if i == 1:
            target.append(1)
        else:
            target.append(0)

    print(np.shape(E))

    # 开始降维
    X_tsne = TSNE(n_components=2, random_state=313).fit_transform(E)  # T-SNE降维
    X_pca = PCA(n_components=2, random_state=233).fit_transform(E)  # PCA降维
    reducer = umap.UMAP(n_neighbors=5, n_components=2, random_state=1323).fit_transform(E)  # UMAP降维

    colors = ['r', 'g']
    # Label_name = ['Non-Crystalization','Crystalization']
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    for i in np.unique(target):
        mask = target == i
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=colors[i], label=None, alpha=0.5)
    plt.title('TSNE')

    plt.subplot(132)
    for i in np.unique(target):
        mask = target == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], label=None, alpha=0.5)
    plt.title('PCA')

    plt.subplot(133)
    for i in np.unique(target):
        mask = target == i
        plt.scatter(reducer[mask, 0], reducer[mask, 1], c=colors[i], label=None, alpha=0.5)
    plt.title('UMAP')


    # 保存图片
    plt.savefig('../output/Plt/Dimension PMMS.eps', dpi=1000, format='eps')
    plt.savefig('../output/Plt/Dimension PMMS.jpeg', dpi=1000, format='jpeg')
    plt.show()

t_SNE_2D()