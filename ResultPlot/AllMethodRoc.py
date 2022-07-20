# -*- coding: utf-8 -*-
# @Time : 2022/5/3 9:17
# @Author :
# @Site : 
# @File : ResultPlot.py
# @Software: PyCharm

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from scipy import interpolate
from numpy import interp
import matplotlib.pyplot as plt

#
def read_one_col(filepath, col, header=False, sep="\t", encoding='utf-8'):
    """
    read a column from a table file
    :param filepath: a path of a table file which contains at least two columns
    :param header: a boolean variable indicates if the first row is a head or not,
    default False
    :param encoding: encoding
    :param sep: delimiter, a string, default "\t"
    :param col: the number of col needs to be extracted
    :return: a list contains elements from the extracted column
    """
    eles = []
    col -= 1
    #with open(filepath, encoding=encoding, mode='r') as f:
    with open(filepath, "r") as f:
        if header:
            next(f)
        for line in f:
            words = line.strip().split(sep)
            if len(words) > col:
                eles.append(float(words[col].strip()))
    return eles


number = 0
auc_val = 0
acc_val = 0
f1_val = 0

titlename = "5-fold CV"
methodnum = 6
indexmeth = 1

color = False

while indexmeth < methodnum:
    if indexmeth == 1: # MLP 0.9557
        # 用输出的预测值和标签重新计算画
        number = 0
        auc_val = 0
        acc_val = 0
        f1_val = 0
        for i in range(1, 2):
            for j in range(1, 6):
                number = number + 1
                filename = "MethodResult/PMMS/" + str(i) + "-" + str(j) + ".txt"
                # filename = "F:/PyProjects/Bio_test/ml_predict_lable/Gaus_NB/" + str(i) + "-" + str(j) + ".txt"

                predict = read_one_col(filename, 1, False, '\t')
                labels = read_one_col(filename, 2, False, '\t')

                fpr, tpr, auc_thresholds = roc_curve(labels, predict)
                auc_val = auc_val + auc(fpr, tpr)

                # show ROC curve
                if number == 1:
                    mean_tpr_Gaus = 0.0
                    mean_fpr_Gaus = np.linspace(0, 1, 20)
                    # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
                mean_tpr_Gaus += interp(mean_fpr_Gaus, fpr, tpr)
                # show ROC curve

                precision, recall, pr_thresholds = precision_recall_curve(labels, predict)
                all_F_measure = np.zeros(len(pr_thresholds))
                for k in range(0, len(pr_thresholds)):
                    if (precision[k] + precision[k]) > 0:
                        all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
                    else:
                        all_F_measure[k] = 0
                max_index = all_F_measure.argmax()
                threshold = pr_thresholds[max_index]

                predicted_score = np.zeros(len(labels))
                predicted_score[predict > threshold] = 1

                f1_val = f1_val + f1_score(labels, predicted_score)
                acc_val = acc_val + accuracy_score(labels, predicted_score)

        # show ROC curve
        mean_tpr_Gaus[0] = 0.0  # 初始处为0
        mean_tpr_Gaus /= number  # 在每个点处插值插值多次取平均
        mean_tpr_Gaus[-1] = 1.0  # 坐标最后一个点为（1,1）
        mean_auc = auc(mean_fpr_Gaus, mean_tpr_Gaus)  # 计算平均AUC值

        if color:
            plt.plot(mean_fpr_Gaus, mean_tpr_Gaus, '-', color='blue',
                     label='PMMS (AUC=%0.4f)' % (auc_val / number), lw=1)
        else:
            plt.plot(mean_fpr_Gaus, mean_tpr_Gaus, '-', color='black',
                     label='PMMS (AUC=%0.4f)' % (auc_val / number), lw=1)
        print("PMMS Average accuracy:{:.6f}, f1:{:.6f}, auc:{:.6f}\n".
              format((acc_val / number), (f1_val / number), (auc_val / number)))

    if indexmeth == 2: # PESM 0.9117
        # 用输出的预测值和标签重新计算画
        number = 0
        auc_val = 0
        acc_val = 0
        f1_val = 0
        for i in range(1, 6):
            for j in range(1, 6):
                number = number + 1
                filename = "MethodResult/xgboost-data1/" + str(i) + "-" + str(j) + ".txt"

                predict = read_one_col(filename, 1, False, '\t')
                labels = read_one_col(filename, 2, False, '\t')

                fpr, tpr, auc_thresholds = roc_curve(labels, predict)
                auc_val = auc_val + auc(fpr, tpr)

                # show ROC curve
                if number == 1:
                    mean_tpr_boost = 0.0
                    # mean_fpr_miES = np.linspace(0, 1, len(fpr))
                    mean_fpr_boost = np.linspace(0, 1, 20)
                    # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
                mean_tpr_boost += interp(mean_fpr_boost, fpr, tpr)
                # show ROC curve

                precision, recall, pr_thresholds = precision_recall_curve(labels, predict)
                all_F_measure = np.zeros(len(pr_thresholds))
                for k in range(0, len(pr_thresholds)):
                    if (precision[k] + precision[k]) > 0:
                        all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
                    else:
                        all_F_measure[k] = 0
                max_index = all_F_measure.argmax()
                threshold = pr_thresholds[max_index]

                predicted_score = np.zeros(len(labels))
                predicted_score[predict > threshold] = 1

                f1_val = f1_val + f1_score(labels, predicted_score)
                acc_val = acc_val + accuracy_score(labels, predicted_score)

        # show ROC curve
        mean_tpr_boost[0] = 0.0  # 初始处为0
        mean_tpr_boost /= number  # 在每个点处插值插值多次取平均
        mean_tpr_boost[-1] = 1.0  # 坐标最后一个点为（1,1）
        mean_auc = auc(mean_fpr_boost, mean_tpr_boost)  # 计算平均AUC值

        # plt.plot(mean_fpr_miES, mean_tpr_miES, '-', color='red',
        #        label='miES (AUC=%0.4f)' % (0.8876), lw=1)
        # plt.plot(mean_fpr_miES, mean_tpr_miES, '-', color='red',
        #        label='miES (AUC=%0.4f)' % mean_auc, lw=1)
        # plt.plot(mean_fpr_boost, mean_tpr_boost, '-', color='blue',
        #          label='PESM (AUC=%0.4f)' % (auc_val / number), lw=1)
        if color:
            plt.plot(mean_fpr_boost, mean_tpr_boost, '-', color='red',
                     label='PESM (AUC=%0.4f)' % (0.9117), lw=1)
        else:
            plt.plot(mean_fpr_boost, mean_tpr_boost, '-.', color='black',
                     label='PESM (AUC=%0.4f)' % (0.9117), lw=1)
        print("XGboost Average accuracy:{:.6f}, f1:{:.6f}, auc:{:.6f}\n".
              format((acc_val / number), (f1_val / number), (auc_val / number)))

    if indexmeth == 3: # miES 0.8837
        number = 0
        auc_val = 0
        acc_val = 0
        f1_val = 0
        for i in range(1,21):
            for j in range(1,6):
                number = number + 1
                filename = "MethodResult/LgTopython/"+ str(i) + "-" + str(j) + ".txt"
                predict = read_one_col(filename, 1, False, '\t')
                labels = read_one_col(filename, 2, False, '\t')

                fpr, tpr, auc_thresholds = roc_curve(labels, predict)
                auc_val = auc_val + auc(fpr, tpr)

                #show ROC curve
                if number == 1:
                    mean_tpr_miES = 0.0
                    # mean_fpr_miES = np.linspace(0, 1, len(fpr))
                    mean_fpr_miES = np.linspace(0, 1, 20)
                    # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
                mean_tpr_miES += interp(mean_fpr_miES, fpr, tpr)
                # show ROC curve



                precision, recall, pr_thresholds = precision_recall_curve(labels, predict)
                all_F_measure = np.zeros(len(pr_thresholds))
                for k in range(0, len(pr_thresholds)):
                    if (precision[k] + precision[k]) > 0:
                        all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
                    else:
                        all_F_measure[k] = 0
                max_index = all_F_measure.argmax()
                threshold = pr_thresholds[max_index]

                predicted_score = np.zeros(len(labels))
                predicted_score[predict > threshold] = 1

                f1_val = f1_val + f1_score(labels, predicted_score)
                acc_val = acc_val + accuracy_score(labels, predicted_score)

        # show ROC curve
        mean_tpr_miES[0] = 0.0  # 初始处为0
        mean_tpr_miES /= number  # 在每个点处插值插值多次取平均
        mean_tpr_miES[-1] = 1.0  # 坐标最后一个点为（1,1）
        mean_auc = auc(mean_fpr_miES, mean_tpr_miES)  # 计算平均AUC值

        # plt.plot(mean_fpr_miES, mean_tpr_miES, '-', color='red',
        #        label='miES (AUC=%0.4f)' % (0.8876), lw=1)
        # plt.plot(mean_fpr_miES, mean_tpr_miES, '-', color='red',
        #        label='miES (AUC=%0.4f)' % mean_auc, lw=1)
        if color:
            plt.plot(mean_fpr_miES, mean_tpr_miES, '-', color='peru',
                     label='miES (AUC=%0.4f)' % (auc_val / number), lw=1)
        else:
            plt.plot(mean_fpr_miES, mean_tpr_miES, ':', color='black',
                     label='miES (AUC=%0.4f)' % (auc_val / number), lw=1)
        print("miES Average accuracy:{:.6f}, f1:{:.6f}, auc:{:.6f}\n".
            format((acc_val/number), (f1_val/number), (auc_val/number)))

    if indexmeth == 4: #Gaus_NB 0.8720
        # 用输出的预测值和标签重新计算画
        number = 0
        auc_val = 0
        acc_val = 0
        f1_val = 0
        for i in range(1, 6):
            for j in range(1, 6):
                number = number + 1
                filename = "MethodResult/Gaus_NB/" + str(i) + "-" + str(j) + ".txt"
                # filename = "F:/PyProjects/Bio_test/ml_predict_lable/Gaus_NB/" + str(i) + "-" + str(j) + ".txt"

                predict = read_one_col(filename, 1, False, '\t')
                labels = read_one_col(filename, 2, False, '\t')

                fpr, tpr, auc_thresholds = roc_curve(labels, predict)
                auc_val = auc_val + auc(fpr, tpr)

                # show ROC curve
                if number == 1:
                    mean_tpr_Gaus = 0.0
                    mean_fpr_Gaus = np.linspace(0, 1, 20)
                    # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
                mean_tpr_Gaus += interp(mean_fpr_Gaus, fpr, tpr)
                # show ROC curve



                precision, recall, pr_thresholds = precision_recall_curve(labels, predict)
                all_F_measure = np.zeros(len(pr_thresholds))
                for k in range(0, len(pr_thresholds)):
                    if (precision[k] + precision[k]) > 0:
                        all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
                    else:
                        all_F_measure[k] = 0
                max_index = all_F_measure.argmax()
                threshold = pr_thresholds[max_index]

                predicted_score = np.zeros(len(labels))
                predicted_score[predict > threshold] = 1

                f1_val = f1_val + f1_score(labels, predicted_score)
                acc_val = acc_val + accuracy_score(labels, predicted_score)

        # show ROC curve
        mean_tpr_Gaus[0] = 0.0  # 初始处为0
        mean_tpr_Gaus /= number  # 在每个点处插值插值多次取平均
        mean_tpr_Gaus[-1] = 1.0  # 坐标最后一个点为（1,1）
        mean_auc = auc(mean_fpr_Gaus, mean_tpr_Gaus)  # 计算平均AUC值

        if color:
            plt.plot(mean_fpr_Gaus, mean_tpr_Gaus, '-', color='gray',
                     label='Gaus_NB (AUC=%0.4f)' % (auc_val / number), lw=1)
        else:
            plt.plot(mean_fpr_Gaus, mean_tpr_Gaus, '--', color='black',
                     label='Gaus_NB (AUC=%0.4f)' % (auc_val / number), lw=1)
        print("Gaus_NB Average accuracy:{:.6f}, f1:{:.6f}, auc:{:.6f}\n".
            format((acc_val/number), (f1_val/number), (auc_val/number)))
    if indexmeth == 5: # SVM 0.8571
        # 用输出的预测值和标签重新计算画
        number = 0
        auc_val = 0
        acc_val = 0
        f1_val = 0
        for i in range(1, 6):
            for j in range(1, 6):
                number = number + 1
                filename = "MethodResult/SVM/" + str(i) + "-" + str(j) + ".txt"
                # filename = "F:/PyProjects/Bio_test/ml_predict_lable/SVM/" + str(i) + "-" + str(j) + ".txt"

                predict = read_one_col(filename, 1, False, '\t')
                labels = read_one_col(filename, 2, False, '\t')

                fpr, tpr, auc_thresholds = roc_curve(labels, predict)
                auc_val = auc_val + auc(fpr, tpr)

                # show ROC curve
                if number == 1:
                    mean_tpr_SVM = 0.0
                    mean_fpr_SVM = np.linspace(0, 1, 20)
                    # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
                mean_tpr_SVM += interp(mean_fpr_SVM, fpr, tpr)
                # show ROC curve

                precision, recall, pr_thresholds = precision_recall_curve(labels, predict)
                all_F_measure = np.zeros(len(pr_thresholds))
                for k in range(0, len(pr_thresholds)):
                    if (precision[k] + precision[k]) > 0:
                        all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
                    else:
                        all_F_measure[k] = 0
                max_index = all_F_measure.argmax()
                threshold = pr_thresholds[max_index]

                predicted_score = np.zeros(len(labels))
                predicted_score[predict > threshold] = 1

                f1_val = f1_val + f1_score(labels, predicted_score)
                acc_val = acc_val + accuracy_score(labels, predicted_score)

        # show ROC curve
        mean_tpr_SVM[0] = 0.0  # 初始处为0
        mean_tpr_SVM /= number  # 在每个点处插值插值多次取平均
        mean_tpr_SVM[-1] = 1.0  # 坐标最后一个点为（1,1）
        mean_auc = auc(mean_fpr_SVM, mean_tpr_SVM)  # 计算平均AUC值

        if color:
            plt.plot(mean_fpr_SVM, mean_tpr_SVM, '-', color='green',
                     label='SVM (AUC=%0.4f)' % (auc_val / number), lw=1)
        else:
            plt.plot(mean_fpr_SVM, mean_tpr_SVM, linestyle=(0,(3,1,1,1,1,1)), color='black',
                     label='SVM (AUC=%0.4f)' % (auc_val / number), lw=1)
        print("SVM Average accuracy:{:.6f}, f1:{:.6f}, auc:{:.6f}\n".
            format((acc_val/number), (f1_val/number), (auc_val/number)))

    indexmeth += 1


plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(titlename)
plt.legend(loc="lower right")

if color:
    savefile = '../output/Plt/ROC_PMMS_Color'
else:
    savefile = '../output/Plt/ROC_PMMS_Gray'
plt.savefig(savefile + '.eps', dpi=1000, format='eps')
plt.savefig(savefile + '.jpeg', dpi=1000, format='jpeg')
plt.show()
#show ROC curve
#print("AUC:%g,acc:%g,f1:%g"%(auc_val/number,acc_val/number,f1_val/number))
