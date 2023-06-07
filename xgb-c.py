import xgboost
from sklearn.metrics import confusion_matrix
import input_data
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import math
import os
from numpy import *
# from imblearn.over_sampling import SMOTE
# from collections import Counter
import xlwt
import warnings
warnings.filterwarnings("ignore")


def eva(pre, true):
    pre = np.reshape(pre , (-1, 1))
    matrix = confusion_matrix(true, pre, labels=[1, 0])
    TP = matrix[0][0]
    TN = matrix[1][1]
    FP = matrix[1][0]
    FN = matrix[0][1]
    SE = 1.0*  TP  / (TP + FN)
    SP = 1.0 * TN / (TN + FP)
    ACC = 1.0*(TP + TN) / (TP + TN + FP + FN)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    F1 = 2 * TP / (2 * TP + FP + FN)
    return TP, TN, FP, FN, SE, SP, ACC, MCC, F1


file_root_path = 'dataset'
file_num_path = os.listdir(file_root_path)
TP_list = []
TN_list = []
FP_list = []
FN_list = []
SE_list = []
SP_list = []
ACC_list = []
MCC_list = []
F1_list = []
AUC_list = []
setFileNames = []
Optimal_parameters = [(10, 3), (10, 3), (8, 3), (8, 1), (10, 3), (8, 1), (8, 1), (10, 1), (10, 5), (8, 1), (10, 3),
                      (8, 1), (8, 3), (8, 1), (10, 1), (10, 3), (8, 3), (10, 1)]  # max_depth	min_child_weight
for file_num in file_num_path:
    setFileNames.clear()
    train_datadir = file_root_path + '/' + str(file_num)
    csv_file_list = os.listdir(train_datadir)
    setFileNames.append(csv_file_list[1])
    setFileNames.append(csv_file_list[0])
    print(setFileNames)
    huang = input_data.read_data_sets(train_datadir, setFileNames=setFileNames, one_hot=True)
    trX, trY, k_X, k_Y, teX, teY = huang.train.images, huang.train.labels, huang.k_train.images, huang.k_train.labels, huang.test.images, \
                                   huang.test.labels
    print('(max_depth, min_child_weight): ', Optimal_parameters[int(file_num) - 1])
    max_depth_param = Optimal_parameters[int(file_num)-1][0]
    min_child_weight_param = Optimal_parameters[int(file_num)-1][1]
    model = XGBClassifier(learning_rate=0.01,  # 学习速率
                          n_estimators=1000,
                          max_depth=max_depth_param,  # 树的最大深度，这个参数的取值最好在3-10之间， 需要使用CV函数来进行调优
                          min_child_weight=min_child_weight_param,  # 决定最小叶子节点样本权重和， 这个参数用于避免过拟合，使用 CV来调整
                          gamma=0,
                          subsample=0.8,
                          colsample_bytree=0.8,
                          objective='binary:logistic',
                          nthread=4,
                          scale_pos_weight=1,
                          seed=None)
    model.fit(trX, trY)

    # pred_y = model.predict(trX)
    # TP, TN, FP, FN, SE, SP,  ACC, MCC, = eva(pred_y, trY)
    # print("train:")
    # print("TP:", TP)
    # print("TN:", TN)
    # print("FP:", FP)
    # print("FN:", FN)
    # print("SE:", SE)
    # print("SP:", SP)
    # print("ACC: ", ACC)
    # print("MCC: ", MCC)
    # print("F1: ", F1)
    # tr_auc =roc_auc_score(trY, pred_y)
    # print(tr_auc)
    # #
    model.fit(trX, trY)
    pred_y = model.predict(teX)
    TP, TN, FP, FN, SE, SP, ACC, MCC, F1 = eva(pred_y, teY)
    auc = roc_auc_score(teY, pred_y)
    TP_list.append(TP)
    TN_list.append(TN)
    FP_list.append(FP)
    FN_list.append(FN)
    SE_list.append(SE)
    SP_list.append(SP)
    ACC_list.append(ACC)
    MCC_list.append(MCC)
    F1_list.append(F1)
    AUC_list.append(auc)
tp_mean = mean(TP_list)
tn_mean = mean(TN_list)
fp_mean = mean(FP_list)
fn_mean = mean(FN_list)
se_mean = mean(SE_list)
sp_mean = mean(SP_list)
acc_mean = mean(ACC_list)
mcc_mean = mean(MCC_list)
f1_mean = mean(F1_list)
auc_mean = mean(AUC_list)
print("************************************************************")
print("test_list:")
print("TP_list:", TP_list)
print("TN_list:", TN_list)
print("FP_list:", FP_list)
print("FN_list:", FN_list)
print("SE_list:", SE_list)
print("SP_list:", SP_list)
print("ACC_list:", ACC_list)
print("MCC_list:", MCC_list)
print("F1_list:", F1_list)
print("AUC_list:", AUC_list)
print("***********************************************************")
print("test_mean:")
print("TP:", tp_mean)
print("TN:", tn_mean)
print("FP:", fp_mean)
print("FN:", fn_mean)
print("SE:", se_mean)
print("SP:", sp_mean)
print("ACC: ", acc_mean)
print("MCC: ", mcc_mean)
print("F1: ", f1_mean)
print("AUC：",auc_mean)



