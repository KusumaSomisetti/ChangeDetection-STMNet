import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score


def Predict_Label2Img(predict_label, img_gt):
    predict_img = torch.zeros_like(img_gt)
    num = predict_label.shape[0]

    for i in range(num):
        x = int(predict_label[i][1])
        y = int(predict_label[i][2])
        l = predict_label[i][3]
        predict_img[x][y] = l

    return predict_img


def PredictImg_addnolabel(predict_img, img_gt):
    predict_img_ = torch.zeros_like(predict_img)
    indices = torch.where(predict_img == 1)
    predict_img_[indices] = 2
    indices = torch.where(img_gt == 2)
    predict_img_[indices] = 1

    return predict_img_


def accuracy_assessment(img_gt, changed_map):
    esp = 1e-6
    height, width = changed_map.shape

    changed_map_ = np.reshape(changed_map, (-1,))
    img_gt_ = np.reshape(img_gt, (-1,))
    # print('np.unique(img_gt_)',np.unique(img_gt_))  #
    # print('np.unique(changed_map)', np.unique(changed_map))

    cm = np.ones((height * width,))
    cm[changed_map_ == 1] = 2
    cm[changed_map_ == 0] = 1

    gt = np.zeros((height * width,))
    gt[img_gt_ == 1] = 2
    gt[img_gt_ == 0] = 1

    conf_mat = confusion_matrix(y_true=gt, y_pred=cm, labels=[1, 2])
    kappa_co = cohen_kappa_score(y1=gt, y2=cm, labels=[1, 2])

    TN, FN, FP, TP = conf_mat.ravel()
    P = TP / (TP + FP + esp)
    R = TP / (TP + FN + esp)
    F1 = 2 * P * R / (P + R + esp)
    acc = (TP + TN) / (TP + TN + FP + FN + esp)

    oa = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)

    oa_0 = TN / (TN + FN + esp)  # unchange
    oa_1 = TP / (TP + FP + esp)  # change

    return conf_mat, oa, kappa_co, P, R, F1, acc, oa_0, oa_1


def aa_012(img_gt, changed_map):
    esp = 1e-6
    height, width = changed_map.shape
    img_gt_ = np.reshape(img_gt, (-1,))
    changed_map_ = np.reshape(changed_map, (-1,))

    cm = np.ones((height * width,))
    cm[img_gt_ == 2] = 3
    cm[changed_map_ == 1] = 2
    cm[changed_map_ == 0] = 1

    gt = np.zeros((height * width,))
    gt[img_gt_ == 2] = 3
    gt[img_gt_ == 1] = 2
    gt[img_gt_ == 0] = 1

    conf_mat = confusion_matrix(y_true=gt, y_pred=cm, labels=[1, 2])
    kappa_co = cohen_kappa_score(y1=gt, y2=cm, labels=[1, 2])

    # TN, FN, FP, TP
    TN, FN, FP, TP = conf_mat.ravel()
    P = TP / (TP + FP + esp)
    R = TP / (TP + FN + esp)
    F1 = 2 * P * R / (P + R + esp)
    acc = (TP + TN) / (TP + TN + FP + FN + esp)

    oa = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)

    oa_0 = TN / (TN + FN + esp)
    oa_1 = TP / (TP + FP + esp)

    return conf_mat, oa, kappa_co, P, R, F1, acc, oa_0, oa_1
