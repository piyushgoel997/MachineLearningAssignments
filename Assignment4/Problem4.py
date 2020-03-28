import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

start_time = time.time()
np.random.seed(101)
data1 = pd.read_csv("Datasets/data_banknote_authentication.txt", header=None)
data2 = pd.read_csv("Datasets/spambase.data", header=None)
data3 = pd.read_csv("Datasets/adult.data", header=None)
enc = OrdinalEncoder()
data3 = pd.DataFrame(enc.fit_transform(data3.sample(5000)))  # considering less examples to save time


def sigmoid(X, w):
    return 1 / (1 + np.exp(-np.dot(X, w)))


def log_reg(X, Y, eta=1, max_steps=100):
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
    # print(X.shape)
    # print(Y.shape)
    # w = np.random.randn(X.shape[1], 1)
    step = 0
    Y = np.array(Y)
    while step < max_steps:
        p = sigmoid(X, w)
        inv = np.dot(np.transpose(X), np.dot(np.diagflat(np.multiply(p, 1 - p)), X))
        if np.linalg.det(inv) == 0: break
        temp = np.linalg.inv(inv)
        w_new = w + eta * np.dot(temp, np.dot(np.transpose(X), Y - p))
        w = w_new
        step += 1
    return w


def calc_accuracy(y_test, test_pred, thresh=0.5):
    correct = 0
    y_test_ = list(y_test)
    test_pred_ = list(test_pred)
    for i in range(y_test.shape[0]):
        if test_pred_[i] > thresh:
            if y_test_[i] == 1:
                correct += 1
        else:
            if y_test_[i] == 0:
                correct += 1
    return correct / y_test.shape[0]


def roc(y_test, probs):
    roc = [(0, 0)]
    prev_c = None
    zeros = 0
    ones = 0
    total_ones = sum(list(y_test))
    total_zeros = len(list(y_test)) - total_ones
    for _, c in sorted(zip(probs, list(y_test)), reverse=True):
        if not (prev_c is None or prev_c == c):
            tpr = ones / total_ones
            fpr = zeros / total_zeros
            roc.append((fpr, tpr))
        if c == 0:
            zeros += 1
        else:
            ones += 1
        prev_c = c
    roc.append((1, 1))
    return roc


def auc(roc):
    area = 0
    prev = 0
    for a, b in roc:
        area += (b - prev) * (1 - a)
        prev = b
    return area


def cross_val(name, data, k=10, normalizer=None, preprocessor=None):
    X = data.sample(data.shape[0])
    Y = X[X.shape[1] - 1]
    X = X.drop([X.shape[1] - 1], axis=1)
    accuracies = []
    aucs = []
    for i in range(k):
        start = int(X.shape[0] * (i / k))
        end = int(X.shape[0] * ((i + 1) / k))
        test_data = X[start:end]
        y_test = Y[start:end]
        train_data = pd.concat([X[:start], X[end:]])
        y_train = pd.concat([Y[:start], Y[end:]])
        if normalizer is not None:
            train_data, test_data = normalizer(train_data, test_data)
        if preprocessor is not None:
            train_data, test_data = preprocessor(train_data, test_data)
        w = log_reg(train_data, y_train)
        prob = sigmoid(test_data, w)
        acc = calc_accuracy(y_test, prob)
        curve = roc(y_test, prob)
        area = auc(curve)
        # print(i + 1, acc, area)
        aucs.append(area)
        accuracies.append(acc)
        x_axis = [x for (x, y) in curve]
        y_axis = [y for (x, y) in curve]
        plt.plot(x_axis, y_axis)
    plt.savefig("Plots/" + name)
    plt.clf()
    return sum(accuracies) / k, sum(aucs) / k


print("No normalization and pre-processing")

cv = cross_val("Banknote.jpg", data1)
print("Dataset: banknote authentication: Avg AUC =", str(round(cv[1], 5)))

cv = cross_val("Spambase.jpg", data2)
print("Dataset: spambase: Avg AUC =", str(round(cv[1], 5)))

cv = cross_val("Adult.jpg", data3)
print("Dataset: adults with over 50k salary: Avg AUC =", str(round(cv[1], 5)))


############################################################################################################
# z-score normalization

def z_score_normalization(data1, data2):
    new_data1 = []
    new_data2 = []
    for col1, col2 in zip(data1, data2):
        curr = list(data1[col1])
        mean = 0
        ct = 0
        for x in curr:
            mean += x
            ct += 1
        mean = mean / ct
        std_dev = 0
        for x in curr:
            std_dev += (x - mean) ** 2
        std_dev = std_dev / ct
        std_dev = math.sqrt(std_dev)
        upd = [(x - mean) / std_dev for x in curr]
        new_data1.append(upd)
        upd = [(x - mean) / std_dev for x in list(data2[col2])]
        new_data2.append(upd)

    return pd.DataFrame(np.transpose(np.array(new_data1))), pd.DataFrame(np.transpose(np.array(new_data2)))


print("z-score normalization")

cv = cross_val("Banknote_Z_Score.jpg", data1, normalizer=z_score_normalization)
print("Dataset: banknote authentication: Avg AUC =", str(round(cv[1], 5)))

cv = cross_val("Spambase_Z_Score.jpg", data2, normalizer=z_score_normalization)
print("Dataset: spambase: Avg AUC =", str(round(cv[1], 5)))

cv = cross_val("Adult_Z_Score.jpg", data3, normalizer=z_score_normalization)
print("Dataset: adults with over 50k salary: Avg AUC =", str(round(cv[1], 5)))


############################################################################################################
# PCA with zero mean normalization

def zero_mean_normalization(data1, data2):
    new_data1 = []
    new_data2 = []
    for col1, col2 in zip(data1, data2):
        curr = list(data1[col1])
        mean = 0
        ct = 0
        for x in curr:
            mean += x
            ct += 1
        mean = mean / ct
        upd = [x - mean for x in curr]
        new_data1.append(upd)
        upd = [x - mean for x in list(data2[col2])]
        new_data2.append(upd)
    return pd.DataFrame(np.transpose(np.array(new_data1))), pd.DataFrame(np.transpose(np.array(new_data2)))


def cov_matrix(data):
    X = np.array(data)
    cols = X.shape[1]
    cov = np.zeros((cols, cols))
    for i in range(cols):
        for j in range(cols):
            cov[i, j] = sum(X[:, i] * X[:, j]) / X.shape[0]
    return cov


def PCA(new_data, retain_variance_fraction=0.99):
    cov = cov_matrix(new_data)
    W, V = np.linalg.eig(cov)
    var = 0
    chosen = []
    retain_variance = sum(list(W))
    for w, v in sorted(zip(W, V), reverse=True):
        if var > retain_variance * retain_variance_fraction:
            break
        var += w
        chosen.append(v)
    print("Number of chosen Columns = ", len(chosen), "Total number of Columns = ", len(W))
    return np.array(chosen)


def PCA_processor(data1, data2):
    V = PCA(data1)
    return np.dot(data1, V.T), np.dot(data2, V.T)


print("PCA with zero mean normalization")

cv = cross_val("Banknote_PCA_Zero_Mean.jpg", data1, normalizer=zero_mean_normalization, preprocessor=PCA_processor)
print("Dataset: banknote authentication: Avg AUC =", str(round(cv[1], 5)))

cv = cross_val("Spambase_PCA_Zero_Mean.jpg", data2, normalizer=zero_mean_normalization, preprocessor=PCA_processor)
print("Dataset: spambase: Avg AUC =", str(round(cv[1], 5)))

cv = cross_val("Adult_PCA_Zero_Mean.jpg", data3, normalizer=zero_mean_normalization, preprocessor=PCA_processor)
print("Dataset: adults with over 50k salary: Avg AUC =", str(round(cv[1], 5)))

######################################################################################################
# PCA with z-score normalization

print("PCA with z-score normalization")

cv = cross_val("Banknote_PCA_Z_Score.jpg", data1, normalizer=z_score_normalization, preprocessor=PCA_processor)
print("Dataset: banknote authentication: Avg AUC =", str(round(cv[1], 5)))

cv = cross_val("Spambase_PCA_Z_Score.jpg", data2, normalizer=z_score_normalization, preprocessor=PCA_processor)
print("Dataset: spambase: Avg AUC =", str(round(cv[1], 5)))

cv = cross_val("Adult_PCA_Z_Score.jpg", data3, normalizer=z_score_normalization, preprocessor=PCA_processor)
print("Dataset: adults with over 50k salary: Avg AUC =", str(round(cv[1], 5)))

print("Total time taken:", str(time.time() - start_time))
