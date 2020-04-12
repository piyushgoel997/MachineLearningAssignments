import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


def create_dataset(n, part="A", random=True):
    if random:
        X = np.random.rand(n, 2)
        X = np.multiply(np.array([12, 8]), X) - np.array([6, 4])
    else:
        X = []
        for i in range(int(-n / 2), int(n / 2) + 1):
            for j in range(int(-n / 2), int(n / 2) + 1):
                X.append([(i * 6) / int(n / 2), (j * 4) / int(n / 2)])

    Y = []
    for x in X:
        if part == "A":
            if (-4 <= x[0] <= -1 and 0 <= x[1] <= 3) or (-2 <= x[0] <= 1 and -4 <= x[1] <= -1) or (
                    2 <= x[0] <= 5 and -2 <= x[1] <= 1):
                Y.append(1)
            else:
                Y.append(0)
        else:
            if (-4 <= x[0] <= -3 and 2 <= x[1] <= 3) or (-1 <= x[0] <= 0 and -3 <= x[1] <= -2) or (
                    2 <= x[0] <= 3 and -1 <= x[1] <= 0):
                Y.append(1)
            else:
                Y.append(0)
    return np.array(X), np.array(Y)


def get_model(h1, h2, initializer="random_uniform"):
    model = Sequential()
    model.add(Dense(h1, activation="tanh", kernel_initializer=initializer))
    if h2 != 0:
        model.add(Dense(h2, activation="tanh", kernel_initializer=initializer))
    model.add(Dense(1, activation="tanh", kernel_initializer=initializer))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def mean_std_err(nos):
    return np.mean(nos), np.std(nos)


def prediction(scores, thresh=0.5):
    pred = np.zeros(scores.shape)
    pred[scores > thresh] = 1
    return pred


def k_fold_cross_validation(h1, h2, data, k=10, epochs=100):
    X, Y = data
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    roc_auc = np.zeros((100,))
    acc = np.zeros((100,))
    for train_idx, test_idx in skf.split(X, Y):
        train_data, test_data = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        model = get_model(h1, h2)
        model.fit(train_data, y_train, epochs=epochs, verbose=0)

        # bootstrapping
        roc_auc_ = []
        acc_ = []
        for b in range(100):
            d_b, y_true = sample_with_replacement(test_data, y_test, len(y_test))
            scores = model.predict(d_b)
            roc_auc[b] += 100 * auc(*roc_curve(y_true, scores)[0:2])
            y_pred = prediction(scores)
            acc[b] += 100 * accuracy_score(y_true, y_pred)

        del model

    acc = mean_std_err(acc/k)
    roc_auc = mean_std_err(roc_auc/k)
    return {"acc": acc[0], "acc_se": acc[1], "roc_auc": roc_auc[0], "roc_auc_se": roc_auc[1]}


def plot_heatmap_print_acc(model, part="A", filename=None):
    fine_data = create_dataset(500, random=False, part=part)
    scores = model.predict(fine_data[0], verbose=0)
    acc = accuracy_score(fine_data[1], prediction(scores))
    print("Accuracy =", str(round(100 * acc, 1)))
    c = np.concatenate((fine_data[0], scores), axis=1)
    c = pd.DataFrame(c, columns=["x1", "x2", "class"]).pivot("x2", "x1", "class")
    ax = sns.heatmap(pd.DataFrame(c), vmax=1.0)
    ax.invert_yaxis()
    plt.savefig(filename)
    plt.clf()


def sample_with_replacement(x, y, n):
    idx = random.choices(range(len(x)), k=n)
    while np.sum(y[idx]) == 0:
        idx = random.choices(range(len(x)), k=n)
    return x[idx], y[idx]



def k_fold_cross_validation_ensemble(h1, h2, data, k=10, epochs=100, num_models=10, decision="average", threshold=0.5,
                                     sample_train=False):
    X, Y = data
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    acc = np.zeros((100,))
    roc_auc = np.zeros((100,))
    for train_idx, test_idx in skf.split(X, Y):
        train_data, test_data = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        models = []
        x, y = train_data, y_train
        for a in range(num_models):
            if sample_train:
                x, y = sample_with_replacement(train_data, y_train, len(train_data))
            model = get_model(h1, h2)
            model.fit(x, y, epochs=epochs, verbose=0)
            models.append(model)

        for b in range(100):
            d_b, y_true = sample_with_replacement(test_data, y_test, len(y_test))

            scores = []
            for m in models:
                scores.append(m.predict(d_b))
            scores = np.array(scores)

            if decision == "average":
                scores = np.mean(scores, axis=0)
                y_pred = prediction(scores)
            elif decision == "majority_vote":
                y_pred = prediction(np.mean(prediction(scores), axis=0))
            else:
                print("Incorrect decision name")
                exit(-1)
            if decision != "majority_vote":
                roc_auc[b] += 100 * auc(*roc_curve(y_true, scores)[0:2])
            acc[b] += 100 * accuracy_score(y_true, y_pred)

        del models
    acc = mean_std_err(acc/k)
    roc_auc = mean_std_err(roc_auc/k)
    return {"acc": acc[0], "acc_se": acc[1], "roc_auc": roc_auc[0], "roc_auc_se": roc_auc[1]}
