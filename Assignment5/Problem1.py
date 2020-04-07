import numpy as np
from sklearn.metrics import roc_curve, auc

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


def create_dataset(n, part="A"):
    X = np.random.rand(n, 2)
    X = np.multiply(np.array([12, 8]), X) - np.array([6, 4])
    Y = []
    for x in X:
        if part == "A":
            if (-4 <= x[0] <= -1 and 0 <= x[1] <= 3) or (-2 <= x[0] <= 1 and -4 <= x[1] <= -1) or (
                    2 <= x[0] <= 5 and -2 <= x[1] <= 1):
                Y.append(1)
            else:
                Y.append(-1)
        else:
            if (-4 <= x[0] <= -3 and 2 <= x[1] <= 3) or (-1 <= x[0] <= 0 and -3 <= x[1] <= -2) or (
                    2 <= x[0] <= 3 and -1 <= x[1] <= 0):
                Y.append(1)
            else:
                Y.append(-1)
    return (X, np.array(Y))


def get_model(h1, h2):
    model = Sequential()
    model.add(Dense(h1, activation="tanh"))
    if h2 != 0:
        model.add(Dense(h2, activation="tanh"))
    model.add(Dense(1, activation="tanh"))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def k_fold_cross_validation(model, data, k=10):
    X, Y = data
    avg_accuracy_val = 0
    avg_roc_auc = 0
    roc_curves = []
    for i in range(k):
        print("========Cross Validation:", str(i + 1) + "/" + str(k) + "========")
        start = int(X.shape[0] * (i / k))
        end = int(X.shape[0] * ((i + 1) / k))
        test_data = X[start:end]
        y_test = Y[start:end]
        train_data = np.concatenate([X[:start], X[end:]])
        y_train = np.concatenate([Y[:start], Y[end:]])
        history = model.fit(train_data, y_train, validation_data=(test_data, y_test), batch_size=1000, epochs=10000, verbose=0)
        test_out = model.predict(test_data)
        fpr, tpr, _ = roc_curve(y_test, test_out)
        roc_curves.append((fpr, tpr))
        avg_roc_auc += auc(fpr, tpr)
        avg_accuracy_val += history.history['val_accuracy'][-1]

    return {"acc": avg_accuracy_val / k, "roc_auc": avg_roc_auc / k, "roc": roc_curves}


data = create_dataset(1000)

for h1 in [1, 4, 8]:
    for h2 in [0, 3]:
        model = get_model(h1, h2)
        result = k_fold_cross_validation(model, data)
        print(result["acc"], result["roc_auc"])
        del model
