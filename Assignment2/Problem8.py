import random

import numpy as np


def calculate_derivatives_sse(prev_weights, data, correct_outputs):
    derivatives = []
    for k in range(len(prev_weights)):
        derivative = 0
        for i in range(len(data)):
            d = 0
            x_i = data[i]
            y_i = correct_outputs[i]
            for j in range(len(x_i)):
                d += prev_weights[j] * x_i[j]
            d -= y_i
            d *= x_i[k]
            derivative += d
        derivative *= 2
        derivatives.append(derivative)
    return derivatives


def calculate_derivatives_ssd(prev_weights, data, correct_outputs):
    derivatives = []
    for k in range(len(prev_weights)):
        derivative = 0
        for i in range(len(data)):
            x_i = data[i]
            y_i = correct_outputs[i]
            b = 0  # sum of squares of w_j
            a = -y_i  # prediction - y_i
            for j in range(len(x_i)):
                b += prev_weights[j] ** 2
                a += prev_weights[j] * x_i[j]
            derivative += (a * (x_i[k] * b - prev_weights[k] * a)) / (b ** 2)
        derivative *= 2
        derivatives.append(derivative)
    return derivatives


def grad_descent_iter(prev_weights, data, correct_outputs, learning_rate, objective_function):
    if objective_function is "sse":
        delta = calculate_derivatives_sse(prev_weights, data, correct_outputs)
    elif objective_function is "ssd":
        delta = calculate_derivatives_ssd(prev_weights, data, correct_outputs)
    else:
        print("Invalid Objective function.")
        return
    new_weights = []
    for i in range(len(prev_weights)):
        new_weights.append(prev_weights[i] - learning_rate * delta[i])
    return new_weights


def initialize_weights(k):
    return [random.random() for _ in range(k)]


def learn_weights(data, correct_outputs, objective_function, learning_rate=0.00001, iterations=1000):
    weights = initialize_weights(len(data[0]))
    for i in range(iterations):
        new_weights = grad_descent_iter(weights, data, correct_outputs, learning_rate, objective_function)
        diff = abs(sum([new_weights[k] - weights[k] for k in range(len(weights))]))
        weights = new_weights
        if diff <= 0.0000001:
            break
        # print(weights)
    return weights


def learn_weights_closed_form(data, correct_outputs):
    X = np.array(data)
    Y = np.array(correct_outputs)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), Y)


# don't forget to add ones in the first column of data.
def generate_random_data(original_weights, std_dev=0.0, num_points=100):
    data = []
    for i in range(num_points):
        x_i = [1]
        for i in range(len(original_weights) - 1):
            x_i.append(10 * random.random())
        data.append(x_i)

    correct_outputs = []
    for x_i in data:
        correct_outputs.append(sum([x_i[j] * original_weights[j] + std_dev * random.random() for j in range(len(x_i))]))

    return data, correct_outputs


def predict(data, weights):
    pred = []
    for xi in data:
        pred.append(sum([xi[j] * weights[j] for j in range(len(xi))]))
    return pred


def calc_r2(actual_output, predicted_output):
    ss_res = 0
    for i in range(len(actual_output)):
        ss_res += (actual_output[i] - predicted_output[i]) ** 2

    mean_actual = 0
    for y in actual_output:
        mean_actual += y
    mean_actual /= len(actual_output)

    ss_tot = 0
    for y in actual_output:
        ss_tot += (y - mean_actual) ** 2

    return 1 - (ss_res / ss_tot)


random.seed = 1
w = [1, 2, 3, 4]
data, outputs = generate_random_data(w, num_points=100, std_dev=0.5)
print("Original weights", w)
learned_weights_closed_form = learn_weights_closed_form(data, outputs)
print("Closed Form")
print("Learned weights", learned_weights_closed_form)
print("R2", calc_r2(outputs, predict(data, learned_weights_closed_form)))
# plt.plot(np.array(data)[:, 1], outputs)
learned_weights_sse = learn_weights(data, outputs, objective_function="sse", learning_rate=0.00001, iterations=10000)
print("SSE")
print("Learned weights", learned_weights_sse)
print("R2", calc_r2(outputs, predict(data, learned_weights_sse)))
learned_weights_ssd = learn_weights(data, outputs, objective_function="ssd", learning_rate=0.00001, iterations=10000)
print("SSD")
print("Learned weights", learned_weights_ssd)
print("R2", calc_r2(outputs, predict(data, learned_weights_ssd)))
# data, outputs = generate_random_data(learned_weights, num_points=10)
# plt.plot(np.array(data)[:, 1], outputs)
# plt.show()

# Original weights [1, 2, 3, 4]
# Closed Form
# Learned weights [2.02271925 1.97841801 3.01558258 4.00074691]
# R2 0.9997030883657995
# SSE
# Learned weights [1.8995071661015586, 1.9857066706075073, 3.0233519580556356, 4.008558459319282]
# R2 0.9996960847774278
# SSD
# Learned weights [2.333128758543021, 2.0538434231897953, 2.790335606860405, 4.092240408037085]
# R2 0.9980189478657999
