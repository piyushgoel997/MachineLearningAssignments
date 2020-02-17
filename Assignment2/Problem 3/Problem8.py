import random

import numpy as np


# function to calculate the derivatives of the sse objective function
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


# function to calculate the derivatives of the ssd objective function
def calculate_derivatives_ssd(prev_weights, data, correct_outputs):
    derivatives = []
    for k in range(len(prev_weights)):
        derivative = 0
        b = 1  # 1 + sum of squares of w_j
        for j in range(len(prev_weights)):
            if j is not 0:
                b += prev_weights[j] ** 2
        for i in range(len(data)):
            x_i = data[i]
            y_i = correct_outputs[i]
            a = -y_i  # prediction - y_i
            for j in range(len(x_i)):
                a += prev_weights[j] * x_i[j]
            if k is 0:
                derivative += a
            else:
                derivative += (a * (x_i[k] * b - prev_weights[k] * a)) / (b ** 2)
        derivative *= 2
        derivatives.append(derivative)
    return derivatives


# function to do just one iteration of the gradient descent algorithm
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


# a function to initialize weights by random values.
def initialize_weights(k):
    return [random.random() for _ in range(k)]


# learn weights using gradient descent algorithm
def learn_weights(data, correct_outputs, objective_function, learning_rate=0.00001, iterations=1000):
    weights = initialize_weights(len(data[0]))
    for i in range(iterations):
        new_weights = grad_descent_iter(weights, data, correct_outputs, learning_rate, objective_function)
        # diff = abs(sum([new_weights[k] - weights[k] for k in range(len(weights))]))
        weights = new_weights
        # if diff <= 0.000001:
        #     break
        # print(weights)
    return weights


# learn weights using the maximum likelyhood closed form solution
def learn_weights_closed_form(data, correct_outputs):
    X = np.array(data)
    Y = np.array(correct_outputs)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), Y)


# random data generator
def generate_random_data(original_weights, std_dev=0.0, num_points=100):
    data = []
    for i in range(num_points):
        x_i = [1]
        for i in range(len(original_weights) - 1):
            x_i.append(10 * random.random())
        data.append(x_i)

    correct_outputs = []
    for x_i in data:
        correct_outputs.append(
            sum([x_i[j] * original_weights[j] + std_dev * np.random.normal(0.0, std_dev) for j in range(len(x_i))]))

    return data, correct_outputs


# function to predict the output values using the input data and the weights to be used.
def predict(data, weights):
    pred = []
    for xi in data:
        pred.append(sum([xi[j] * weights[j] for j in range(len(xi))]))
    return pred


# function to calculate the r2 value
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


def print_dec_places(l):
    str = ""
    for x in l:
        str += "%.3f" % x + " "
    return str


for i in range(5):
    if i is not 4:
        continue
    print("Dataset", i + 1)
    w = [1 + i, 2 + i, 3 + i, 4 + i]
    std_dev = 0.5 * i
    data, outputs = generate_random_data(w, num_points=100, std_dev=std_dev)
    print("Original weights ->", w, "    Standard Deviation of noise ->", std_dev)
    learned_weights_closed_form = learn_weights_closed_form(data, outputs)
    print("Closed Form (Maximum Likelyhood)     Learned weights -> [" + print_dec_places(learned_weights_closed_form) +
          "]      R2 ->", print_dec_places([calc_r2(outputs, predict(data, learned_weights_closed_form))]))
    # plt.plot(np.array(data)[:, 1], outputs)
    learned_weights_sse = learn_weights(data, outputs, objective_function="sse", learning_rate=0.00001,
                                        iterations=100000)
    print("Sum of Squared Errors                Learned weights -> [" + print_dec_places(
        learned_weights_sse) + "]      R2 ->", print_dec_places([calc_r2(outputs, predict(data, learned_weights_sse))]))

    learned_weights_ssd = learn_weights(data, outputs, objective_function="ssd", learning_rate=0.00001,
                                        iterations=100000)
    print("Sum of Squared Euclidean Distances   Learned weights -> [" + print_dec_places(
        learned_weights_ssd) + "]      R2 ->", print_dec_places([calc_r2(outputs, predict(data, learned_weights_ssd))]))
    print()

# data, outputs = generate_random_data(learned_weights, num_points=10)
# plt.plot(np.array(data)[:, 1], outputs)
# plt.show()

# Original weights [1, 2, 3, 4]
# Closed Form (Maximum Likelyhood)-
# Learned weights [1.98959659 2.03284349 2.98990884 3.98177351]
# R2 0.999622743347507
# Sum of Squared Errors-
# Learned weights [1.7694139777872926, 2.049135767835966, 2.99916767019839, 3.9969805557459295]
# R2 0.9996033970619937
# Sum of Squared Distances-
# Learned weights [2.447238430748347, 1.9935916023769753, 2.9810605459785293, 3.942414035492492]
# R2 0.9995331695885914
