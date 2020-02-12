import random


def calculate_derivatives(prev_weights, data, correct_outputs):
    derivatives = []
    for k in range(len(prev_weights)):
        derivative = 0
        for i in range(len(data)):
            x_i = data[i]
            y_i = correct_outputs[i]
            b = y_i ** 2  # sum of squares of w_j plus square of y_i
            a = -y_i  # prediction - y_i
            for j in range(len(x_i)):
                b += prev_weights[j] ** 2
                a += prev_weights[j] * x_i[j]
            derivative += (a * (x_i[k] * b - prev_weights[k] * a)) / (b ** 2)
        derivative *= 2
        derivatives.append(derivative)
    return derivatives


def grad_descent_iter(prev_weights, data, correct_outputs, learning_rate):
    delta = calculate_derivatives(prev_weights, data, correct_outputs)
    new_weights = []
    for i in range(len(prev_weights)):
        new_weights.append(prev_weights[i] - learning_rate * delta[i])
    return new_weights


def initialize_weights(k):
    return [random.random() for _ in range(k)]


def learn_weights(data, correct_outputs, learning_rate=0.00001, iterations=1000):
    weights = initialize_weights(len(data[0]))
    for i in range(iterations):
        new_weights = grad_descent_iter(weights, data, correct_outputs, learning_rate)
        diff = abs(sum([new_weights[k] - weights[k] for k in range(len(weights))]))
        weights = new_weights
        if diff <= 0.0000001:
            break
        # print(weights)
    return weights


# don't forget to add ones in the first column of data.
def generate_random_data(original_weights, std_dev=0.0, num_points=100):
    data = []
    for i in range(num_points):
        x_i = []
        x_i.append(1)
        for i in range(len(original_weights) - 1):
            x_i.append(10 * random.random())
        data.append(x_i)

    correct_outputs = []
    for x_i in data:
        correct_outputs.append(sum([x_i[j] * original_weights[j] + std_dev * random.random() for j in range(len(x_i))]))

    return data, correct_outputs


w = [1, 2, 3, 4, 5]
data, outputs = generate_random_data(w, num_points=1000, std_dev=0.5)
# plt.plot(np.array(data)[:, 1], outputs)
learned_weights = learn_weights(data, outputs, learning_rate=0.00001, iterations=100000)
# data, outputs = generate_random_data(learned_weights, num_points=10)
# plt.plot(np.array(data)[:, 1], outputs)
print(w)
print(learned_weights)
# plt.show()

# [1.4084638666772067, 2.088353088780477, 3.0638003820103963, 4.024282094647186, 4.99564101355804]
