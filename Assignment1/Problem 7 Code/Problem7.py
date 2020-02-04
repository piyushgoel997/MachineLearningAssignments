import math
import random

import numpy as np


# Calculates the first order derivatives of the gumbel distribution for the given data points using the given params.
def first_derivative(data, theta, probs):
    [alpha, beta] = theta
    ans = np.zeros((2,))
    ans[0] = sum([probs[i] * ((1 / beta) * (1 - math.exp(-(data[i] - alpha) / beta))) for i in range(len(data))])
    ans[1] = sum(
        [probs[i] * ((-1 / beta) + ((data[i] - alpha) / beta ** 2) * (1 - math.exp(-(data[i] - alpha) / beta))) for i in
         range(len(data))])
    return ans


# Calculates the second order derivatives of the gumbel distribution for the given data points using the given params.
def second_derivative(data, theta, probs):
    [alpha, beta] = theta
    k = sum(
        [probs[i] * (-(1 / beta ** 2) * (1 - math.exp(-(data[i] - alpha) / beta)) - ((data[i] - alpha) / beta ** 3) * (
            math.exp(-(data[i] - alpha) / beta))) for i in range(len(data))])
    ans = np.zeros((2, 2))
    ans[0][0] = sum([probs[i] * (-(1 / beta ** 2) * (math.exp(-(data[i] - alpha) / beta))) for i in range(len(data))])
    ans[0][1] = k
    ans[1][0] = k
    ans[1][1] = sum(
        [probs[i] * ((1 / beta ** 2) - 2 * ((data[i] - alpha) / beta ** 3) + 2 * ((data[i] - alpha) / beta ** 3) * (
            math.exp(-(data[i] - alpha) / beta)) - (((data[i] - alpha) / beta ** 2) ** 2) * math.exp(
            -(data[i] - alpha) / beta)) for i in range(len(data))])
    return ans


# Uses newton raphson to find the optimized values of alpha and beta.
def find_param(data, probs, theta_old):
    theta = np.array(theta_old)
    step = 0
    while True:
        grad = first_derivative(data, theta, probs)
        sec = second_derivative(data, theta, probs)
        if sum(sum(sec)) == 0:
            break
        inv = np.linalg.inv(sec)
        theta_new = theta - np.matmul(inv, grad)
        if abs(theta_new[0] - theta[0]) < 0.000001 or abs(theta_new[1] - theta[1]) < 0.000001:
            break
        if step > 100:
            break
        theta = theta_new
    return theta


# find the output if a set of points is fed into the gumbel distribution with the given set of params.
def gumbel(data, param):
    [alpha, beta] = param
    return [(1 / beta) * math.exp(-(x - alpha) / beta) * math.exp(-math.exp(-(x - alpha) / beta)) for x in data]


# find the output if a set of points is fed into the normal distribution with the given set of params.
def normal(data, param):
    [mu, sig] = param
    return [(1 / math.sqrt(2 * math.pi * math.pow(sig, 2))) * math.exp(-((x - mu) ** 2) / (2 * math.pow(sig, 2))) for x
            in data]


# expectation maximization
def em(data):
    # alpha, beta, mu, sigma, w1, w2
    theta = np.array([4, 6, 2, 3, 0.5, 0.5])  # initialize the values over here
    py = np.zeros((2, len(data)))
    # num_steps = 10000
    t = 0
    while True:
        # step a
        p_1 = gumbel(data, theta[:2])
        p_2 = normal(data, theta[2:4])
        for i in range(len(data)):
            den = theta[4] * p_1[i] + theta[5] * p_2[i]
            py[0][i] = (theta[4] * p_1[i]) / den
            py[1][i] = (theta[5] * p_2[i]) / den

        # step b
        w_1_next = np.mean(py[0])
        w_2_next = np.mean(py[1])

        # step c
        alpha_next, beta_next = find_param(data, py[0], theta[:2])

        mu_next = 0
        sig_next = 0
        p_sum = 0
        for i in range(len(data)):
            mu_next += py[1][i] * data[i]
            sig_next += py[1][i] * ((data[i] - theta[2]) ** 2)
            p_sum += py[1][i]
        mu_next = mu_next / p_sum
        sig_next = math.sqrt(sig_next / p_sum)

        t += 1

        theta_next = [alpha_next, beta_next, mu_next, sig_next, w_1_next, w_2_next]
        # print("step", t, theta_next)
        # if t > num_steps:
        #     break

        diff = sum([abs(theta[i] - theta_next[i]) for i in range(len(theta))])
        if diff < 0.0000001:
            break

        theta = theta_next
    # print(theta)
    return theta


# generate random data
def random_data(theta, n):
    [alpha, beta, mu, sig, w_1, w_2] = theta
    if w_1 + w_2 != 1:
        return
    data = []
    for x in np.random.gumbel(alpha, beta, int(w_1 * n)):
        data.append(x)
    for x in np.random.normal(mu, sig, int(w_2 * n)):
        data.append(x)
    random.shuffle(data)
    return np.array(data)


# alternative random data generator
def random_data2(theta, n):
    [alpha, beta, mu, sig, w_1, w_2] = theta
    if w_1 + w_2 != 1:
        return

    # https://stackoverflow.com/questions/47759577/creating-a-mixture-of-probability-distributions-for-sampling/47763145
    distributions = [
        {"type": np.random.gumbel, "kwargs": {"loc": alpha, "scale": beta}},
        {"type": np.random.normal, "kwargs": {"loc": mu, "scale": sig}},
    ]
    coefficients = np.array([w_1, w_2])
    num_distr = len(distributions)
    data = np.zeros((n, num_distr))
    for idx, distr in enumerate(distributions):
        data[:, idx] = distr["type"](size=(n,), **distr["kwargs"])
    random_idx = np.random.choice(np.arange(num_distr), size=(n,), p=coefficients)
    sample = data[np.arange(n), random_idx]
    return sample


# calculate the mean of the given list of numbers
def calc_mean(nos):
    return sum(nos) / len(nos)


# calculate the standard deviation of the given list of numbers
def calc_std_dev(nos):
    mean = calc_mean(nos)
    return math.sqrt(sum([(x - mean) ** 2 for x in nos]) / len(nos))


# reports the mean and standard deviation by running the em algorithm for n random datapoints, iter number of times.
def report_mean_std_dev(theta, n, iter):
    thetas = []
    for i in range(iter):
        data = random_data2(theta.copy(), n)
        thetas.append(em(data))

    result = []
    for i in range(len(thetas[0])):
        nos = []
        for t in thetas:
            nos.append(t[i])
        result.append(calc_mean(nos))
        result.append(calc_std_dev(nos))

    return result


np.random.seed(1)
print(report_mean_std_dev([3, 5, 1, 2, 0.45, 0.55], 100, 10))
print(report_mean_std_dev([3, 5, 1, 2, 0.45, 0.55], 1000, 10))
print(report_mean_std_dev([3, 5, 1, 2, 0.45, 0.55], 10000, 10))
# d = random_data([3, 5, 1, 2, 0.45, 0.55], 10000)
# em(d)


# [3.653984973165401, 2.9760953521077025, 4.313052468594124, 0.6874999748946583, 1.0356624439597317, 0.5401259356986734, 1.633016497018561, 0.5835387592453563, 0.5066483912227837, 0.23589574492532026, 0.4933516087772163, 0.23589574492532026]
# [3.2583357475032657, 0.825352933724691, 4.820691260808451, 0.17753672855244898, 0.937698056408698, 0.0922954298550373, 1.9397270787995804, 0.1026674759560784, 0.4530064737679769, 0.07604119414517353, 0.5469935262320231, 0.07604119414517353]
# [3.1703620855291623, 0.199418224581099, 5.087169362186025, 0.10505213169002149, 0.9988390492745067, 0.04058879509011444, 2.026044284182844, 0.0545868743625404, 0.43161192290021716, 0.01815939822942271, 0.5683880770997829, 0.018159398229422728]
