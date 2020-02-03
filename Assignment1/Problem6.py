import math
import numpy as np


def first_derivative(data, theta):
    [alpha, beta] = theta
    return [sum([(1 / beta) * (1 - math.exp(-(x - alpha) / beta)) for x in data]),
            sum([(-1 / beta) + ((x - alpha) / beta ** 2) - ((x - alpha) / beta ** 2) * math.exp(-(x - alpha) / beta) for
                 x in data])]


def second_derivative(data, theta):
    [alpha, beta] = theta
    k = sum([-(1 / beta ** 2) * (1 - math.exp(-(x - alpha) / beta)) - ((x - alpha) / beta ** 3) * math.exp(
        -(x - alpha) / beta) for x in data])
    return [[
        sum([-(1 / beta ** 2) * math.exp(-(x - alpha) / beta) for x in data]),
        k
    ], [
        k,
        sum([(1 / beta ** 2) + 2 * ((x - alpha) / beta ** 3) * (-1 + math.exp(-(x - alpha) / beta)) + (
                (x - alpha) ** 2 / beta ** 4) * math.exp(-(x - alpha) / beta) for x in data])
    ]]


def first(data, theta):
    [alpha, beta] = theta
    ans = np.zeros((2,))
    ans[0] = sum([(1 / beta) * (1 - math.exp(-(x - alpha) / beta)) for x in data])
    ans[1] = sum([(-1 / beta) + ((x - alpha) / beta ** 2) * (1 - math.exp(-(x - alpha) / beta)) for x in data])
    return ans


def second(data, theta):
    [alpha, beta] = theta
    k = sum([-(1 / beta ** 2) * (1 - math.exp(-(x - alpha) / beta)) - ((x - alpha) / beta ** 3) * (
        math.exp(-(x - alpha) / beta)) for x in data])
    ans = np.zeros((2, 2))
    ans[0][0] = sum([-(1 / beta ** 2) * (math.exp(-(x - alpha) / beta)) for x in data])
    ans[0][1] = k
    ans[1][0] = k
    ans[1][1] = sum([(1 / beta ** 2) - 2 * ((x - alpha) / beta ** 3) + 2 * ((x - alpha) / beta ** 3) * (
        math.exp(-(x - alpha) / beta)) - (((x - alpha) / beta ** 2) ** 2) * math.exp(-(x - alpha) / beta) for x in
                     data])
    return ans


def find_param(data):
    theta = np.array([5, 6])
    step = 0
    while True:
        grad = first(data, theta)
        inv = np.linalg.inv(second(data, theta))
        theta_new = theta - np.matmul(inv, grad)
        if abs(theta_new[0] - theta[0]) < 0.000001 and abs(theta_new[1] - theta[1]) < 0.000001:
            break
        if step > 100:
            break
        theta = theta_new
    return theta


def calc_mean(nos):
    return sum(nos) / len(nos)


def calc_std_dev(nos):
    mean = calc_mean(nos)
    return math.sqrt(sum([(x - mean) ** 2 for x in nos]) / len(nos))


def report_mean_std_dev(alpha, beta, n, iter):
    alphas = []
    betas = []
    for data in range(iter):
        data = np.random.gumbel(alpha, beta, n)
        theta = find_param(data)
        alphas.append(theta[0])
        betas.append(theta[1])
    return [calc_mean(alphas), calc_std_dev(alphas), calc_mean(betas), calc_std_dev(betas)]


alpha = 3
beta = 5
np.random.seed(1)
print(report_mean_std_dev(alpha, beta, 100, 10))
print(report_mean_std_dev(alpha, beta, 1000, 10))
print(report_mean_std_dev(alpha, beta, 10000, 10))

# [3.2438274360538704, 0.9100805236531226, 4.663582232241932, 1.9785750394903054]
# [3.0202042219788705, 0.15077821975369488, 5.008440719177886, 0.11575451286201524]
# [3.0009860191398006, 0.05182994545147269, 4.99328653633901, 0.04934931694014936]

# [3.0481305426284777, 0.37239480572410094, 4.983757931364503, 0.4023394157281205]
# [3.03592666089177, 0.22179821437548128, 4.950361269162326, 0.12611333498088606]
# [3.006725071615766, 0.03999678145499263, 5.01829747130164, 0.04281649305356705]
