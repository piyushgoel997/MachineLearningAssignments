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


def em(data):
    theta = np.array([])  # initialize the values over here
    py = np.zeros()