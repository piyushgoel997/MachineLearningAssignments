import math
import random

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# generates a list of random points
def random_points(n, k):
    points = []
    for i in range(n):
        pt = []
        for j in range(k):
            pt.append(random.random())
        points.append(pt)
    return points


# def dist_between(p, q):
#     dist = 0
#     k = 0
#     for i in range(len(p)):
#         dist += (p[i] - q[i]) ** 2
#         k += 1
#     return dist / k


# finds the min and max distance between the points.
def min_max_dist(points):
    distances = cdist(points, points, metric='euclidean')
    distances = [d for dx in distances for d in dx if d > 0]
    return min(distances), max(distances)
    # min_dist = math.inf
    # max_dist = 0
    # for p in points:
    #     for q in points:
    #         if p is not q:
    #             dist = dist_between(p, q)
    #             min_dist = min(min_dist, dist)
    #             max_dist = max(max_dist, dist)
    # return min_dist, max_dist


# calculates the value of the function r(k) given the min and max distance.
def calc_r(min_dist, max_dist):
    return math.log10((max_dist - min_dist) / min_dist)


# averages out the value of r(k) for 100 different datasets with n points.
def get_averages(n):
    averages = []
    for k in range(1, 101):
        print(k)
        avg = 0
        t = 100
        for i in range(t):
            avg += calc_r(*min_max_dist(random_points(n, k)))
        avg = avg / t
        print(avg)
        averages.append(avg)
    return averages


plt.figure()
hundred_points, = plt.plot(range(1, 101), get_averages(100), label='n=100')
thousand_points, = plt.plot(range(1, 101), get_averages(1000), label='n=1000')
plt.legend([hundred_points, thousand_points], ['n = 100', 'n = 1000'])
plt.xlabel('k')
plt.ylabel('r(k)')
plt.savefig('k-vs-rk.jpg')
plt.show()
