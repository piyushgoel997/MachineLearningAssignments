file = open("temp")
nos = [x for x in file.read().split() if len(x) > 3]

one = nos[:100]
two = nos[100:]

one = [float(i) for i in one]
two = [float(i) for i in two]


import matplotlib.pyplot as plt

plt.figure()
hundred_points, = plt.plot(range(1, 101), one, label='n=100')
thousand_points, = plt.plot(range(1, 101), two, label='n=1000')
plt.legend([hundred_points, thousand_points], ['n = 100', 'n = 1000'])
plt.xlabel('k')
plt.ylabel('r(k)')
plt.savefig('k-vs-rk.jpg')
plt.show()