import numpy as np
from matplotlib import pyplot as plt

X = np.array([400, 450, 900, 390, 550])

T = np.linspace(0.01, 5, 100)

P = np.zeros((len(T), len(X)))

SUM_T = np.zeros((len(T)))
for t_index, t_value in enumerate(T):
    pw = -float(1 / t_value)
    SUM_T[t_index] = np.sum(X ** pw)

for x_index, x_value in enumerate(X):
    numerator = np.zeros(len(T))
    for t_index, t_value in enumerate(T):
        pw = -float(1 / t_value)
        numerator[t_index] = x_value ** pw

    P[:, x_index] = numerator / SUM_T[:]

for i in range(len(X)):
    plt.plot(T, P[:, i], label=str(X[i]))

plt.xlabel("T")
plt.ylabel("P")
plt.title("Probability as a function of the temperature")
plt.legend()
plt.grid()
plt.show()
exit()
