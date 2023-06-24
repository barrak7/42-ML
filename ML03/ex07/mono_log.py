import numpy as np
import sys
import re
import pandas as pd
from my_logistic_regression import MyLogisticRegression as mlr
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 2:
        print("usage: mono_log -zipcode=x")
        return
    rg = re.compile(r"-zipcode=[0-3]$")
    if not rg.match(sys.argv[1]):
        print("usage: mono_log -zipcode=x (0 >= x <= 3)")
        return
    zc = int(sys.argv[1][-1])
    data = pd.read_csv('solar_system_census.csv').to_numpy()[:, 1:]
    y = pd.read_csv('solar_system_census_planets.csv').to_numpy()[:, 1:]
    y[y != zc] = 0
    y[y == zc] = 1
    data_t, data_s = data[:int(data.shape[0] *
                          0.7), :], data[int(data.shape[0] * 0.7):, :]
    y_t, y_s = y[:int(y.shape[0] *
                      0.7), :], y[int(y.shape[0] * 0.7):, :]
    theta = np.ones((4, 1))
    MLR = mlr(theta, alpha=0.0001, max_iter=5000000)
    MLR.fit_(data_t, y_t)
    y_hat = MLR.predict_(data_s)
    y_hat[y_hat <= 0.5] = 0
    y_hat[y_hat > 0.5] = 1
    print(
        f"success over total {np.sum(y_s[y_s == y_hat].shape[0])/y_s.shape[0]}, total {y_s.shape[0]}")
    fig, axe = plt.subplots(1, 3)
    axe[0].scatter(data_s[:, 0], y_s, label='planets')
    axe[0].scatter(data_s[:, 0], y_hat, label='predicted')
    axe[1].scatter(data_s[:, 1], y_s, label='planets')
    axe[1].scatter(data_s[:, 1], y_hat, label='predicted')
    axe[2].scatter(data_s[:, 2], y_s, label='planets')
    axe[2].scatter(data_s[:, 2], y_hat, label='predicted')

    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
