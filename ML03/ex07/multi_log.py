import numpy as np
import sys
import re
import pandas as pd
from my_logistic_regression import MyLogisticRegression as mlr
import matplotlib.pyplot as plt


def fit_n(data, y, n):
    y_ = y.copy()
    y_[y == n] = 1
    y_[y != n] = 0
    theta = np.ones((4, 1))
    MLR = mlr(theta, 0.0001, 5000000)
    MLR.fit_(data, y_)
    return (MLR)


def main():
    data = pd.read_csv('solar_system_census.csv').to_numpy()[:, 1:]
    y = pd.read_csv('solar_system_census_planets.csv').to_numpy()[:, 1:]
    data_t, data_s = data[:int(data.shape[0] *
                          0.7), :], data[int(data.shape[0] * 0.7):, :]
    y_t, y_s = y[:int(y.shape[0] *
                      0.7), :], y[int(y.shape[0] * 0.7):, :]

    planets: list[mlr] = []
    for i in range(4):
        planets.append(fit_n(data_t, y_t, i))
    planets = np.concatenate((np.array(planets[0].predict_(data_s)), np.array(planets[1].predict_(
        data_s)), np.array(planets[2].predict_(data_s)), np.array(planets[3].predict_(data_s))), 1)

    mask = np.max(planets, axis=1)
    mask = planets == mask[:, np.newaxis]
    planets[mask] = 1
    planets[~mask] = 0
    print(planets)


if __name__ == '__main__':
    main()
