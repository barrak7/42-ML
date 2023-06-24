import numpy as np


class MyLogisticRegression():
    """
    Description:
        My personnal logistic regression to classify things.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

    def predict_(self, x):
        try:
            x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
            x = x @ self.theta
            return 1 / (1 + np.exp(-1 * x))
        except:
            return

    def gradient(self, x, y):
        try:
            y_hat = self.predict_(x)
            X = np.concatenate((np.ones((x.shape[0], 1)), x), 1)
            return (X.T @ (y_hat - y)) / x.shape[0]
        except:
            return

    def loss_elem_(self, y, y_hat):
        try:
            eps = 1e-15
            y = y.flatten()
            y_hat = y_hat.flatten()
            return np.dot(y, np.log(y_hat + eps)) + np.dot((1 - y), np.log(1 - y_hat + eps))
        except:
            return

    def loss_(self, x, y):
        y_hat = self.predict_(x)

        m = y.shape[0]
        y = y.flatten()
        y_hat = y_hat.flatten()

        return (-1/m) * self.loss_elem_(y, y_hat)

    def fit_(self, x, y):
        try:
            for _ in range(self.max_iter):
                self.theta -= self.alpha * self.gradient(x, y)
        except:
            return
