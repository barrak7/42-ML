import numpy as np
from dataclasses import dataclass


@dataclass
class MyRidge:
    theta: np.ndarray
    alpha: float = 0.0001
    max_iter: int = 1500
    lambda_: float = 0.5

    def get_params_(self):
        return (self.theta, self.alpha, self.max_iter, self.lambda_)

    def set_params_(self, theta, alpha=0.0001,  max_iter=1500, lambda_=0.5):
        self.theta = theta
        self.alpha = alpha
        self.max_iter = max_iter
        self.lambda_ = lambda_

    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of dimension m * 1.
            theta: has to be an numpy.array, a vector of dimension 2 * 1.
        Returns:
            y_hat as a numpy.array, a vector of dimension m * 1.
            None if x and/or theta are not numpy.array.
            None if x or theta are empty numpy.array.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exceptions.
        """
        try:
            m = x.shape[0]
            x = np.concatenate((np.ones((m, 1)), x), 1)
            return x @ self.theta
        except:
            return

    def elem_loss_(self, y, y_hat):
        """Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop.Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
            lambda_: has to be a float.
        Returns:
            The regularized loss as a float.
            None if y, y_hat, or theta are empty numpy.ndarray.
            None if y and y_hat do not share the same shapes.
        Raises:
            This function should not raise any Exception.
        """
        try:
            return np.square((y_hat - y)) + self.lambda_ * self.theta[1:, :].T @ self.theta[1:, :]
        except:
            return

    def loss_(self, y, y_hat):
        """Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop.Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
            lambda_: has to be a float.
        Returns:
            The regularized loss as a float.
            None if y, y_hat, or theta are empty numpy.ndarray.
            None if y and y_hat do not share the same shapes.
        Raises:
            This function should not raise any Exception.
        """
        try:
            m = y.shape[0]
            return (1/(2*m)) * ((y_hat - y).T @ (y_hat - y) + self.lambda_ * (self.theta[1:, :].T @ self.theta[1:, :]))[0, 0]
        except:
            return

    def gradient_(self, y, x):
        """Computes the regularized linear gradient of three non-empty numpy.ndarray,
        without any for-loop. The three arrays must have compatible shapes.
        Args:
                y: has to be a numpy.ndarray, a vector of shape m * 1.
                x: has to be a numpy.ndarray, a matrix of dimesion m * n.
                theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
                lambda_: has to be a float.
        Return:
                A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
                None if y, x, or theta are empty numpy.ndarray.
                None if y, x or theta does not share compatibles shapes.
                None if y, x or theta or lambda_ is not of the expected type.
        Raises:
                This function should not raise any Exception.
        """
        try:
            y_hat = self.predict_(x)
            x = np.concatenate((np.ones((x.shape[0], 1)), x), 1)
            theta = self.theta
            theta[0, :] = 0
            m = y.shape[0]
            return (x.T @ (y_hat - y) + self.lambda_ * theta) / m
        except:
            return

    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.array, a matrix of dimension m * n:
            (number of training examples, number of features).
            y: has to be a numpy.array, a vector of dimension m * 1:
            (number of training examples, 1).
            theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
            (number of features + 1, 1).
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during the gradient descent
        Return:
            new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
            None if there is a matching dimension problem.
            None if x, y, theta, alpha or max_iter is not of expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            max = self.max_iter
            while max:
                gr = self.gradient_(y, x)
                old_th = self.theta
                self.theta = self.theta - self.alpha * gr
                if np.all(np.absolute(self.theta - old_th) <= 0.000001):
                    break
                max -= 1
            return self.theta
        except:
            return
