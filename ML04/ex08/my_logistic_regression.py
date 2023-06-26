import numpy as np


class MyLogisticRegression():
    """
        Description:
        My personnal logistic regression to classify things.
    """
    supported_penalities = [
        'l2']  # We consider l2 penality only. One may wants to implement other penalities

    def __init__(self, theta, alpha=0.001, max_iter=1000, penalty='l2', lambda_=1.0):
        # Check on type, data type, value ... if necessary
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        self.penalty = penalty
        self.lambda_ = lambda_ if penalty in self.supported_penalities else 0

    def loss_elems_(y, y_hat, theta, lambda_):
        """Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, without any for lArgs:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
            lambda_: has to be a float.
        Returns:
            The regularized loss as a float.
            None if y, y_hat, or theta is empty numpy.ndarray.
            None if y and y_hat do not share the same shapes.
        Raises:
            This function should not raise any Exception.
        """
        try:
            m = y.shape[0]
            return y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat) + (lambda_ / (2 * m)) * np.square(theta[1:, :])
        except:
            return

    def loss_(y, y_hat, theta, lambda_):
        """Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, without any for lArgs:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
            lambda_: has to be a float.
        Returns:
            The regularized loss as a float.
            None if y, y_hat, or theta is empty numpy.ndarray.
            None if y and y_hat do not share the same shapes.
        Raises:
            This function should not raise any Exception.
        """
        try:
            m = y.shape[0]
            return ((-1 / m) * (y.T @ np.log(y_hat) + (1 - y).T @ np.log(1 - y_hat)) + (lambda_ / (2 * m)) * (theta[1:, :].T @ theta[1:, :]))[0, 0]
        except:
            return

    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
        Args:
            x: has to be an numpy.ndarray, a vector of dimension m * n.
            theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
        Returns:
            y_hat as a numpy.ndarray, a vector of dimension m * 1.
            None if x or theta are empty numpy.ndarray.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exception."""
        try:
            x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
            x = x @ self.theta
            return 1 / (1 + np.exp(-1 * x))
        except:
            return

    def gradient_(self, y, x):
        """Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any for-loop. The three arrArgs:
            y: has to be a numpy.ndarray, a vector of shape m * 1.
            x: has to be a numpy.ndarray, a matrix of shape m * n.
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
            lambda_: has to be a float.
        Returns:
            A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
            None if y, x, or theta are empty numpy.ndarray.
            None if y, x or theta does not share compatibles shapes.
        Raises:
            This function should not raise any Exception.
        """
        try:
            y_hat = self.logistic_predict_(x)
            tc = self.theta.copy()
            m = x.shape[0]
            tc[0, :] = 0
            x = np.concatenate((np.ones((x.shape[0], 1)), x), 1)
            return (x.T @ (y_hat - y) + self.lambda_ * np.square(tc)) / m
        except:
            return

    def fit_(self, x, y):
        try:
            for _ in range(self.max_iter):
                self.theta -= self.alpha * self.gradient(x, y)
        except:
            return


# tests::


theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
# Example 1:
model1 = MyLogisticRegression(theta, lambda_=5.0)
print(model1.penalty)
# Output
'l2'
print(model1.lambda_)
# Output
5.0
# Example 2:
model2 = MyLogisticRegression(theta, penalty=None)
print(model2.penalty)
# Output
None
print(model2.lambda_)
# Output
0.0
# Example 3:
model3 = MyLogisticRegression(theta, penalty=None, lambda_=2.0)
print(model3.penalty)
# Output
None
print(model3.lambda_)
# Output
