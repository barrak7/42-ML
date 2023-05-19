import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mse_(y, y_hat):
    """
    Description:
        Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    try:
        return np.sum(np.square(y - y_hat)) / len(y)
    except:
        return


def rmse_(y, y_hat):
    """
    Description:
        Calculate the RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    try:
        return sqrt(np.sum(np.square(y - y_hat)) / len(y))
    except:
        return


def mae_(y, y_hat):
    """
    Description:
        Calculate the MAE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mae: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    try:
        return np.sum(np.abs(y - y_hat)) / len(y)
    except:
        return


def r2score_(y, y_hat):
    """
    Description:
        Calculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    try:
        return 1 - (np.sum(np.square(y - y_hat)) / np.sum(np.square(y - y.mean())))
    except:
        return


# tests


# Example 1:
x = np.array([0, 15, -9, 7, 12, 3, -21])
y = np.array([2, 14, -13, 5, 12, 4, -19])
# Mean squared error
# your implementation
print(mse_(x, y))
# Output:
# 4.285714285714286
# sklearn implementation
print(mean_squared_error(x, y))
# Output:
# 4.285714285714286
# Root mean squared error
# your implementation
print(rmse_(x, y))
# Output:
# 2.0701966780270626
# sklearn implementation not available: take the square root of MSE
print(sqrt(mean_squared_error(x, y)))
# Output:
# 2.0701966780270626
# Mean absolute error
# your implementation
print(mae_(x, y))
# Output:
# 1.7142857142857142
# sklearn implementation
print(mean_absolute_error(x, y))
# Output:
# 1.7142857142857142
# R2-score
# your implementation
print(r2score_(x, y))
# Output:
# 0.9681721733858745
# sklearn implementation
print(r2_score(x, y))
# Output:
# 0.9681721733858745
