import numpy as np


def logistic_predict_(x, theta):
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
        x = x @ theta
        return 1 / (1 + np.exp(-1 * x))
    except:
        return


def vec_log_loss_(y, y_hat, eps=1e-15):
    """
    Compute the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        y_hat += eps
        m = y.shape[0]
        y = y.flatten()
        y_hat = y_hat.flatten()
        return (-1/m) * (y @ np.log(y_hat) + (1 - y) @ np.log(1 - y_hat))
    except:
        return


# tests:

# Example 1:
y1 = np.array([1]).reshape((-1, 1))
x1 = np.array([4]).reshape((-1, 1))
theta1 = np.array([[2], [0.5]])
y_hat1 = logistic_predict_(x1, theta1)
print(vec_log_loss_(y1, y_hat1))
# Output:
0.018149927917808714
# Example 2:
y2 = np.array([[1], [0], [1], [0], [1]])
x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
theta2 = np.array([[2], [0.5]])
y_hat2 = logistic_predict_(x2, theta2)
print(vec_log_loss_(y2, y_hat2))
# Output:
2.4825011602472347
# Example 3:
y3 = np.array([[0], [1], [1]])
x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
y_hat3 = logistic_predict_(x3, theta3)
print(vec_log_loss_(y3, y_hat3))
# Output:
2.993853310859968
