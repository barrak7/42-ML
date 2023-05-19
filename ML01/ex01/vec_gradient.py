import numpy as np


def predict_(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
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
    if x.ndim == 1:
        x = x.reshape((x.shape[0], 1))
    n = np.ones(x.shape)
    x = np.append(n, x, axis=1)
    x = np.matmul(x, theta)
    return x


def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for loop.
    The three arrays must have compatible shapes.
    Args:
            x: has to be a numpy.array, a matrix of shape m * 1.
            y: has to be a numpy.array, a vector of shape m * 1.
            theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
            The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
            None if x, y, or theta is an empty numpy.ndarray.
            None if x, y and theta do not have compatible dimensions.
    Raises:
            This function should not raise any Exception.
    """
    try:
        y_hat = predict_(x, theta)
        m = len(y)
        y_hat = np.tile(y_hat, 2)
        y = np.tile(y, 2)
        x = np.concatenate((np.ones(x.shape), x), 1)
        df = np.sum((y_hat - y) * x, 0) / m
        return df
    except:
        return


x = np.array([12.4956442, 21.5007972, 31.5527382,
             48.9145838, 57.5088733]).reshape((-1, 1))
y = np.array([37.4013816, 36.1473236, 45.7655287,
             46.6793434, 59.5585554]).reshape((-1, 1))
# Example 0:
theta1 = np.array([2, 0.7]).reshape((-1, 1))
print(simple_gradient(x, y, theta1))
# Output:
# array([[-19.0342...], [-586.6687...]])
# Example 1:
theta2 = np.array([1, -0.4]).reshape((-1, 1))
print(simple_gradient(x, y, theta2))
# Output:
# array([[-57.8682...], [-2230.1229...]])
