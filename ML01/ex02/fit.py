import numpy as np


def simple_gradient(x, y, theta) -> np.ndarray:
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
    y_hat = predict_(x, theta)
    m = len(y)
    y_hat = np.concatenate((y_hat, y_hat), 1)
    y = np.concatenate((y, y), 1)
    x = np.concatenate((np.ones(x.shape), x), 1)
    df = np.sum((y_hat - y) * x, 0) / m
    return df.reshape((-1, 1))


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


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    while (max_iter):
        theta = theta - (simple_gradient(x, y, theta) * alpha)

        max_iter -= 1
    return theta


x = np.array([[12.4956442], [21.5007972], [
             31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [
             45.7655287], [46.6793434], [59.5585554]])
theta = np.array([1, 1]).reshape((-1, 1))
# Example 0:
theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
# Output:
# array([[1.40709365],
# [1.1150909 ]])
# Example 1:
print(predict_(x, theta1))
# Output:
# array([[15.3408728 ],
# [25.38243697],
# [36.59126492],
# [55.95130097],
# [65.53471499]])
