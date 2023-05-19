from matplotlib import pyplot as plt
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


def plot_with_loss(x: np.ndarray, y: np.ndarray, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
            x: has to be an numpy.ndarray, a vector of dimension m * 1.
            y: has to be an numpy.ndarray, a vector of dimension m * 1.
            theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
            Nothing.
    Raises:
            This function should not raise any Exception.
    """
    y_hat = predict_(x, theta)
    plt.plot(x, y, 'o', x, y_hat)
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i], y_hat[i]], 'r--')
    plt.show()


## TESTs ##
x = np.arange(1, 6)
y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
# Example 1:
theta1 = np.array([18, -1])
plot_with_loss(x, y, theta1)

# Example 2:
theta2 = np.array([14, 0])
plot_with_loss(x, y, theta2)

# Example 3:
theta3 = np.array([12, 0.8])
plot_with_loss(x, y, theta3)
