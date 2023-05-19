import numpy as np
from matplotlib import pyplot as plt


def plot(x: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
            x: has to be an numpy.array, a vector of dimension m * 1.
            y: has to be an numpy.array, a vector of dimension m * 1.
            theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
            Nothing.
    Raises:
            This function should not raise any Exceptions.
    """
    X = x.reshape((x.shape[0], 1))
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    r = np.matmul(X, theta)
    plt.plot(x, y, 'o', x, r)
    plt.show()


## TESTs ##
x = np.arange(1, 6)
y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
# Example 1:
theta1 = np.array([[4.5], [-0.2]])
plot(x, y, theta1)

# Example 2:
theta2 = np.array([[-1.5], [2]])
plot(x, y, theta2)

# Example 3:
theta3 = np.array([[3], [0.3]])
plot(x, y, theta3)
