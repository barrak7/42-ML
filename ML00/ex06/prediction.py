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
