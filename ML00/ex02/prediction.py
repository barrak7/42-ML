import numpy as np


def simple_predict(x: np.ndarray, theta: np.ndarray):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
        Args:
                x: has to be an numpy.ndarray, a vector of dimension m * 1.
                theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
        Returns:
                y_hat as a numpy.ndarray, a vector of dimension m * 1.
                None if x or theta are empty numpy.ndarray.
                None if x or theta dimensions are not appropriate.
        Raises:
                This function should not raise any Exception.
        """
    if (x.ndim == 2 and x.shape[1] != 1) or (theta.ndim == 2 and theta.shape != (2, 1)) or theta.ndim == 1 and theta.shape != (2,):
        print('ERROR: x and theta have to be vectors of shape (m, 1) and shape (2, 1) respectively.')
        return
    if not x.any() or not theta.any():
        return
    r = x * theta[1] + theta[0]
    print(r)
    return r
