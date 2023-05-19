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
    print(x)
    return x


x = np.arange(1, 6)
# Example 1:
theta1 = np.array([[5], [0]])
predict_(x, theta1)
# Ouput:
# array([[5.], [5.], [5.], [5.], [5.]])
# Do you remember why y_hat contains only 5’s here?
# Example 2:
theta2 = np.array([[0], [1]])
predict_(x, theta2)
# Output:
# array([[1.], [2.], [3.], [4.], [5.]])
# Do you remember why y_hat == x here?
# Example 3:
theta3 = np.array([[5], [3]])
predict_(x, theta3)
# Output:
# array([[ 8.], [11.], [14.], [17.], [20.]])
# Example 4:
theta4 = np.array([[-3], [1]])
predict_(x, theta4)
# Output:
# array([[-2.], [-1.], [ 0.], [ 1.], [ 2.]])
