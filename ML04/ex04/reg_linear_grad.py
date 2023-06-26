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
    n = np.ones((x.shape[0], 1))
    x = np.append(n, x, axis=1)
    x = np.matmul(x, theta)
    return x


def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
    with two for-loop. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    y_hat = predict_(x, theta)
    m = y.shape[0]
    tc = theta.copy()
    tc[0, :] = np.sum((y_hat - y)) / m
    tc[1:, :] = (np.sum((y_hat - y) * x, axis=0).reshape(-1, 1) +
                 theta[1:, :] * lambda_) / m
    return tc


def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        y_hat = predict_(x, theta)
        x = np.concatenate((np.ones((x.shape[0], 1)), x), 1)
        theta[0, :] = 0
        m = y.shape[0]
        return (x.T @ (y_hat - y) + lambda_ * theta) / m
    except:
        return


x = np.array([
    [-6, -7, -9],
    [13, -2, 14],
    [-7, 14, -1],
    [-8, -4, 6],
    [-5, -9, 6],
    [1, -5, 11],
    [9, -11, 8]])
y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
theta = np.array([[7.01], [3], [10.5], [-6]])
# Example 1.1:
print(reg_linear_grad(y, x, theta, 1))
# Output:
# array([[-60.99],
#        [-195.64714286],
#        [863.46571429],
#        [-644.52142857]])
# Example 1.2:
print(vec_reg_linear_grad(y, x, theta, 1))
# Output:
# array([[-60.99],
#        [-195.64714286],
#        [863.46571429],
#        [-644.52142857]])
# Example 2.1:
print(reg_linear_grad(y, x, theta, 0.5))
# Output:
# array([[-60.99],
#        [-195.86142857],
#        [862.71571429],
#        [-644.09285714]])
# Example 2.2:
print(vec_reg_linear_grad(y, x, theta, 0.5))
# Output:
# array([[-60.99],
#        [-195.86142857],
#        [862.71571429],
#        [-644.09285714]])
# Example 3.1:
print(reg_linear_grad(y, x, theta, 0.0))
# Output:
# array([[-60.99],
#        [-196.07571429],
#        [861.96571429],
#        [-643.66428571]])
# Example 3.2:
print(vec_reg_linear_grad(y, x, theta, 0.0))
# Output:
# array([[ -60.99 ],
# [-196.07571429],
# [ 861.96571429],
# [-643.66428571]])
