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


def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, with two for-loops. The three arrayArgs:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    Raises:
    This function should not raise any Exception.
    """
    try:
        y_hat = logistic_predict_(x, theta)
        tc = theta.copy()
        m = y.shape[0]
        tc[0, :] = np.sum(y_hat - y) / m
        tc[1:, :] = (np.sum((y_hat - y) * x, axis=0).reshape(-1,
                     1) + lambda_ * theta[1:, :])/m
        return tc
    except:
        return


def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any for-loop. The three arrArgs:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of shape m * n.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    Raises:
    This function should not raise any Exception.
    """
    try:
        y_hat = logistic_predict_(x, theta)
        tc = theta.copy()
        m = x.shape[0]
        tc[0, :] = 0
        x = np.concatenate((np.ones((x.shape[0], 1)), x), 1)
        return (x.T @ (y_hat - y) + lambda_ * tc) / m
    except:
        return


x = np.array([[0, 2, 3, 4],
              [2, 4, 5, 5],
              [1, 3, 2, 7]])
y = np.array([[0], [1], [1]])
theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
# Example 1.1:
print(reg_logistic_grad(y, x, theta, 1))
# Output:
# array([[-0.55711039],
# [-1.40334809],
# [-1.91756886],
# [-2.56737958],
# [-3.03924017]])
# Example 1.2:
print(vec_reg_logistic_grad(y, x, theta, 1))
# Output:
# array([[-0.55711039],
# [-1.40334809],
# [-1.91756886],
# [-2.56737958],
# [-3.03924017]])
# Example 2.1:
print(reg_logistic_grad(y, x, theta, 0.5))
# Output:
# array([[-0.55711039],
# [-1.15334809],
# [-1.96756886],
# [-2.33404624],
# [-3.15590684]])
# Example 2.2:
print(vec_reg_logistic_grad(y, x, theta, 0.5))
# Output:
# array([[-0.55711039],
# [-1.15334809],
# [-1.96756886],
# [-2.33404624],
# [-3.15590684]])
# Example 3.1:
print(reg_logistic_grad(y, x, theta, 0.0))
# Output:
# array([[-0.55711039],
# [-0.90334809],
# [-2.01756886],
# [-2.10071291],
# [-3.27257351]])
# Example 3.2:
print(vec_reg_logistic_grad(y, x, theta, 0.0))
# Output:
# array([[-0.55711039],
# [-0.90334809],
# [-2.01756886],
# [-2.10071291],
# [-3.27257351]])
