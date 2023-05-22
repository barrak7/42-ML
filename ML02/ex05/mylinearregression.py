import numpy as np


class MyLinearRegression:
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.theta = theta
        self.alpha = alpha
        self.max_iter = max_iter

    def predict_(self, x):
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
        try:
            m = x.shape[0]
            x = np.concatenate((np.ones((m, 1)), x), 1)

            return x @ self.theta
        except:
            return

    def loss_elem_(slef, y, y_hat):
        """
        Description:
            Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_elem: numpy.array, a vector of dimension (number of the training examples,1).
            None if there is a dimension matching problem between X, Y or theta.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            return np.square(y - y_hat)
        except:
            return

    def loss_(self, y: np.ndarray, y_hat):
        """
        Description:
            Calculates the value of loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_value : has to be a float.
            None if there is a dimension matching problem between X, Y or theta.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            m = y.size
            loss = self.loss_elem_(y, y_hat)
            return np.sum(loss) / (2 * m)
        except:
            return

    def gradient(self, x: np.ndarray, y: np.ndarray):
        """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
        The three arrays must have the compatible dimensions.
        Args:
            x: has to be an numpy.array, a matrix of dimension m * n.
            y: has to be an numpy.array, a vector of dimension m * 1.
            theta: has to be an numpy.array, a vector (n +1) * 1.
        Return:
            The gradient as a numpy.array, a vector of dimensions n * 1,
            containg the result of the formula for all j.
            None if x, y, or theta are empty numpy.array.
            None if x, y and theta do not have compatible dimensions.
            None if x, y or theta is not of expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            return x.T @ (x @ self.theta - y) / y.size
        except:
            return

    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.array, a matrix of dimension m * n:
            (number of training examples, number of features).
            y: has to be a numpy.array, a vector of dimension m * 1:
            (number of training examples, 1).
            theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
            (number of features + 1, 1).
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during the gradient descent
        Return:
            new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
            None if there is a matching dimension problem.
            None if x, y, theta, alpha or max_iter is not of expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            m = y.size
            max = self.max_iter
            x = np.concatenate((np.ones((m, 1)), x), 1)

            while max:
                gr = self.gradient(x, y)
                old_th = self.theta
                self.theta = self.theta - self.alpha * gr
                if np.all(np.absolute(self.theta - old_th) <= 0.000001):
                    break
                max -= 1
            return self.theta
        except:
            return


# TESTs
X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])

# Example 0:
y_hat = mylr.predict_(X)

print(y_hat)
# Output:
# array([[8.], [48.], [323.]])

# Example 1:
print(mylr.loss_elem_(Y, y_hat))

# Output:
# array([[225.], [0.], [11025.]])

# Example 2:
print(mylr.loss_(Y, y_hat))

# Output:
# 1875.0

# Example 3:
mylr.alpha = 1.6e-4
mylr.max_iter = 200000
mylr.fit_(X, Y)
print(mylr.theta)

# Output:
# array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])

# Example 4:
y_hat = mylr.predict_(X)

print(y_hat)
# Output:
# array([[23.417..], [47.489..], [218.065...]])

# Example 5:
print(mylr.loss_elem_(Y, y_hat))

# Output:
# array([[0.174..], [0.260..], [0.004..]])

# Example 6:
print(mylr.loss_(Y, y_hat))

# Output:
# 0.0732
