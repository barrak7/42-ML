import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class MyLinearRegression():
    """
    Description:
            My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray):
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

    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
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
        loss = self.loss_elem_(y, y_hat)
        try:
            return np.sum(loss, dtype=float) / (2*len(y))
        except:
            return

    def predict_(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
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

    def _simple_gradient(self, x, y) -> np.ndarray:
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
        y_hat = self.predict_(x, self.thetas)
        m = len(y)
        y_hat = np.concatenate((y_hat, y_hat), 1)
        y = np.concatenate((y, y), 1)
        x = np.concatenate((np.ones(x.shape), x), 1)
        df = np.sum((y_hat - y) * x, 0) / m
        return df.reshape((-1, 1))

    def fit_(self, x, y):
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
        while (self.max_iter):
            self.thetas = self.thetas - \
                (self._simple_gradient(x, y) * self.alpha)

            self.max_iter -= 1
        return self.thetas


def linear_model():
    df = pd.read_csv('are_blue_pills_magics.csv')
    x = np.array(df['Micrograms']).reshape(-1, 1)
    y = np.array(df['Score']).reshape(-1, 1)

    ml = MyLinearRegression(np.array([[1], [1]]), max_iter=50000)
    theta = ml.fit_(x, y)
    y_hat = ml.predict_(x, theta)

    plt.plot(x, y, 'o', x, y_hat)
    plt.show()

    th = np.linspace(80, 96, 6)
    th1 = np.linspace(-14, -4, 100).reshape(-1, 1)

    rgb = ['r', 'g', 'b', 'y', 'k', 'c']
    for i, e in enumerate(th):
        p = []
        for v in th1:
            p.append(ml.loss_(y, ml.predict_(x, np.array([e, v[0]]))))
        p = np.array(p)
        plt.plot(th1, p, color=rgb[i], label=f'h{i}')

    plt.xlabel('MSE loss')
    plt.ylabel('Theta0')
    plt.title('MSE')
    # Add a legend
    plt.legend()

    plt.show()


def main():
    linear_model()


if __name__ == '__main__':
    main()
