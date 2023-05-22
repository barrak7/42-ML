import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class MyLinearRegression:
    def __init__(self, theta, alpha=0.00001, max_iter=60000):
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


def main():
    data = pd.read_csv('spacecraft_data.csv').to_numpy()

    print('-------- Part I --------')

    # age:
    myLR_age = MyLinearRegression(theta=np.array(
        [[1], [1]]), alpha=0.001, max_iter=100000)
    age = data[:, 0].reshape(-1, 1)
    y = data[:, -1].reshape(-1, 1)
    myLR_age.fit_(age, y)
    y_hat = myLR_age.predict_(age)
    print('Age Weights:', myLR_age.theta)
    print('Age MSE:', myLR_age.loss_(y, y_hat))
    # age plot

    _, axe = plt.subplots(1, 3, sharey=True, figsize=(15, 8))

    axe[0].plot(age, y, 'o', label='Sell price', color='midnightblue')
    axe[0].plot(age, y_hat, '.', label='predicted price', color='deepskyblue')
    axe[0].set_xlabel('Age')
    axe[0].set_ylabel('Price as a function of age')
    axe[0].set_title('Spacecraft price based on age')
    axe[0].legend()

    # thrust power
    thrust = data[:, 1].reshape(-1, 1)
    myLR_th = MyLinearRegression(theta=np.array(
        [[1], [1]]), alpha=0.0001, max_iter=500000)
    myLR_th.fit_(thrust, y)
    y_hat = myLR_th.predict_(thrust)
    print('Thrust weights:', myLR_th.theta)
    print('Thrust MSE:', myLR_th.loss_(y, y_hat))

    # thrust power plot
    axe[1].plot(thrust, y, 'o', label='Sell Price', color='darkgreen')
    axe[1].plot(thrust, y_hat, '.', label='Predicted price', color='lime')
    axe[1].set_xlabel('Thrust Power')
    axe[1].set_ylabel('Price')
    axe[1].set_title('Spacecraft price based on thrust power')
    axe[1].legend()

    # terameters
    trm = data[:, 2].reshape(-1, 1)

    myLR_trm = MyLinearRegression(
        np.array([[1], [1]]), 0.0001, max_iter=500000)
    myLR_trm.fit_(trm, y)
    y_hat = myLR_trm.predict_(trm)
    print('Terameters weights:', myLR_trm.theta)
    print('Terameters MSE:', myLR_trm.loss_(y, y_hat))

    # terameters plot

    axe[2].plot(trm, y, 'o', label='Sell Price', color='purple')
    axe[2].plot(trm, y_hat, '.', label='Predicted price', color='pink')
    axe[2].set_xlabel('Distance')
    axe[2].set_ylabel('Price')
    axe[2].set_title('Price based on distance')
    axe[2].legend()

    plt.show()

    print('-------- Part II --------')

    myLR = MyLinearRegression(
        np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1), 0.00001, max_iter=50000000)
    myLR.fit_(data[:, :-1], y)
    y_hat = myLR.predict_(data[:, :-1])
    print('Theta:', myLR.theta)
    print('MSE:', myLR.loss_(y, y_hat))

    fig, axe = plt.subplots(1, 3, sharey=True, figsize=(20, 10))
    axe[0].plot(age, y, 'o', label='Price', color='midnightblue')
    axe[0].plot(age, y_hat, '.', label='Predicted Price', color='dodgerblue')
    axe[0].set_xlabel('age')
    axe[0].set_ylabel('price')
    axe[0].set_title('Spacecraft price as a function of Age')
    axe[0].legend()

    axe[1].plot(thrust, y, 'o', label='Price', color='green')
    axe[1].plot(thrust, y_hat, '.', label='Predicted Price', color='lime')
    axe[1].set_xlabel('thrust power')
    axe[1].set_ylabel('price')
    axe[1].set_title('Spacecraft price as a function of Thrust Power')
    axe[1].legend()

    axe[2].plot(trm, y, 'o', label='Price', color='purple')
    axe[2].plot(trm, y_hat, '.', label='Predicted Price', color='violet')
    axe[2].set_xlabel('Distance')
    axe[2].set_ylabel('price')
    axe[2].set_title('Spacecraft price as a function of Distance')
    axe[2].legend()

    plt.show()


if __name__ == '__main__':
    main()
