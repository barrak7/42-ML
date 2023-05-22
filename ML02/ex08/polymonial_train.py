import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MyLinearRegression:
    def __init__(self, theta, alpha=0.0001, max_iter=100000):
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
    data = pd.read_csv('are_blue_pills_magics.csv').to_numpy()[:, 1:]
    X = data[:, 0]
    y = data[:, 1].reshape(-1, 1)
    x_lst = [np.vander(x=X, N=e, increasing=True)[:, 1:] for e in range(2, 8)]

    # level one:
    print('--- Different level ponylomial models ---\n')

    myLR: list[MyLinearRegression] = []
    myLR.append(MyLinearRegression(np.array([[1], [1]]), max_iter=300000))
    myLR.append(MyLinearRegression(np.array([[1], [1], [1]]), max_iter=400000))
    myLR.append(MyLinearRegression(
        np.array([[1], [1], [1], [1]]), alpha=0.00001, max_iter=500000))
    myLR.append(MyLinearRegression(
        np.array([[-20], [160], [-80], [10], [-1]]), alpha=0.000001, max_iter=6000000))
    myLR.append(MyLinearRegression(
        np.array([[1140], [-1850], [1110], [-305], [40], [-2]]), alpha=0.00000001, max_iter=7000000))
    myLR.append(MyLinearRegression(
        np.array([[9110], [-18015], [13400], [-4935], [966], [-96.4], [3.86]]), alpha=0.000000001, max_iter=10000000))

    # initiate plots
    fig, axe = plt.subplots(1, 2, figsize=(15, 10))
    axe[1].plot(X, y, 'o', label='Score')
    axe[1].set_xlabel('blue pills consumed (in micrograms)')
    axe[1].set_ylabel('Spacecraft driving test scores')
    axe[1].set_title('Spacecraft driving test results / blue pill consumption')

    axe[0].set_xlabel('Polynomial level')
    axe[0].set_ylabel('Mean Squared Error')
    axe[0].set_title('MSE of models of different polynomial levels')

    for i, model in enumerate(myLR):
        model.fit_(x_lst[i], y)
        y_hat = model.predict_(x_lst[i])
        mse = model.loss_(y, y_hat)
        print(f"Model {i + 1} score: {mse}\n")
        axe[0].bar(i + 1, mse, label=f"Model {i + 1}")
        axe[1].plot(X, y_hat, label=f"Model {i + 1}")

    axe[1].legend()
    axe[0].legend()

    plt.show()


if __name__ == '__main__':
    main()
