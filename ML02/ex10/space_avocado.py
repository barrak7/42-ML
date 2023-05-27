import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the
        training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible dimensions.
        None if x, y or proportion is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        data = np.concatenate((x, y), 1)
        data_splt = np.array_split(data, [int(proportion * data.shape[0])], 0)
        lst = []
        for e in data_splt:
            e = np.array_split(e, [3], 1)
            for i in e:
                lst.append(i)
        return tuple(lst)
    except:
        return


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
                if np.all(np.absolute(self.theta - old_th) <= 0.0000001):
                    break
                max -= 1
            return self.theta
        except:
            return


def main():
    data = pd.read_csv('space_avocado.csv').to_numpy()[:, 1:]
    x = pd.read_csv('space_avocado.csv').to_numpy()[:, 1:]

    data[:, :-1] = (data[:, :-1] - np.min(data[:, :-1])) / \
        (np.max(data[:, :-1]) - np.min(data[:, :-1]))

    x_train, y_train, x_test, y_test = data_spliter(
        data[:, :-1], data[:, -1:], 0.7)

    x_train = np.concatenate((np.vander(x_train[:, 0], increasing=True, N=5)[:, 1:], np.vander(
        x_train[:, 1], increasing=True, N=5)[:, 1:], np.vander(x_train[:, 2], increasing=True, N=5)[:, 1:]), 1)
    x_test = np.concatenate((np.vander(x_test[:, 0], increasing=True, N=5)[:, 1:], np.vander(
        x_test[:, 1], increasing=True, N=5)[:, 1:], np.vander(x_test[:, 2], increasing=True, N=5)[:, 1:]), 1)

    MyLR4 = MyLinearRegression(
        np.ones((13, 1)), 0.9, max_iter=100000000)

    MyLR4.fit_(x_train, y_train)

    MyLR4.theta = np.array([-606886.2760054065, 11799101.068385752, -53126054.2316492, -2212428.470560373, -71632.7373920127, 6950287.068415019, -14616281.195844509,
                            12129217.078146117, -3285971.1699807504, -393942.6346091833, 6295.956369318442, 69.20167701288442, 1.4283220520866542]).reshape(-1, 1)
    y_hat = MyLR4.predict_(x_test)

    print(f"My module's loss: {MyLR4.loss_(y_hat, y_test)}")
    print(y_test[20:25, :])
    print(y_hat[20:25, :])
    f = open('models.csv', 'a')
    w = csv.writer(f)
    w.writerow(MyLR4.theta.flatten())

    fig, axe = plt.subplots(1, 3, figsize=(30, 12), sharey=True)
    axe[0].plot(x_test[:, 0], y_test, 'o', label='Price')

    axe[0].plot(x_test[:, 0], y_hat,
                'o', label='Predicted Price')

    axe[0].set_title('Avocado price as a function of Weight')
    axe[0].set_xlabel('Avocado order weight (in Ton)')
    axe[0].set_ylabel('Price (in Trantorian unit)')

    axe[0].legend()

    axe[1].plot(x_test[:, 4], y_test, 'o', label='Price')

    axe[1].plot(x_test[:, 4], y_hat,
                'o', label='Predicted Price')

    axe[1].set_title('Avocado price as a function of production distance')
    axe[1].set_xlabel('Avocado order production distance (in Mkm)')

    axe[1].legend()

    axe[2].plot(x_test[:, 8], y_test, 'o', label='Price')

    axe[2].plot(x_test[:, 8], y_hat,
                'o', label='Predicted Price')

    axe[2].set_title('Avocado price as a function of delivery time')
    axe[2].set_xlabel('Avocado order delivery time (in days)')

    axe[2].legend()

    plt.show()

    f.close()


if __name__ == '__main__':
    main()
