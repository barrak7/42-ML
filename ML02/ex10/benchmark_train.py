import numpy as np
import pandas as pd
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


def poly_reg(x, y, lvl):
    x_lst = []

    f = open('models.csv', 'a')
    w = csv.writer(f)

    # creating data for polymonial modules
    for e in range(2, lvl + 2):
        x_lst.append(np.concatenate(
            (np.vander(x[:, 0], e)[:, :-1], np.vander(x[:, 1], e)[:, :-1], np.vander(x[:, 2], e)[:, :-1]), 1))

    # initalizing the modules
    modules: list[MyLinearRegression] = []
    for e in range(4, 14, 3):
        modules.append(MyLinearRegression(
            np.ones((e, 1)), alpha=1, max_iter=100000000))

    y_hats = []

    # training module 1
    modules[0].fit_(x_lst[0], y)
    # modules[0].theta = np.array(
    #     [520515.3183257043, 10388511.107938895, -26299.392023240744, -592189.1483410596]).reshape(-1, 1)
    print('---- Module 1 ----\n')
    y_hats.append(modules[0].predict_(x_lst[0]))
    print(f'Module 1 MSE: {modules[0].loss_(y, y_hats[0])}')
    w.writerow(modules[0].theta.flatten())

    # training module 2
    modules[1].fit_(x_lst[1], y)
    # modules[1].theta = np.array([933179.8866939729, -6790852.585587305, 10545823.305779511,
    #                             907232.3165369655, -1289057.5938945322, -749.8343221693696, -545869.1684212943]).reshape(-1, 1)
    y_hats.append(modules[1].predict_(x_lst[1]))
    print(f'Module 2 MSE: {modules[1].loss_(y_hats[1], y)}')
    w.writerow(modules[1].theta.flatten())

    # training module 3
    modules[2].fit_(x_lst[2], y)
    # modules[2].theta = np.array([118881.78271904068, -22730.917657761656, -438863.35269656795, 10337318.820257034, 2726117.3886806453, -
    #                             4800396.580546383, 2534398.481918348, -3.915898646258405, -1374.1055360446642, -349930.61720909696]).reshape(-1, 1)
    y_hats.append(modules[2].predict_(x_lst[2]))
    print(f'Module 3 MSE: {modules[2].loss_(y, y_hats[2])}')
    w.writerow(modules[2].theta.flatten())

    # training module 4
    modules[3].alpha = 0.9
    modules[3].fit_(x_lst[3], y)
    # modules[3].theta = np.array([137210.08257959323, -8435.493525015238, -259177.7680343705, -6126571.914023113, 10494453.064273758, 251669.58386292643,
    #                             2125114.24251287, -4299464.898449173, 2365208.535874637, 1.0488872838336898, 7.51601224899672, -34.84727812221522, -384633.00880223635]).reshape(-1, 1)
    y_hats.append(modules[3].predict_(x_lst[3]))
    print(f'Module 4 MSE: {modules[3].loss_(y, y_hats[3])}')
    w.writerow(modules[3].theta.flatten())

    return modules


def main():
    data = pd.read_csv('space_avocado.csv').to_numpy()[:, 1:]
    data[:, :-1] = (data[:, :-1] - np.min(data[:, :-1])) / \
        (np.max(data[:, :-1]) - np.min(data[:, :-1]))

    x_train, y_train, x_test, y_test = data_spliter(
        data[:, :-1], data[:, -1:], 0.7)

    x_lst = []

    # creating test data for polymonial modules
    for e in range(2, 6):
        x_lst.append(np.concatenate(
            (np.vander(x_test[:, 0], e)[:, :-1], np.vander(x_test[:, 1], e)[:, :-1], np.vander(x_test[:, 2], e)[:, :-1]), 1))

    modules: list[MyLinearRegression] = poly_reg(x_train, y_train, 4)
    print('--- module 1 on test set ---')
    print(modules[0].loss_(modules[0].predict_(x_lst[0]), y_test), '\n')

    print('--- module 2 on test set ---')
    print(modules[1].loss_(modules[1].predict_(x_lst[1]), y_test), '\n')

    print('--- module 3 on test set ---')
    print(modules[2].loss_(modules[2].predict_(x_lst[2]), y_test), '\n')

    print('--- module 4 on test set ---')
    print(modules[3].loss_(modules[3].predict_(x_lst[3]), y_test), '\n')

    print(x_lst[3][30:33, :], '\n')
    print(y_test[30:33, :], '\n')

    print(modules[3].predict_(x_lst[3][30:33, :]))


if __name__ == '__main__':
    main()
