import numpy as np
import pandas as pd
from ridge import MyRidge as MR
import matplotlib.pyplot as plt


def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power giveArgs:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        power: has to be an int, the power up to which the columns of matrix x are going to be raised.
    Returns:
        The matrix of polynomial features as a numpy.ndarray, of shape m * (np), containg the polynomial feature vaNone if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    try:
        return np.concatenate([x ** i for i in range(1, power + 1)], axis=1)
    except:
        return


def main():
    data = pd.read_csv('train.csv').to_numpy()[:, 1:]
    # training set
    y = data[:, -1]
    data = data[:, :-1]
    data = (data - data.min()) / (data.max() - data.min())

    # cross validation set
    x_c = pd.read_csv('CV.csv').to_numpy()[:, 1:]
    y_c = x_c[:30, -1]
    x_c = x_c[:30, :-1]

    # test set
    x_t = pd.read_csv('test.csv').to_numpy()[:, 1:]
    y_t = x_t[:30, -1]
    x_t = x_t[:30, :-1]

    models: list[list[MR]] = []
    best_l = 0
    bm_i = 0

    f = open("models.csv", 'w')
    for i in range(1, 5):
        x = add_polynomial_features(data, i)
        models.append([])
        for e in np.arange(0, 1, 0.2):
            models[-1].append(MR(np.ones((x.shape[1] + 1, 1)),
                              max_iter=1000, lambda_=e, alpha=0.001))
            models[-1][-1].fit_(x, y)
            y_hat = models[-1][-1].predict_(add_polynomial_features(x_c, i))
            loss = models[-1][-1].loss_(y_c, y_hat)
            if best_l == 0:
                best_m = models[-1][-1]
                bm_i = i
            elif loss < best_l:
                best_m = models[-1][-1]
                bm_i = i
            f.write(str(models[-1][-1].get_params_))

    f.close()
    print(
        f"The best module out of 20 is : theta = {best_m.theta}; alpha = {best_m.alpha}; lambda_ : {best_m.lambda_} with a loss of {best_m.loss_(y_t, best_m.predict_(add_polynomial_features(x_t, bm_i)))}")
    fig, axe = plt.subplots(2, figsize=(12, 12))
    lmbda = np.arange(0, 1, 0.2)

    for i, model in enumerate(models, 1):
        losses = []
        for ml in model:
            losses.append(ml.loss_(y, ml.predict_(
                add_polynomial_features(x_t, i))))
        axe[0].plot(lmbda, losses, label=f"Module - {i}")
    axe[0].set_title("Learning Curves")
    axe[0].set_xlabel('Lambda')
    axe[0].set_ylabel('Loss')

    axe[1].plot(x_t[:, 0], y_t, label='Price')
    axe[1].plot(x_t[:, 0], best_m.predict_(
        add_polynomial_features(x_t, bm_i)), label='Predicted Price')
    axe[1].set_title("Prediction of the best module")
    axe[1].set_xlabel("Avocado weight")
    axe[1].set_ylabel("Space avocado price")

    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
