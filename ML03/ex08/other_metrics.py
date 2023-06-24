from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    Returns:
    The accuracy score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    try:
        tpn = y[y == y_hat].shape[0]
        return (tpn/y_hat.shape[0])
    except:
        return


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The precision score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    try:
        tp = y[y == y_hat]
        tp = tp[tp == pos_label].shape[0]
        fntp = y_hat[y_hat == pos_label].shape[0]
        return (tp / fntp)
    except:
        return


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The recall score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    try:
        tp = y[y == y_hat]
        tp = tp[tp == pos_label].shape[0]
        fn = y_hat[y_hat != y]
        fn = fn[fn != pos_label].shape[0]
        return (tp / (tp + fn))
    except:
        return


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
    The f1 score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    try:
        p = precision_score_(y, y_hat, pos_label)
        r = recall_score_(y, y_hat, pos_label)
        return (2 * p * r) / (p + r)
    except:
        return


# tests

# Example 1:
y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))
# Accuracy
# your implementation
print(accuracy_score_(y, y_hat))
# Output:
0.5
# sklearn implementation
print(accuracy_score(y, y_hat))
# Output:
0.5
# Precision
# your implementation
print(precision_score_(y, y_hat))
# Output:
0.4
# sklearn implementation
print(precision_score(y, y_hat))
# Output:
0.4
# Recall
# your implementation
print(recall_score_(y, y_hat))
# Output:
0.6666666666666666
# sklearn implementation
print(recall_score(y, y_hat))
# Output:
0.6666666666666666
# F1-score
# your implementation
print(f1_score_(y, y_hat))
# Output:
0.5
# sklearn implementation
print(f1_score(y, y_hat))
# Output
0.5
# Example 2:
y_hat = np.array(['norminet', 'dog', 'norminet',
                 'norminet', 'dog', 'dog', 'dog', 'dog'])
y = np.array(['dog', 'dog', 'norminet', 'norminet',
             'dog', 'norminet', 'dog', 'norminet'])
# Accuracy
# your implementation
print(accuracy_score_(y, y_hat))
# Output:
0.625
# sklearn implementation
print(accuracy_score(y, y_hat))
# Output:
0.625
# Precision
# your implementation
print(precision_score_(y, y_hat, pos_label='dog'))
# Output:
0.6
# sklearn implementation
print(precision_score(y, y_hat, pos_label='dog'))
# Output:
0.6
# Recall
# your implementation
print(recall_score_(y, y_hat, pos_label='dog'))
# Output:
0.75
# sklearn implementation
print(recall_score(y, y_hat, pos_label='dog'))
# Output:
0.75
# F1-score
# your implementation
print(f1_score_(y, y_hat, pos_label='dog'))
# Output:
0.6666666666666665
# sklearn implementation
print(f1_score(y, y_hat, pos_label='dog'))
# Example 3:
y_hat = np.array(['norminet', 'dog', 'norminet',
                 'norminet', 'dog', 'dog', 'dog', 'dog'])
y = np.array(['dog', 'dog', 'norminet', 'norminet',
             'dog', 'norminet', 'dog', 'norminet'])
# Precision
# your implementation
print(precision_score_(y, y_hat, pos_label='norminet'))
# Output:
0.6666666666666666
# sklearn implementation
print(precision_score(y, y_hat, pos_label='norminet'))
# Output:
0.6666666666666666
# Recall
# your implementation
print(recall_score_(y, y_hat, pos_label='norminet'))
# Output:
0.5
# sklearn implementation
print(recall_score(y, y_hat, pos_label='norminet'))
# Output:
0.5
# F1-score
# your implementation
print(f1_score_(y, y_hat, pos_label='norminet'))
# Output:
0.5714285714285715
# sklearn implementation
print(f1_score(y, y_hat, pos_label='norminet'))
# Output:
0.5714285714285715
