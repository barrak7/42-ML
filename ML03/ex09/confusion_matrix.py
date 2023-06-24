from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        labels: optional, a list of labels to index the matrix.
        This may be used to reorder or select a subset of labels. (default=None)
        df_option: optional, if set to True the function will return a pandas DataFrame
        instead of a numpy array. (default=False)
    Return:
        The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
        None if any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        if labels == None:
            labels = np.unique(np.vstack((y_true, y_hat)))
        else:
            labels = np.array(labels)
        cm = np.zeros((labels.shape[0] * labels.shape[0]))
        i = 0

        for label in labels:
            for v in labels:
                y_h = y_hat.copy()
                y_h = y_hat[y_true == label]
                cm[i,] = y_h[y_h == v].shape[0]
                i += 1

        cm = cm.reshape(labels.shape[0], labels.shape[0])
        labels = labels.tolist()
        if (df_option == True):
            return (pd.DataFrame(cm, columns=labels, index=labels))
        return cm
    except:
        return


y_hat = np.array([['norminet'], ['dog'], ['norminet'],
                 ['norminet'], ['dog'], ['bird']])
y = np.array([['dog'], ['dog'], ['norminet'], [
             'norminet'], ['dog'], ['norminet']])
# Example 1:
# your implementation
print(confusion_matrix_(y, y_hat))
# Output:
# array([[0 0 0]
# [0 2 1]
# [1 0 2]])
# sklearn implementation
print(confusion_matrix(y, y_hat))
# Output:
# array([[0 0 0]
# [0 2 1]
# [1 0 2]])
# Example 2:
# your implementation
print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))
# Output:
# array([[2 1]
# [0 2]])
# sklearn implementation
print(confusion_matrix(y, y_hat, labels=['dog', 'norminet']))
# Output:
# array([[2 1]
# [0 2]])
# Example 3:
print(confusion_matrix_(y, y_hat, df_option=True))
# Output:
#             bird    dog     norminet
# bird        0       0       0
# dog         0       2       1
# norminet    1       0       2
# Example 2:
print(confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True))
# Output:
#         bird    dog
# bird    0       0
# dog     0       2
