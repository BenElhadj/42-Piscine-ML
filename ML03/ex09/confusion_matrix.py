import numpy as np
import pandas as pd
np.seterr(all='ignore')

def DataChecker(func):
    def wrapper(*args, **kwargs):
        y, y_hat = args[0], args[1]
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)\
            or (not np.issubdtype(y.dtype, np.number) and not np.issubdtype(y.dtype, np.str_))\
            or (not np.issubdtype(y_hat.dtype, np.number) and not np.issubdtype(y_hat.dtype, np.str_))\
            or y.dtype != y_hat.dtype or y.shape != y_hat.shape or (len(y.shape) == 2 and y.shape[1] != 1):
            raise BaseException("error type or shape of y or y_hat")
        res = func(*args, **kwargs)
        return res

    return wrapper

@DataChecker
def confusion_matrix_(y_true, y_hat, labels=None, df_option = False):
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

    if labels:
        len_labels = len(labels)

    else:
        labels = np.unique(np.vstack((y_true, y_hat)))
        len_labels = labels.shape[0]

    matrix = np.zeros(shape=(len_labels, len_labels), dtype=int)

    for i, pred_label in enumerate(labels):
        for j, true_label in enumerate(labels):
            matrix[j][i] += np.sum((y_hat == pred_label) & (y_true == true_label))

    if df_option:
        return pd.DataFrame(data=matrix, index=labels, columns=labels, dtype=int)
    
    return matrix

if __name__ == '__main__':

    import numpy as np
    from sklearn.metrics import confusion_matrix
    y_hat = np.array([['norminet'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['bird']])
    y = np.array([['dog'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['norminet']])
    
    # Example 1:
    print('\n******************** Example 1 ********************\n')
    ## your implementation
    print(confusion_matrix_(y, y_hat), '\n')
    ## Output:
    # array([[0 0 0]
    # [0 2 1]
    # [1 0 2]])
    ## sklearn implementation
    print(confusion_matrix(y, y_hat))
    ## Output:
    # array([[0 0 0]
    # [0 2 1]
    # [1 0 2]])
    
    # Example 2:
    print('\n******************** Example 2 ********************\n')
    ## your implementation
    print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']), '\n')
    ## Output:
    # array([[2 1]
    # [0 2]])
    ## sklearn implementation
    print(confusion_matrix(y, y_hat, labels=['dog', 'norminet']))
    ## Output:
    # array([[2 1]
    # [0 2]])
    
    #Example 3:
    print('\n******************** Example 3 ********************\n')    
    print(confusion_matrix_(y, y_hat, df_option=True))
    #Output:
    # bird dog norminet
    # bird 0 0 0
    # dog 0 2 1
    # norminet 1 0 2

    # #Example 2:
    print('\n******************** Example 4 ********************\n')    
    print(confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True))
    #Output:
    # bird dog
    # bird 0 0
    # dog 0 2