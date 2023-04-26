import numpy as np
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

    return np.sum(y == y_hat) / y.shape[0]

@DataChecker
def precision_score_(y, y_hat, pos_label = 1):
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

    tp = np.sum((y == pos_label) & (y_hat == pos_label))
    fp = np.sum((y != pos_label) & (y_hat == pos_label))

    return tp / (tp + fp)

@DataChecker
def recall_score_(y, y_hat, pos_label = 1):
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

    tp = np.sum((y == pos_label) & (y_hat == pos_label))
    fn = np.sum((y == pos_label) & (y_hat != pos_label))

    return tp / (tp + fn)

@DataChecker
def f1_score_(y, y_hat, pos_label = 1):
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

    tp = np.sum((y == pos_label) & (y_hat == pos_label))
    fp = np.sum((y != pos_label) & (y_hat == pos_label))
    fn = np.sum((y == pos_label) & (y_hat != pos_label))

    return (2 * (tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn)))

if __name__ == '__main__':

    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # # # Example 1:
    print('\n******************** Example 1 ********************\n')
    
    y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))
    
    # Accuracy
    print('\n============ Accuracy ============')

    ## your implementation
    print(accuracy_score_(y, y_hat))
    ## Output:
    # 0.5

    ## sklearn implementation
    print(accuracy_score(y, y_hat))
    ## Output:
    # 0.5
    
    # Precision
    print('\n============ Precision ============')

    ## your implementation
    print(precision_score_(y, y_hat))
    ## Output:
    # 0.4
    
    ## sklearn implementation
    print(precision_score(y, y_hat))
    ## Output:
    # 0.4
    
    # Recall
    print('\n============ Recall ============')

    ## your implementation
    print(recall_score_(y, y_hat))
    ## Output:
    # 0.6666666666666666
    
    ## sklearn implementation
    print(recall_score(y, y_hat))
    ## Output:
    # 0.6666666666666666
    
    # F1-score
    print('\n============ F1-score ============')

    ## your implementation
    print(f1_score_(y, y_hat))
    ## Output:
    # 0.5
    
    ## sklearn implementation
    print(f1_score(y, y_hat), '\n')
    ## Output:
    # 0.5

    # # #  Example 2:
    print('\n******************** Example 2 ********************\n')
    
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog']).reshape(-1, 1)
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet']).reshape(-1, 1)
    
    # Accuracy
    print('\n============ Accuracy ============')

    ## your implementation
    print(accuracy_score_(y, y_hat))
    ## Output:
    # 0.625
    
    ## sklearn implementation
    print(accuracy_score(y, y_hat))
    ## Output:
    # 0.625
    
    # Precision
    print('\n============ Precision ============')

    ## your implementation
    print(precision_score_(y, y_hat, pos_label='dog'))
    ## Output:
    # 0.6
    
    # ## sklearn implementation
    print(precision_score(y, y_hat, pos_label='dog'))
    ## Output:
    # 0.6
    
    # Recall
    print('\n============ Recall ============')

    ## your implementation
    print(recall_score_(y, y_hat, pos_label='dog'))
    ## Output:
    # 0.75
    
    ## sklearn implementation
    print(recall_score(y, y_hat, pos_label='dog'))
    ## Output:
    # 0.75
    
    # F1-score
    print('\n============ F1-score ============')

    ## your implementation
    print(f1_score_(y, y_hat, pos_label='dog'))
    ## Output:
    # 0.6666666666666665
    
    ## sklearn implementation
    print(f1_score(y, y_hat, pos_label='dog'), '\n')
    ## Output:
    # 0.6666666666666665

    # # # Example 3:
    print('\n******************** Example 3 ********************\n')
    
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

    # Precision
    print('\n============ Precision ============')

    ## your implementation
    print(precision_score_(y, y_hat, pos_label='norminet'))
    ## Output:
    # # 0.6666666666666666

    ## sklearn implementation
    print(precision_score(y, y_hat, pos_label='norminet'))
    ## Output:
    # # 0.6666666666666666

    # Recall
    print('\n============ Recall ============')

    ## your implementation
    print(recall_score_(y, y_hat, pos_label='norminet'))
    ## Output:
    # # 0.5

    ## sklearn implementation
    print(recall_score(y, y_hat, pos_label='norminet'))
    ## Output:
    # # 0.5

    # F1-score
    print('\n============ F1-score ============')

    ## your implementation
    print(f1_score_(y, y_hat, pos_label='norminet'))
    ## Output:
    # # 0.5714285714285715

    ## sklearn implementation
    print(f1_score(y, y_hat, pos_label='norminet'), '\n')
    ## Output:
    # # 0.5714285714285715