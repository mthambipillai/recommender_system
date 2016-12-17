import numpy as np

def baseline_global_mean(train, test):
    """baseline method: use the global mean.

    Args:
        train: The training set
        test: The test set

    Returns:
        The mean over the training set, the mean over the test set
    """
    train_mean = train[train.nonzero()].mean()
    test_mean = test[test.nonzero()].mean()
    
    return train_mean, test_mean

def baseline_user_mean(train, test):
    """baseline method: use the user means as the prediction.

    Args:
        train: The training set
        test: The test set

    Returns:
        The user means over the training set, the user means over the test set
    """
    mse = 0
    num_items, num_users = train.shape
    
    train_means = train.sum(axis = 0) / (train.getnnz(axis=0)+1e-12)
    test_means = test.sum(axis = 0) / (test.getnnz(axis=0)+1e-12)
    
    return train_means, test_means

def baseline_item_mean(train, test):
    """baseline method: use item means as the prediction.

    Args:
        train: The training set
        test: The test set

    Returns:
        The item means over the training set, the item means over the test set
    """
    mse = 0
    num_items, num_users = train.shape
    
    train_means = train.sum(axis = 1) / (train.getnnz(axis=1)+1e-12).reshape(num_items, 1)
    test_means = test.sum(axis = 1) / (test.getnnz(axis=1)+1e-12).reshape(num_items, 1)
    
    return train_means, test_means

def compute_rmse(valid_ratings, pred):
    """Computes the RMSE.

    Args:
        valid_ratings: The ratings on which we compare our prediction
        pred: The prediction

    Returns:
        The RMSE of the prediction over the ratings
    """
    nz = valid_ratings.nonzero()
    diff = valid_ratings[nz] - pred[nz]
    rmse = np.sqrt(np.sum(np.square(diff)) / len(nz[0]))
    return rmse