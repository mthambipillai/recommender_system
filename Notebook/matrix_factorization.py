import numpy as np
from plots import visualization
from helpers import compute_error
from baselines import baseline_global_mean

def init_MF(train, num_features):
    """init the parameters for matrix factorization. The matrices have all their
    entries set to the same value such that the matrix product gives average ratings.

    Args:
        train: The training set
        num_features: The number of features 'k'

    Returns:
        The 'Z' matrix for the user features (shape = num_features, num_user),
        The 'W' matrix for the item features (shape = num_features, num_item),
    """
    mean = baseline_global_mean(train, train)[0]
    num_items, num_users = train.shape
    user_features = np.ones((num_users, num_features))/np.sqrt(num_features/mean)
    item_features = np.ones((num_items, num_features))/np.sqrt(num_features/mean)
    
    return user_features, item_features


def matrix_factorization_SGD(train, test, num_epochs=50, num_features=20):
    """matrix factorization optimized by SGD.
    Args:
        train: The training set
        test: The test set
        num_epochs: The number of iterations for SGD
        num_features: The number of features 'k'

    Returns:
        The optimized 'Z' matrix for the user features (shape = num_features, num_user),
        The optimized 'W' matrix for the item features (shape = num_features, num_item),
    """
    # define parameters
    gamma = 0.0005
    lambda_user = 0.1
    lambda_item = 0.7
    errors = []
    errors_test = []
    
    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    

    rmse = compute_error(train, user_features, item_features, train.nonzero())

    print("iter: k = {}, RMSE on training set: {}.".format(num_features, rmse))

    print("learn the matrix factorization using SGD...")
    
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1
        
        #compute prediction
        mult = item_features @ user_features.T
        #init gradients
        item_grad = np.zeros(item_features.shape)
        user_grad = np.zeros(user_features.shape)
        nb = int(len(nz_train)/(5+np.sqrt(it)))
        for d, n in nz_train[:nb]:
            pred_error = train[d, n] - mult[d, n]
            #compute gradients
            item_grad[d,:] += pred_error * user_features[n,:] * gamma
            user_grad[n,:] += pred_error * item_features[d,:] * gamma
        
        #update item and user matrices
        item_features += item_grad
        user_features += user_grad
        
        #compute the train error
        rmse = compute_error(train, user_features, item_features, train.nonzero())
        #print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        errors.append(rmse)
        
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        
        #compute the test error
        rmse = compute_error(test, user_features, item_features, test.nonzero())
        #print("iter: {}, RMSE on test set: {}.".format(it, rmse))
        errors_test.append(rmse)
        
    print(len(errors))
    print(len(errors_test))
    print(np.linspace(1,num_epochs,num_epochs))
    #plot the train and test errors
    visualization(np.linspace(1,num_epochs,num_epochs),errors,errors_test)
    
    return user_features, item_features