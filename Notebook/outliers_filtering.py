import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
from baselines import baseline_item_mean, compute_rmse

def filter_outliers(train, disagreements, threshold = 3.8):
    """For each rating in train, we keep it only if its disagreement is not above some threshold.

    Args:
        train: The training set
        disagreements: the disagreement of each rating (same shape as train)
        threshold: The value above which we filter
    Returns:
        The filtered training set
    """
    num_items, num_users = train.shape
    train_sums = train.sum(axis = 1) #sum all users ratings for each item
    train_nnz = train.getnnz(axis=1)
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    train_filtered = sp.lil_matrix((num_items, num_users))

    for item, user in nz_train:
        if(disagreements[item,user] <= threshold):
            train_filtered[item,user] = train[item,user]

    return train_filtered



def disagreements(train):
    """For each rating in train, we compute its disagreement : The absolute difference between
    the rating itself and the mean of all other ratings for the same item.

    Args:
        train: The training set

    Returns:
        The disagreement of each rating as a sparse matrix (same shape as train)
    """
    train_sums = train.sum(axis = 1) #sum all users ratings for each item
    train_nnz = train.getnnz(axis=1)
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    disagreements = sp.lil_matrix(train.shape)
    for item, user in nz_train:
        item_sum = train_sums[item]
        item_nnz = train_nnz[item]-1

        rating = train[item,user]

        mean_without_user = (item_sum-rating)/item_nnz
        disagreements[item,user] = abs(mean_without_user - rating)

    return disagreements

def plot_disagreements(d):
    """Plots the disagreements"""
    nz_row, nz_col = d.nonzero()
    nz_d = list(zip(nz_row, nz_col))
    l = np.empty(len(nz_d))#np.array([]) 
    for idx, e in enumerate(nz_d): 
        l[idx] = d[e[0], e[1]] 
    print(sum(l)/len(l))
    plt.plot(sorted(l))


def threshold_tests(d, train_unfiltered, valid_ratings, test):
    """For each threshold value between 3 and 4 with a step size of 0.05, computes
    the filtered training set and the rmse with the item mean.
    Then plots all the errors.
    """
    errors=[]
    for i in np.arange(3,4,0.05):
        train_i = filter_outliers(train_unfiltered,d,i)
        train_means, test_means = baseline_item_mean(train_i, test)
        train_means_list = train_means.tolist()
        pred = np.ones(train_i.shape)
        for col in range(train_i.shape[0]):
            pred[col,:] *= train_means_list[col]
        rmse = compute_rmse(valid_ratings, pred)
        errors.append(rmse)

    plt.xlabel("threshold")
    plt.ylabel("RMSE")
    plt.plot(np.arange(3,4,0.05),errors)
    plt.savefig("outlier_filtering")