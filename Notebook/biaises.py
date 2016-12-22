import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp

def biaises(data):
    """computes the biaises for all ratings in data.

    Args:
        data: The data on which the biaises are computed

    Returns:
        the biaises for all ratings in data
    """
    print("Computing the biaises of the data...")
    num_items, num_users = data.shape
    mean = data[data.nonzero()].mean()
    user_means = data.sum(axis = 0).reshape(num_users, 1) / (data.getnnz(axis=0)+1e-12).reshape(num_users, 1)
    item_means = (data.sum(axis = 1) / (data.getnnz(axis=1)+1e-12).reshape(num_items, 1))

    nz_row, nz_col = data.nonzero()
    nz_data = list(zip(nz_row, nz_col))

    biaises = sp.lil_matrix(data.shape)
    for item, user in nz_data:
        user_deviation = user_means[user] - mean
        item_deviation = item_means[item] - mean
        biaises[item, user] = mean + user_deviation + item_deviation

    return biaises, mean, item_means, user_means

def submission_biaises(submission_sample, mean, item_means, user_means):
    num_items, num_users = submission_sample.shape
    nz_row, nz_col = submission_sample.nonzero()
    nz_data = list(zip(nz_row, nz_col))

    biaises = sp.lil_matrix(submission_sample.shape)
    for item, user in nz_data:
        user_deviation = user_means[user] - mean
        item_deviation = item_means[item] - mean
        biaises[item, user] = mean + user_deviation + item_deviation

    return biaises