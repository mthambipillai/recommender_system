import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp

def filter_outliers(train, disagreements, threshold = 2.5):
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
    num_items, num_users = train.shape
    train_sums = train.sum(axis = 1) #sum all users ratings for each item
    train_nnz = train.getnnz(axis=1)
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    disagreements = sp.lil_matrix((num_items, num_users))

    for item, user in nz_train:
        item_sum = train_sums[item]
        item_nnz = train_nnz[item]-1

        rating = train[item,user]

        mean_without_user = (item_sum-rating)/item_nnz

        disagreements[item,user] = abs(mean_without_user - rating)

    return disagreements

def plot_disagreements(d):
    nz_row, nz_col = d.nonzero()
    nz_d = list(zip(nz_row, nz_col))
    l = np.array([])
    for i, u in nz_d:
        l = np.append(l,d[i,u])
    print(sum(l)/len(l))
    plt.plot(sorted(l))