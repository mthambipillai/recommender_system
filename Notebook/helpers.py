# -*- coding: utf-8 -*-
"""some functions for help."""

from itertools import groupby

import numpy as np
import scipy.sparse as sp
import pandas as pd


def write_data(path_dataset, submission_ratings):
    f = open(path_dataset, 'wt')
    nz_row, nz_col = submission_ratings.nonzero()
    try:
        writer = csv.writer(f)
        writer.writerow( ('Id', 'Prediction') )
        for i in range(0, len(nz_col)):
            ide = "r" + str(nz_col[i]+1) + "_c" + str(nz_row[i]+1)
            writer.writerow((ide, int(submission_ratings[nz_row[i],nz_col[i]] + 0.5)))
    finally:
        f.close()

def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data).T

# Read in data
def loadDataFrame(path):
    d = []
    with open(path, "r") as f:
        lines = f.read().splitlines()[1:]
    for line in lines:
        (row_col, rating)=line.split(',')
        row_col = row_col.replace("r", "")
        row_col = row_col.replace("c", "")
        row, col = row_col.split('_')
        d.append([int(row), int(col), float(rating)])
        
    data = pd.DataFrame(d, columns = ["user", "item", "rating"])

    return data


def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_col, max_row))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def group_by(data, index):
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices


def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)

import csv
import sys
def exportSubmission(path, pred):
    path_dataset = "data/sampleSubmission.csv"
    submission_ratings = load_data(path_dataset)
    submission_ratings.shape
    nz_row, nz_col = submission_ratings.nonzero()
    for i in range(0, len(nz_col)):
        submission_ratings[nz_row[i],nz_col[i]] = pred[nz_row[i],nz_col[i]]

    f = open(path, 'wt')
    try:
        writer = csv.writer(f)
        writer.writerow( ('Id', 'Prediction') )
        for i in range(0, len(nz_col)):
            ide = "r" + str(nz_col[i]+1) + "_c" + str(nz_row[i]+1)
            writer.writerow((ide, int(submission_ratings[nz_row[i],nz_col[i]] + 0.5)))
    finally:
        f.close()


def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):
    """split the ratings to training data and test data.
    Args:
        min_num_ratings: 
            all users and items we keep must have at least min_num_ratings per user and per item. 
    """
    # set seed
    np.random.seed(988)
    
    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][: , valid_users]  
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data and return train and test data. TODO
    # NOTE: we only consider users and movies that have more
    # than 10 ratings
    # ***************************************************
    
    # build rating matrix.
    rows, cols = ratings.get_shape()

    
    train = sp.lil_matrix((rows, cols))
    test = sp.lil_matrix((rows, cols))
    
    print(rows, cols)
    
    nz_row, nz_col = valid_ratings.nonzero()
    print(len(nz_col))
    print(len(nz_row))
    for i in range(0, len(nz_col)):
        rand = np.random.random()
        if rand > p_test:
            train[nz_row[i],nz_col[i]] = valid_ratings[nz_row[i],nz_col[i]]
        else:
            test[nz_row[i],nz_col[i]] = valid_ratings[nz_row[i],nz_col[i]]
               
    print("Total number of nonzero elements in original data:{v}".format(v=valid_ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test



def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # calculate rmse (we only consider nonzero entries.)
    # ***************************************************
    pred = item_features @ user_features.T
    diff = data[nz] - pred[nz]
    rmse = 1/2 * np.sum(np.square(diff)) / len(nz[0])
    
    return rmse