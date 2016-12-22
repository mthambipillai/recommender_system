import numpy as np 
from matrix_factorization import matrix_factorization_SGD_submission
from biaises import biaises
from helpers import load_data, preprocess_data, write_data

path_dataset = "data/data_train.csv"
ratings = load_data(path_dataset)

num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
min_num_ratings=10
valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
valid_ratings = ratings[valid_items, :][: , valid_users]

b = biaises(valid_ratings)
user_features, item_features = matrix_factorization_SGD_submission(valid_ratings, b, num_epochs = 300, num_features = 20)

product = (item_features @ user_features.T) + b
preds = valid_ratings.copy()
nz = preds.nonzero()
preds[nz] = product[nz[0],nz[1]]


path_submission = "data/trysubmission.csv"
write_data(path_submission, preds)