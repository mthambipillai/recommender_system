# -*- coding: utf-8 -*-
"""some functions for plots."""

import numpy as np
import matplotlib.pyplot as plt


def plot_raw_data(ratings):
    """plot the statistics result on raw rating data."""
    # do statistics.
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    sorted_num_movies_per_user = np.sort(num_items_per_user)[::-1]
    sorted_num_users_per_movie = np.sort(num_users_per_item)[::-1]

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(sorted_num_movies_per_user, color='blue')
    ax1.set_xlabel("users")
    ax1.set_ylabel("number of ratings (sorted)")
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(sorted_num_users_per_movie)
    ax2.set_xlabel("items")
    ax2.set_ylabel("number of ratings (sorted)")
    ax2.set_xticks(np.arange(0, 2000, 300))
    ax2.grid()

    plt.tight_layout()
    plt.savefig("stat_ratings")
    plt.show()
    # plt.close()
    return num_items_per_user, num_users_per_item


def plot_train_test_data(train, test):
    """visualize the train and test data."""
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.spy(train, precision=0.01, markersize=0.05, aspect="auto")
    ax1.set_xlabel("Users")
    ax1.set_ylabel("Items")
    ax1.set_title("Training data")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.spy(test, precision=0.01, markersize=0.05, aspect="auto")
    ax2.set_xlabel("Users")
    ax2.set_ylabel("Items")
    ax2.set_title("Test data")
    plt.tight_layout()
    plt.savefig("train_test")
    plt.show()
    
def plot_average_rating_and_number_per_range(df, avg_rating_col, number_col, ranges_col):
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Users grouped in categories, by number of ratings they gave", fontsize=20)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(df[ranges_col], df[avg_rating_col], marker="o", color='b', label='average rating according to number of ratings')
    ax1.set_xlabel("Number of ratings per user in this category")
    ax1.set_ylabel("Average rating of users in this category")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.semilogy(df[ranges_col], df[number_col], marker="o", color='r', label='number of users according to number ratings')
    ax2.set_xlabel("Number of ratings per user in this category")
    ax2.set_ylabel("Number of users in this category")
    ax2.set_title("Test data")
    plt.savefig("users_categories")
    plt.show()
    
    
def plot_rmse_per_categories(df, y_col, item_col, user_col, mf_col, xlabel):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df[y_col], df[item_col], marker=".", color='r', label="item mean based rmse")
    ax.plot(df[y_col], df[user_col], marker=".", color='g', label="user mean based rmse")
    ax.plot(df[y_col], df[mf_col], marker=".", color='b', label="mf rmse")
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE")
    plt.show()
    
def visualization(epochs, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(epochs, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(epochs, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("k")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
