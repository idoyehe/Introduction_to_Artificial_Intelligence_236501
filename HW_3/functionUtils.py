import pickle
from random import sample
from math import sqrt
from hw3_utils import load_data

FOLDS_PATH = r'./'


def euclidean_distance(feature_list1, feature_list2):
    assert (len(feature_list1) == len(feature_list2))  # TODO:remove before submission
    distance = 0
    for i in range(len(feature_list1)):
        distance += pow((feature_list1[i] - feature_list2[i]), 2)

    return sqrt(distance)


def filename_gen(fold_index: int):
    global FOLDS_PATH
    return FOLDS_PATH + "ecg_fold_" + str(fold_index) + ".data"


def split_crosscheck_groups(dataset, num_folds):
    train_features, train_labels = dataset

    samples_per_fold = len(train_features) // num_folds
    indexes_per_fold = []

    indexes = [i for i in range(len(train_features))]

    for i in range(num_folds - 1):
        samples = sample(indexes, samples_per_fold)  # choosing random features indexes
        indexes_per_fold.append(samples)

        for s in samples:  # removing chosen indexes
            indexes.remove(s)

    indexes_per_fold.append(indexes)

    for fold_index, indexes_in_fold in enumerate(indexes_per_fold):
        fold_index += 1
        fold_train_features = [train_features[content_index] for content_index in indexes_in_fold]
        fold_train_labels = [train_labels[content_index] for content_index in indexes_in_fold]

        pickle.dump((fold_train_features, fold_train_labels, None), open(filename_gen(fold_index), 'wb'))  # save dumped file


def load_k_fold_data(fold_index: int):
    train_features, train_labels, test_features = load_data(path=filename_gen(fold_index))
    assert test_features is None  # TODO:remove before submission
    return train_features, train_labels


""" single call to fold spilt"""
train_features_ds, train_labels_ds, test_features_ds = load_data()
split_crosscheck_groups((train_features_ds, train_labels_ds), 2)
