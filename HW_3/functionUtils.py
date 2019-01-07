import pickle
from random import sample, shuffle
from math import sqrt
from hw3_utils import load_data

FOLDS_PATH = r'./'


def euclidean_distance(feature_list1, feature_list2):
    assert (len(feature_list1) == len(feature_list2))  # TODO:remove before submission
    distance = 0
    for i in range(len(feature_list1)):
        distance += pow(feature_list1[i] - feature_list2[i], 2)
    distance = sqrt(distance)
    return distance


def filename_gen(fold_index: int):
    global FOLDS_PATH
    return FOLDS_PATH + "ecg_fold_" + str(fold_index) + ".data"


def split_crosscheck_groups(dataset, num_folds):
    train_features, train_labels = dataset

    false_labels_indexes = [false_index for false_index in range(len(train_labels)) if not train_labels[false_index]]
    true_labels_indexes = [true_index for true_index in range(len(train_labels)) if train_labels[true_index]]

    false_percent = len(false_labels_indexes) / len(train_labels)
    samples_per_fold = len(train_features) // num_folds
    samples_per_fold_false = int(false_percent * samples_per_fold)
    samples_per_fold_true = samples_per_fold - samples_per_fold_false

    indexes_per_fold = []

    for i in range(num_folds - 1):
        # choosing random true features indexes
        samples_true = sample(true_labels_indexes, samples_per_fold_true)
        # choosing random false features indexes
        samples_false = sample(false_labels_indexes, samples_per_fold_false)

        current_fold_indexes = samples_true + samples_false
        shuffle(current_fold_indexes)
        indexes_per_fold.append(current_fold_indexes)

        for s in samples_true:  # removing chosen true labels indexes
            true_labels_indexes.remove(s)

        for s in samples_false:  # removing chosen false labels indexes
            false_labels_indexes.remove(s)

    current_fold_indexes = false_labels_indexes + true_labels_indexes
    shuffle(current_fold_indexes)
    indexes_per_fold.append(current_fold_indexes)

    for fold_index, indexes_in_fold in enumerate(indexes_per_fold):
        fold_index += 1
        fold_train_features = [train_features[content_index] for content_index in indexes_in_fold]
        fold_train_labels = [train_labels[content_index] for content_index in indexes_in_fold]

        pickle.dump((fold_train_features, fold_train_labels, None), open(filename_gen(fold_index), 'wb'))  # save dumped file


def load_k_fold_data(fold_index: int):
    train_features, train_labels, test_features = load_data(path=filename_gen(fold_index))
    assert test_features is None  # TODO:remove before submission
    return train_features, train_labels
