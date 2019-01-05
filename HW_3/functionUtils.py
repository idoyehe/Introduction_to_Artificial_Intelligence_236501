import pickle
from random import sample
from hw3_utils import load_data


def euclidean_distance(feature_list1, feature_list2):
    assert (len(feature_list1) == len(feature_list2))
    distance = 0
    for i in range(len(feature_list1)):
        distance += (feature_list1[i] - feature_list2[i]) ** 2

    return distance ** 0.5


def save_data(path, index, train_features, train_labels, test_features):
    filename = path + "ecg_fold_" + str(index) + ".data"
    pickle.dump((train_features, train_labels, test_features), open(filename, 'wb'))


def split_crosscheck_groups(dataset, num_folds):
    train_features, train_labels = dataset

    dataset_len = len(train_features)
    fold_len = dataset_len // num_folds
    indexes_per_fold = []

    indexes = [i for i in range(dataset_len)]

    for i in range(num_folds - 1):
        samples = sample(indexes, fold_len)  # choosing random features indexes
        indexes_per_fold.append(samples)
        for s in samples:  # removing chosen indexes
            indexes.remove(s)

    indexes_per_fold.append(indexes)

    path = r'data/'
    for fold_index, fold_content in enumerate(indexes_per_fold):
        fold_train_features = []
        fold_train_labels = []
        for content_index in fold_content:
            fold_train_features.append(train_features[content_index])
            fold_train_labels.append(train_labels[content_index])
        save_data(path, fold_index, fold_train_features, fold_train_labels, None)


# split_crosscheck_groups(([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
#                          [True, True, True, True, True, True, True, True, True, True]), 2)
#
# train_features, train_labels, test_features = load_data(r'data/ecg_fold_0.data')
# pass
# train_features, train_labels, test_features = load_data(r'data/ecg_fold_1.data')
# pass
