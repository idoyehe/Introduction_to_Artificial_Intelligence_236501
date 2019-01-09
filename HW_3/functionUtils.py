from hw3_utils import abstract_classifier_factory
from random import sample, shuffle
from math import sqrt
import pickle

FOLDS_PATH = r'./'


def euclidean_distance(feature_list1, feature_list2):
    assert (len(feature_list1) == len(feature_list2))
    distances_list = [pow(cf1 - cf2, 2) for cf1, cf2 in zip(feature_list1, feature_list2)]
    distance = sqrt(sum(distances_list))
    return distance


def filename_gen(fold_index: int):
    global FOLDS_PATH
    return 'ecg_fold_{}.data'.format(fold_index)


def load_k_fold_data(fold_index: int):
    with open(filename_gen(fold_index), 'rb') as f:
        fold_data = pickle.load(f)
    return tuple(zip(*fold_data))


def split_crosscheck_groups(dataset, num_folds):
    train_features, train_labels = dataset

    true_labels_indexes = []
    false_labels_indexes = []
    for index, label in enumerate(train_labels):
        if label:
            true_labels_indexes.append(index)
        else:
            assert not label
            false_labels_indexes.append(index)
        assert index in false_labels_indexes or index in true_labels_indexes

    samples_per_fold = len(train_features) // num_folds
    false_percent = len(false_labels_indexes) / len(train_labels)
    false_samples_per_fold = int(false_percent * samples_per_fold)
    true_samples_per_fold = samples_per_fold - false_samples_per_fold

    indexes_per_fold = []

    for i in range(num_folds - 1):
        # choosing random true features indexes
        samples_true = sample(true_labels_indexes, true_samples_per_fold)
        # choosing random false features indexes
        samples_false = sample(false_labels_indexes, false_samples_per_fold)

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
        fold_data = []
        for feature_index in indexes_in_fold:
            fold_data.append((train_features[feature_index], train_labels[feature_index]))
        pickle.dump(fold_data, open(filename_gen(fold_index + 1), 'wb'))  # save indexes of fold in dumped file


def __get_k_folds_data(num_folds: int, validation_fold_index: int):
    """
    :param num_folds: get number of folds
    :param validation_fold_index: the fold that use as validation
    :return: return the folds in tuples: ((validation_features, validation_labels),(train_features, train_labels))
    """
    validation_tuple = load_k_fold_data(validation_fold_index)

    train_features = []
    train_labels = []
    for train_fold_index in range(1, num_folds + 1):
        if train_fold_index == validation_fold_index:
            continue
        fold_train_features, fold_train_labels = load_k_fold_data(train_fold_index)
        train_features.extend(fold_train_features)
        train_labels.extend(fold_train_labels)

    train_tuple = (train_features, train_labels)

    return validation_tuple, train_tuple


def evaluate(classifier_factory: abstract_classifier_factory, k: int = 2):
    """
    :param classifier_factory: KNN factory
    :param k: number of folds for k − fold cross validation
    :return: accuracy, error
    """
    accuracy_list = []
    error_list = []
    for validation_fold_index in range(1, k + 1):
        validation_tuple, train_tuple = __get_k_folds_data(k, validation_fold_index)

        clf = classifier_factory.train(train_tuple[0], train_tuple[1])

        correct_class = 0
        incorrect_class = 0
        validation_counter = 0
        for valid_feature, valid_label in zip(validation_tuple[0], validation_tuple[1]):
            res_class = clf.classify(valid_feature)
            if res_class == valid_label:
                correct_class += 1
            else:
                incorrect_class += 1
            validation_counter += 1

        assert correct_class + incorrect_class == validation_counter

        fold_accuracy = correct_class / float(validation_counter)
        fold_error = incorrect_class / float(validation_counter)
        accuracy_list.append(fold_accuracy)
        error_list.append(fold_error)

    return sum(accuracy_list) / len(accuracy_list), sum(error_list) / len(error_list)
