from hw3_utils import abstract_classifier_factory, load_data
from functionUtils import load_k_fold_data, split_crosscheck_groups
from classifier import knn_factory


def __get_k_folds_data(num_folds: int):
    """
    :param num_folds: get number of folds
    :return: return the folds in list of tuples: [...((validation_features, validation_labels),(train_features, train_labels))...]
    """
    k_fold_data = list()

    for valid_fold_index in range(1, num_folds + 1):
        validation_tuple = load_k_fold_data(valid_fold_index)

        fold_train_features = []
        fold_train_labels = []
        for train_fold_index in range(1, num_folds + 1):
            if train_fold_index == valid_fold_index:
                continue
            train_features, train_labels = load_k_fold_data(train_fold_index)
            fold_train_features.extend(train_features)
            fold_train_labels.extend(train_labels)

        train_tuple = (fold_train_features, fold_train_labels)
        k_fold_data.append((validation_tuple, train_tuple))

    return k_fold_data


def evaluate(classifier_factory: abstract_classifier_factory, k: int):
    """
    :param classifier_factory: KNN factory
    :param k: number of folds for k − fold cross validation
    :return: accuracy, error
    """

    k_fold_data = __get_k_folds_data(k)
    assert len(k_fold_data) == k  # TODO:remove before submission

    validation_counter = 0

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for curr_fold in k_fold_data:
        knn = classifier_factory.train(curr_fold[1][0], curr_fold[1][1])

        for valid_index, valid_feature in enumerate(curr_fold[0][0]):
            res_class = knn.classify(valid_feature)
            if res_class:
                if curr_fold[0][1][valid_index]:
                    assert res_class == curr_fold[0][1][valid_index]
                    true_pos += 1
                else:
                    assert res_class != curr_fold[0][1][valid_index]
                    false_pos += 1
            else:
                assert not res_class
                if not curr_fold[0][1][valid_index]:
                    assert res_class == curr_fold[0][1][valid_index]
                    true_neg += 1
                else:
                    assert res_class != curr_fold[0][1][valid_index]
                    false_neg += 1

            validation_counter += 1

    accuracy = (true_pos + true_neg) / float(validation_counter)
    error = (false_pos + false_neg) / float(validation_counter)
    return accuracy, error


""" single call to fold spilt"""
train_features_ds, train_labels_ds, test_features_ds = load_data()
split_crosscheck_groups((train_features_ds, train_labels_ds), 2)

for k in [1, 3, 5, 7, 13]:
    knn = knn_factory(k)
    accuracy, error = evaluate(knn, 2)
    output = str(k) + "," + str(accuracy) + "," + str(error)
    print(output)
