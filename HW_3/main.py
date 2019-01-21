from functionUtils import evaluate, split_crosscheck_groups
from classifier import knn_factory, id3_factory, perceptron_factory, contest_classifier_factory
from hw3_utils import load_data, write_prediction

def experiment_6():
    knn_values_list = [1, 3, 5, 7, 13]
    for k in knn_values_list:
        knn_k = knn_factory(k)
        res_accuracy, res_error = evaluate(knn_k, FOLDS)
        output = str(k) + "," + str(res_accuracy) + "," + str(res_error)
        print(output)


def experiment_7A():
    id3 = id3_factory()
    res_accuracy, res_error = evaluate(id3, FOLDS)
    output = str(1) + "," + str(res_accuracy) + "," + str(res_error)
    print(output)


def experiment_7B():
    perceptron = perceptron_factory()
    res_accuracy, res_error = evaluate(perceptron, FOLDS)
    output = str(2) + "," + str(res_accuracy) + "," + str(res_error)
    print(output)


def experiment_contest():
    clc = contest_classifier_factory()
    """ in order to evaluate classifier with existing Folds"""
    # res_accuracy, res_error = evaluate(clc, FOLDS)
    # output = str("contest") + "," + str(res_accuracy) + "," + str(res_error)
    # print(output)

    clf = clc.train(train_features_ds, train_labels_ds)
    test_class_list = []
    for object_feature in test_features_ds:
        test_class_list.append(clf.classify(object_feature))

    write_prediction(test_class_list)


if __name__ == '__main__':
    """ single call to fold spilt"""
    train_features_ds, train_labels_ds, test_features_ds = load_data()
    FOLDS = 2
    '''remove '#' from the next row for new folding'''
    # split_crosscheck_groups((train_features_ds, train_labels_ds), FOLDS)
    experiment_6()
    experiment_7A()
    experiment_7B()
    experiment_contest()
