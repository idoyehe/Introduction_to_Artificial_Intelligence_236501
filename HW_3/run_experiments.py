from functionUtils import split_crosscheck_groups, evaluate
from classifier import knn_factory
from hw3_utils import load_data

""" single call to fold spilt"""


# train_features_ds, train_labels_ds, test_features_ds = load_data()
# split_crosscheck_groups((train_features_ds, train_labels_ds), 2)

def experiment_6():
    knn_values_list = [1, 3, 5, 7, 13]
    for k in knn_values_list:
        knn_k = knn_factory(k)
        res_accuracy, res_error = evaluate(knn_k, 2)
        output = str(k) + "," + str(res_accuracy) + "," + str(res_error)
        print(output)


def experiment_7A():
    pass


if __name__ == '__main__':
    experiment_6()
    experiment_7A()
