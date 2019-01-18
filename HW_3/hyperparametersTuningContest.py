from random import randint
from functionUtils import evaluate, split_crosscheck_groups
from hw3_utils import load_data
from classifier import contest_classifier_factory

""" single call to fold spilt"""
FOLDS = 4
train_features_ds, train_labels_ds, test_features_ds = load_data()
split_crosscheck_groups((train_features_ds, train_labels_ds), FOLDS)
file = open('./contest', 'w')


def hyperparameters_id3_tuning():
    output_list = []
    for _ in range(1000):
        hyperparameters_dict = {
            "criterion": 'entropy',
            "min_samples_split": randint(2, 12),
            "min_samples_leaf": randint(1, 12),
            'max_depth': randint(10, 40),
            'max_leaf_nodes': randint(20, 200),
            'max_features': 'auto'}
        avg_accuracy = 0
        avg_error = 0
        tries = 3
        for _ in range(tries):
            clc = contest_classifier_factory(hyperparameters_dict)
            res_accuracy, res_error = evaluate(clc, FOLDS)
            avg_accuracy += res_accuracy
            avg_error += res_error

        avg_accuracy /= tries
        avg_error /= tries
        output = str(hyperparameters_dict) + "," + str(avg_accuracy) + "," + str(avg_error)
        output_list.append((output, avg_accuracy))
        # file.writelines(output + "\n")
        print(output)
    file.writelines("Best Hyperparametrs set:\n")
    file.writelines(str(max(output_list, key=lambda t: t[1])) + "\n")
    print(str(max(output_list, key=lambda t: t[1])))


file.writelines("\n")

hyperparameters_id3_tuning()
file.close()
