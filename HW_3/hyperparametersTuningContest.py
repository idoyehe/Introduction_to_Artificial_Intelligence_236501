from hw3_utils import abstract_classifier, abstract_classifier_factory
from sklearn import neural_network, preprocessing, tree
from numpy import linspace

from functionUtils import evaluate, split_crosscheck_groups
from hw3_utils import load_data

""" single call to fold spilt"""

train_features_ds, train_labels_ds, test_features_ds = load_data()
split_crosscheck_groups((train_features_ds, train_labels_ds), 4)


def hyperparameters_neural_network_tuning():
    activation_list = ['relu', 'tanh', 'logistic']
    solver_list = ['sgd', 'adam']
    lr_list = linspace(0.001, 0.01, 5)
    max_iter_list = linspace(400, 600, 5)
    n_iter_list = linspace(15, 35, 5)
    output_list = []

    for curr_act in activation_list:
        for curr_solver in solver_list:
            for curr_lr in lr_list:
                for curr_max_iter in max_iter_list:
                    for curr_n_iter in n_iter_list:
                        hyperparameters_dict = {
                            "activation": curr_act,
                            "solver": curr_solver,
                            "learning_rate_init": curr_lr,
                            "max_iter": int(curr_max_iter),
                            "n_iter_no_change": int(curr_n_iter)}
                        avg_accuracy = 0
                        avg_error = 0
                        for _ in range(5):
                            mlp = neural_network_factory(hyperparameters_dict)
                            res_accuracy, res_error = evaluate(mlp, 4)
                            avg_accuracy += res_accuracy
                            avg_error += res_error

                        avg_accuracy /= 5
                        avg_error /= 5
                        output = str(hyperparameters_dict) + "," + str(avg_accuracy) + "," + str(avg_error)
                        output_list.append((output, avg_accuracy))
                        print(output)

    print(max(output_list, key=lambda t: t[1]))


def hyperparameters_id3_tuning():
    criterion_list = ['gini', 'entropy']
    min_samples_split_list = linspace(2, 10, 5)
    min_samples_leaf_list = linspace(1, 10, 10)
    output_list = []

    for curr_criterion in criterion_list:
        for curr_min_samples_split in min_samples_split_list:
            for curr_min_samples_leaf in min_samples_leaf_list:
                hyperparameters_dict = {
                    "criterion": curr_criterion,
                    "min_samples_split": int(curr_min_samples_split),
                    "min_samples_leaf": int(curr_min_samples_leaf)}
                avg_accuracy = 0
                avg_error = 0
                for _ in range(10):
                    id3 = id3_factory(hyperparameters_dict)
                    res_accuracy, res_error = evaluate(id3, 4)
                    avg_accuracy += res_accuracy
                    avg_error += res_error

                avg_accuracy /= 10
                avg_error /= 10
                output = str(hyperparameters_dict) + "," + str(avg_accuracy) + "," + str(avg_error)
                output_list.append((output, avg_accuracy))
                print(output)

    print(max(output_list, key=lambda t: t[1]))






class neural_network_classifier(abstract_classifier):
    def __init__(self, classified_data, labeled_data, hyperparameters_dict):
        self.scaler = preprocessing.StandardScaler().fit(classified_data)
        norm_data = self.scaler.transform(classified_data)
        self.clf = neural_network.MLPClassifier(**hyperparameters_dict)
        self.clf.fit(norm_data, labeled_data)

    def classify(self, object_features):
        norm_object = self.scaler.transform([object_features])
        return self.clf.predict(norm_object)[0]


class neural_network_factory(abstract_classifier_factory):
    def __init__(self, parameters_dict):
        self.parameters_dict = parameters_dict

    def train(self, data, labels):
        return neural_network_classifier(classified_data=data, labeled_data=labels, hyperparameters_dict=self.parameters_dict)

class id3_classifier(abstract_classifier):
    def __init__(self, classified_data, labeled_data, hyperparameters_dict):
        self.scaler = preprocessing.StandardScaler().fit(classified_data)
        norm_data = self.scaler.transform(classified_data)
        self.clf = tree.DecisionTreeClassifier(**hyperparameters_dict)
        self.clf.fit(norm_data, labeled_data)

    def classify(self, object_features):
        return self.clf.predict([object_features])[0]


class id3_factory(abstract_classifier_factory):
    def __init__(self, parameters_dict):
        self.parameters_dict = parameters_dict

    def train(self, data, labels):
        return id3_classifier(classified_data=data, labeled_data=labels, hyperparameters_dict=self.parameters_dict)

hyperparameters_neural_network_tuning()
hyperparameters_id3_tuning()