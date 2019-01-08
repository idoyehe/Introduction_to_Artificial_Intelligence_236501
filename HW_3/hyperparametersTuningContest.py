from hw3_utils import abstract_classifier, abstract_classifier_factory
from sklearn import neural_network, preprocessing
from numpy import linspace

from functionUtils import evaluate, split_crosscheck_groups
from hw3_utils import load_data

""" single call to fold spilt"""

train_features_ds, train_labels_ds, test_features_ds = load_data()
split_crosscheck_groups((train_features_ds, train_labels_ds), 4)


def hyperparameters_tuning():
    activation_list = ['relu', 'tanh', 'logistic']
    solver_list = ['sgd', 'adam']
    lr_list = linspace(0.0001, 0.01, 100)
    max_iter_list = linspace(200, 550, 8)
    n_iter_list = linspace(5, 50, 10)
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
                        for _ in range(10):
                            mlp = contest_factory(hyperparameters_dict)
                            res_accuracy, res_error, clf = evaluate(mlp, 4)
                            avg_accuracy += res_accuracy
                            avg_error += res_error

                        avg_accuracy /= 10
                        avg_error /= 10
                        output = str(hyperparameters_dict) + "," + str(avg_accuracy) + "," + str(avg_error)
                        output_list.append((output, avg_accuracy))
                        print(output)

    sorted_output_list = sorted(output_list, key=lambda t: t[1])
    print(sorted_output_list)


class contest_classifier(abstract_classifier):
    def __init__(self, classified_data, labeled_data, hyperparameters_dict):
        self.scaler = preprocessing.StandardScaler().fit(classified_data)
        norm_data = self.scaler.transform(classified_data)
        self.clf = neural_network.MLPClassifier(**hyperparameters_dict)
        self.clf.fit(norm_data, labeled_data)

    def classify(self, object_features):
        norm_object = self.scaler.transform([object_features])
        return self.clf.predict(norm_object)[0]


class contest_factory(abstract_classifier_factory):
    def __init__(self, parameters_dict):
        self.parameters_dict = parameters_dict

    def train(self, data, labels):
        return contest_classifier(classified_data=data, labeled_data=labels, hyperparameters_dict=self.parameters_dict)


hyperparameters_tuning()
