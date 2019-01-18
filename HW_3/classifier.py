from hw3_utils import abstract_classifier, abstract_classifier_factory
from functionUtils import euclidean_distance
from sklearn import tree, linear_model, preprocessing, neighbors


class knn_classifier(abstract_classifier):
    def __init__(self, classified_data, labeled_data, k_value: int = 3):
        self.classified_data = classified_data
        self.labeled_data = labeled_data
        self.k_value = k_value

    def classify(self, object_features):
        distances_list = [(self.labeled_data[index], euclidean_distance(current_feature, object_features)) for
                          index, current_feature in
                          enumerate(self.classified_data)]
        sorted_distance_list = sorted(distances_list, key=lambda t: t[1])
        knn_classify = [t[0] for t in sorted_distance_list[:self.k_value]]

        true_counter = sum(knn_classify)
        false_counter = len(knn_classify) - true_counter

        return true_counter > false_counter


class knn_factory(abstract_classifier_factory):
    def __init__(self, k_value: int = 3):
        self.k_value = k_value

    def train(self, data, labels):
        return knn_classifier(classified_data=data, labeled_data=labels, k_value=self.k_value)


class id3_classifier(abstract_classifier):
    def __init__(self, classified_data, labeled_data):
        self.clf = tree.DecisionTreeClassifier(criterion="entropy")
        self.clf.fit(classified_data, labeled_data)

    def classify(self, object_features):
        return self.clf.predict([object_features])[0]


class id3_factory(abstract_classifier_factory):
    def train(self, data, labels):
        return id3_classifier(classified_data=data, labeled_data=labels)


class perceptron_classifier(abstract_classifier):
    def __init__(self, classified_data, labeled_data):
        self.clf = linear_model.Perceptron()
        self.clf.fit(classified_data, labeled_data)

    def classify(self, object_features):
        return self.clf.predict([object_features])[0]


class perceptron_factory(abstract_classifier_factory):
    def train(self, data, labels):
        return perceptron_classifier(classified_data=data, labeled_data=labels)


class contest_classifier(abstract_classifier):
    def __init__(self, classified_data, labeled_data, hyperparameters_dict=None):
        # data is normalized by z normalization
        self.scale = preprocessing.StandardScaler().fit(classified_data)
        norm_data = self.scale.transform(classified_data)

        #   postprocessing of important features;
        #   hyperparameters were chose by hyperparameters optimisation random search;

        if hyperparameters_dict is None:
            hyperparameters_dict = {'criterion': 'entropy',
                                    'min_samples_split': 3,
                                    'min_samples_leaf': 1,
                                    'max_depth': 17,
                                    'max_leaf_nodes': 183,
                                    'max_features': 'auto'}

        assert hyperparameters_dict is not None

        self.id3 = tree.DecisionTreeClassifier(**hyperparameters_dict)
        self.id3.fit(classified_data, labeled_data)
        self.features_weights = self.id3.feature_importances_

        self.knn_1 = neighbors.KNeighborsClassifier(n_neighbors=1)
        self.knn_1.fit(norm_data * self.features_weights, labeled_data)

        self.knn_3 = neighbors.KNeighborsClassifier(n_neighbors=3)
        self.knn_3.fit(norm_data * self.features_weights, labeled_data)

        self.knn_5 = neighbors.KNeighborsClassifier(n_neighbors=5)
        self.knn_5.fit(norm_data * self.features_weights, labeled_data)

        self.total_class = 3

    def classify(self, object_features):
        norm_object = self.scale.transform([object_features]) * self.features_weights
        results = {"KNN_1": self.knn_1.predict(norm_object)[0],
                   "KNN_3": self.knn_3.predict(norm_object)[0],
                   "KNN_5": self.knn_5.predict(norm_object)[0]}

        call = sum(results.values())
        return call > self.total_class - call


class contest_classifier_factory(abstract_classifier_factory):
    def __init__(self, hyperparameters_dict=None):
        self.hyperparameters_dict = hyperparameters_dict

    def train(self, data, labels):
        return contest_classifier(classified_data=data, labeled_data=labels, hyperparameters_dict=self.hyperparameters_dict)
