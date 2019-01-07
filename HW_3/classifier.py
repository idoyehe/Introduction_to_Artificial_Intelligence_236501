from hw3_utils import abstract_classifier, abstract_classifier_factory
from functionUtils import euclidean_distance
from sklearn import tree, linear_model


class knn_classifier(abstract_classifier):
    def __init__(self, classified_data, labeled_data, k_value: int = 3):
        self.classified_data = classified_data
        self.labeled_data = labeled_data
        self.k_value = k_value

    def classify(self, object_features):
        distances_list = [(self.labeled_data[index], euclidean_distance(current_feature, object_features)) for index, current_feature in
                          enumerate(self.classified_data)]
        sorted_distance_list = sorted(distances_list, key=lambda t: t[1])
        knn_classify = [t[0] for t in sorted_distance_list[:self.k_value]]

        assert len(knn_classify) == self.k_value  # TODO:remove before submission
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
        self.clf = tree.DecisionTreeClassifier()
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
