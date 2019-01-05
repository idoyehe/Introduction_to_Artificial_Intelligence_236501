from hw3_utils import load_data, abstract_classifier, abstract_classifier_factory
from euclideanDistance import euclidean_distance


class knn_classifier(abstract_classifier):
    def __init__(self, classified_data, labeled_data, k_value: int = 3):
        self.classified_data = classified_data
        self.labeled_data = labeled_data
        self.k_value = k_value

    def classify(self, features):
        distances_list = [(index, euclidean_distance(current_feature, features)) for index, current_feature in
                          enumerate(self.classified_data)]
        sorted_distance_list = sorted(distances_list, key=lambda t: t[1])
        knn_indexes = [t[0] for t in sorted_distance_list[:self.k_value]]

        true_counter = 0
        false_counter = 0

        for i in knn_indexes:
            true_counter += self.labeled_data[i]
            false_counter += not self.labeled_data[i]

        return true_counter > false_counter


class knn_factory(abstract_classifier_factory):
    def __init__(self, k_value: int = 3):
        self.k_value = k_value

    def train(self, data, labels):
        return knn_classifier(classified_data=data, labeled_data=labels, k_value=self.k_value)
