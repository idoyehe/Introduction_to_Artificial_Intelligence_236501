from hw3_utils import abstract_classifier, abstract_classifier_factory
from functionUtils import euclidean_distance
from sklearn import tree, linear_model, neural_network, preprocessing, neighbors


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
    def __init__(self, classified_data, labeled_data):
        self.scaler = preprocessing.StandardScaler().fit(classified_data)
        norm_data = self.scaler.transform(classified_data)

        #   hyperparameters were chosen by hyperparameters optimization
        self.mlf = neural_network.MLPClassifier(activation='relu',solver='sgd',learning_rate_init=0.004,max_iter=450,n_iter_no_change=20)
        self.mlf.fit(norm_data, labeled_data)

        #   hyperparameters were chosen by hyperparameters optimization
        self.id3 = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=8, min_samples_leaf=6)
        self.id3.fit(classified_data, labeled_data)

        #   hyperparameters were chosen by first section
        self.knn = neighbors.KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(norm_data, labeled_data)

        self.total_class = 3

    def classify(self, object_features):
        norm_object = self.scaler.transform([object_features])
        results = {"nlf": self.mlf.predict(norm_object)[0],
                   "id3": self.id3.predict([object_features])[0],
                   "knn": self.knn.predict(norm_object)[0]}

        call = sum(results.values())
        return call > self.total_class - call


class contest_factory(abstract_classifier_factory):
    def train(self, data, labels):
        return contest_classifier(classified_data=data, labeled_data=labels)
