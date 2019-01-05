def euclidean_distance(feature_list1, feature_list2):
    assert (len(feature_list1) == len(feature_list2))
    distance = 0
    for i in range(len(feature_list1)):
        distance += (feature_list1[i] - feature_list2[i]) ** 2

    return distance ** 0.5
