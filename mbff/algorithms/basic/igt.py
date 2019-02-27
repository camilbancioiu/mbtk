import operator

import mbff.infotheory as infotheory

def algorithm_IGt__binary(datasetmatrix, parameters):
    (sample_count, feature_count) = datasetmatrix.X.get_shape()
    Q = parameters['Q']
    objective_vector = datasetmatrix.get_column_Y(parameters['objective_index'])
    IG_per_feature = []
    for feature_index in range(feature_count):
        feature_vector = datasetmatrix.get_column_X(feature_index)
        feature_IG = infotheory.MI__binary(feature_vector, objective_vector)
        IG_per_feature.append((feature_index, feature_IG))

    sorted_IG_per_feature = sorted(IG_per_feature, key=operator.itemgetter(1), reverse=True)
    return [pair[0] for pair in sorted_IG_per_feature[0:Q]]
