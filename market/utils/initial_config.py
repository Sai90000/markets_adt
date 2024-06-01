import numpy as np
from sklearn import preprocessing
from scipy.spatial import distance
import pandas as pd
from joblib import dump, load

def initial_configuration(data, output_path=None, normalize=False, train=True, config=None):
    assert (type(data) == np.ndarray) or (
                type(data) == pd.core.frame.DataFrame), "Data type is not numpy array or Pandas data frame"

    if type(data) == np.ndarray:
        X = data.copy()
    elif type(data) == pd.core.frame.DataFrame:
        X = data.values

    if train:
        if normalize:
            standard_scaler = preprocessing.StandardScaler()
            scaled_features = X[:, :-1].copy()
            standard_scaler = standard_scaler.fit(scaled_features)
            X[:, :-1] = standard_scaler.transform(scaled_features)
            
            dump(standard_scaler, config.intermediate_file_location + 'scaler.gz', compress='gzip')


        ind_class0 = np.where(X[:, -1] < 0.5)[0]
        ind_class1 = np.where(X[:, -1] >= 0.5)[0]

        X_0 = X[ind_class0, :-1]
        X_1 = X[ind_class1, :-1]

        # distance -> rows denote distances for X_0 and columns denote distances for X_1
        distance_matrix = distance.cdist(X_0, X_1)

        # Add new-columns to hold the minimum and maximum radius
        X = np.append(X, np.zeros(X.shape[0]).reshape(-1, 1), axis=1)
        X = np.append(X, np.zeros(X.shape[0]).reshape(-1, 1), axis=1)

        # fill up the maximum radius column
        # For every datapoint, find the nearest point from other class
        X[ind_class0, -1] = np.min(distance_matrix, axis=1)
        X[ind_class1, -1] = np.min(distance_matrix, axis=0)
        # free-up the space
        del distance_matrix

        # compute pairwise-distances between X_0 and X_1
        distance_X_0 = distance.squareform(distance.pdist(X_0))
        distance_X_1 = distance.squareform(distance.pdist(X_1))

        # fill up the minimum radius column
        # for every datapoint, find the min(nearest point from same class, nearest point from other class/2)
        for i in range(ind_class0.shape[0]):
            X[ind_class0[i], -2] = np.minimum(np.min(distance_X_0[i][np.nonzero(distance_X_0[i])]),
                                              X[ind_class0[i], -1] / 2)

        # free-up space
        del distance_X_0

        for i in range(ind_class1.shape[0]):
            X[ind_class1[i], -2] = np.minimum(np.min(distance_X_1[i][np.nonzero(distance_X_1[i])]),
                                              X[ind_class1[i], -1] / 2)

        # free-up space
        del distance_X_1, ind_class0, ind_class1

        # if min radius is zero or max radius is zero ; handle it with minimum-value/2
        min_of_max_radius = np.min(X[np.nonzero(X[:, -1]), -1]) / 2
        min_of_min_radius = np.min(X[np.nonzero(X[:, -2]), -2]) / 2
        ind = np.where(X[:, -1] == 0.0)[0]
        X[ind, -1] = min_of_max_radius
        ind = np.where(X[:, -2] == 0.0)[0]
        X[ind, -2] = min_of_min_radius
    else:
        if normalize:

            standard_scaler = load(config.intermediate_file_location + 'scaler.gz')
            X = standard_scaler.transform(X.copy())

    if output_path:
        np.savetxt(output_path, X, delimiter=",")

    return X
