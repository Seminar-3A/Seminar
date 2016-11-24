import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


def fitting_forest(data_subdivided, n_estimators=100):

    print('Fitting the random forest for the {}'.format(set))

    X_train = np.array(data_subdivided['X_train'])
    X_test = np.array(data_subdivided['X_test'])
    Y_train = np.array(data_subdivided['Y_train'])
    Y_test = np.array(data_subdivided['Y_test'])

    X_train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)
    forest = RandomForestClassifier(n_estimators=n_estimators, max_features="auto")
    fit_forest = forest.fit(X_train_scaled, Y_train)

    score = forest.score(X_test_scaled, Y_test)
    print('Score error: {}'.format(score))
    prediction = forest.predict(X_test_scaled)

    return fit_forest, score, prediction
