# -*- coding: utf-8 -*-
"""Noiseless Datasets Generator.

Generates noiseless data sets.

Created for 'Label Ranking through Nonparametric Regression'

@author:
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as util_shuffle

import numpy as np
import pandas as pd


def make_regression(n_samples, n_features, n_informative,
                    n_targets, shuffle=True,
                    random_state=None):
    """Generate a random regression problem.

    Parameters
    ----------
    n_samples : int
        The number of samples.

    n_features : int
        The number of features.

    n_informative : int
        The number of informative features, i.e., the number of features used
        to build the linear model used to generate the output.

    n_targets : int
        The number of regression targets.

    shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.


    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The input samples.

    y : array of shape [n_samples] or [n_samples, n_targets]
        The output values.
    """
    n_informative = min(n_features, n_informative)
    generator = check_random_state(random_state)

    X = generator.randint(2, size=(n_samples, n_features))

    y = np.zeros(shape=(n_targets, n_samples))
    for i in range(n_targets):
        ground_truth = np.zeros((n_features, 1))
        for j in range(n_informative):
            ground_truth[generator.
                         randint(0, high=n_features)] = 100 * generator.rand()

        y_i = np.dot(X, ground_truth)
        scaler = MinMaxScaler((0.25, 0.75)).fit(y_i)
        y_i = scaler.transform(y_i)

        y[i] = y_i.T[0]

    y = np.transpose(y)

    # Randomly permute samples and features
    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]
        ground_truth = ground_truth[indices]

    y = np.squeeze(y)

    return X, y


base_data_path = ['SFN', 'LFN']
name = '_noiseless'

for base in base_data_path:
    if base == 'SFN':
        n_features = 100
    elif base == 'LFN':
        n_features = 1000
    n_targets = 5
    n_samples = 10000
    n_informative = 10

    out = make_regression(n_samples, n_features, n_informative, n_targets,
                          shuffle=True, random_state=None)

    feature_names = ['X_' + str(i) for i in range(n_features)]
    df_X = pd.DataFrame(data=out[0], columns=feature_names)

    original_target_names = ['or_y_' + str(i) for i in range(n_targets)]
    target_names = ['y_' + str(i) for i in range(n_targets)]
    df_y = pd.DataFrame(data=out[1], columns=original_target_names)

    rankings = []
    for index, row in df_y.iterrows():
        labels = []
        for t in target_names:
            labels.append((t, row['or_'+t]))
        labels = sorted(labels, key=lambda sub: (sub[1]), reverse=True)
        pred = [lb[0] for lb in labels]
        rankings.append('>'.join(pred))

    df_r = pd.DataFrame(data=rankings, columns=['original_ranking'])
    df_n = pd.DataFrame(data=rankings, columns=['noisy_ranking'])

    dataset = pd.concat([df_X, df_r, df_y, df_n], axis=1)
    dataset.to_csv('../datasets/'+base+'/'+base+name+'.csv', index=False)
