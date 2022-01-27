# -*- coding: utf-8 -*-
"""Label Ranking Algorithms.

Created for 'Label Ranking through Nonparametric Regression'

@author:

"""
from datetime import datetime
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
import time


def workload(idx_train, idx_test, X, y, y_i, y_t, labels):
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y_t.loc[idx_test]
    y_i_train = y_i.loc[idx_train]
    test_number = len(y_test)
    labels_num = len(labels)

    y_train = [s.replace('>', '') for s in y_train]

    dt = DecisionTreeRegressor()
    rf = RandomForestRegressor()
    parameters_dt = {'max_features': ["auto", "sqrt", "log2"],
                     'max_depth': [5, 10, 15, None]
                     }

    parameters_rf = {'n_estimators': [50],
                     'max_features': ["auto", "sqrt", "log2"],
                     'max_depth': [5, 10, 15, None]
                     }

    RFregressors = [GridSearchCV(estimator=rf, param_grid=parameters_rf,
                                 cv=5, n_jobs=-1)
                    .fit(X_train, y_i_train[lb]).best_estimator_
                    for lb in labels]
    DTregressors = [GridSearchCV(estimator=dt, param_grid=parameters_dt,
                                 cv=5, n_jobs=1)
                    .fit(X_train, y_i_train[lb]).best_estimator_
                    for lb in labels for lb in labels]

    y_i_pred_RF = [[] for lb in labels]
    y_i_pred_DT = [[] for lb in labels]
    y_pred_RF = []
    y_pred_DT = []

    for i in range(labels_num):
        y_i_pred_RF[i] = RFregressors[i].predict(X_test)
        y_i_pred_DT[i] = DTregressors[i].predict(X_test)
    for i in range(test_number):
        test = []
        for j in range(labels_num):
            test.append((labels[j], y_i_pred_RF[j][i]))
        test = sorted(test, key=lambda sub: (sub[1]))
        pred = [t[0] for t in test]
        y_pred_RF.append('>'.join(pred))
    for i in range(test_number):
        test = []
        for j in range(labels_num):
            test.append((labels[j], y_i_pred_DT[j][i]))
        test = sorted(test, key=lambda sub: (sub[1]))
        pred = [t[0] for t in test]
        y_pred_DT.append('>'.join(pred))

    L_kendall_tau_coeff_RF = [kendalltau(pred.split('>'),
                                         real.split('>')).correlation
                              for pred, real in zip(y_pred_RF, y_test)]
    mean_kendall_tau_coeff_RF = np.mean(L_kendall_tau_coeff_RF)

    L_kendall_tau_coeff_DT = [kendalltau(pred.split('>'),
                                         real.split('>')).correlation
                              for pred, real in zip(y_pred_DT, y_test)]
    mean_kendall_tau_coeff_DT = np.mean(L_kendall_tau_coeff_DT)

    return (mean_kendall_tau_coeff_RF, mean_kendall_tau_coeff_DT)


def main():
    """
    Get results For all data sets.

    Returns
    -------
    None.

    """
    now = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
    print(now)

    random_state = 42
    datasets_choice = 'semi-synthetic'
    base_data_path = ['LR_DATASETS']
    if datasets_choice == 'semi-synthetic':
        dataset_grid = ['authorship', 'bodyfat', 'calhousing', 'cpu-small',
                        'elevators', 'fried', 'glass', 'iris', 'pendigits',
                        'segment', 'stock', 'vehicle', 'vowel', 'wine',
                        'wisconsin']
    elif datasets_choice == 'real':
        dataset_grid = ['cold', 'diau', 'dtt', 'heat', 'spo']
    else:
        dataset_grid = [datasets_choice]

    # Choose dataset
    for base in base_data_path:
        f = open(base + '_' + datasets_choice + '_results_grid.txt', 'a')
        f.write('Benchmark & RFT & DTT \\\\ \n')
        f.close()

        for dataset_choice in dataset_grid:
            start_time = time.perf_counter()
            rkf = RepeatedKFold(n_splits=10, n_repeats=5,
                                random_state=random_state)
            dataset_path = '../datasets/' + base + '/' + \
                dataset_choice+'.txt'

            dataset = pd.read_csv(dataset_path)

            labels = dataset['ranking'][0]
            labels = sorted(labels.split('>'))
            first = labels[0]
            last = labels[-1]

            X = dataset.drop(
                dataset.loc[:, 'ranking':last].columns,
                axis=1)

            y = dataset['ranking']
            y_t = dataset['ranking']

            y_i = dataset[dataset.loc[:, first:last].columns]

            L_results_kendall_coeff_RF = []
            L_results_kendall_coeff_DT = []

            n = len(dataset)

            print("Starting jobs")
            for idx_train, idx_test in rkf.split(range(n)):
                res = workload(idx_train, idx_test, X, y, y_i, y_t, labels)
                L_results_kendall_coeff_RF.append(res[0])
                L_results_kendall_coeff_DT.append(res[1])
            print("Waiting for results...")

            f = open(base + '_' + datasets_choice + '_results_grid.txt', 'a')
            f.write(dataset_choice+'&'
                    + str(np.round(np.mean(L_results_kendall_coeff_RF), 3))
                    + '\u00B1'
                    + str(np.round(np.std(L_results_kendall_coeff_RF), 3))
                    + '&'
                    + str(np.round(np.mean(L_results_kendall_coeff_DT), 3))
                    + '\u00B1'
                    + str(np.round(np.std(L_results_kendall_coeff_DT), 3))
                    + '\\\\ \n')
            f.close()
            end_time = time.perf_counter()
            print(dataset_choice
                  + ' Duration: {:.2f} min'.format((end_time - start_time)/60))


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print('Total Duration: {:.2f} min'.format((end_time - start_time)/60))
