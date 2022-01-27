# -*- coding: utf-8 -*-
"""Label Ranking Algorithms.

Created for 'Label Ranking through Nonparametric Regression'

@author:

"""
from datetime import datetime
from multiprocessing import Pool
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.tree import DecisionTreeRegressor

import numpy as np
import pandas as pd
import signal
import time


def workload(idx_train, idx_test, X, y, y_i, y_t, labels):
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y_t.loc[idx_test]
    y_i_train = y_i.loc[idx_train]
    test_number = len(y_test)
    labels_num = len(labels)

    y_train = [s.replace('>', '') for s in y_train]
    RFregressors = [RandomForestRegressor(n_estimators=50, criterion='mse').
                    fit(X_train, y_i_train[lb]) for lb in labels]
    DTregressors = [DecisionTreeRegressor(criterion='mse').
                    fit(X_train, y_i_train[lb]) for lb in labels]
    DTSregressors = [DecisionTreeRegressor(criterion='mse', max_depth=15).
                     fit(X_train, y_i_train[lb]) for lb in labels]

    y_i_pred_RF = [[] for lb in labels]
    y_i_pred_DT = [[] for lb in labels]
    y_i_pred_DTS = [[] for lb in labels]
    y_pred_RF = []
    y_pred_DT = []
    y_pred_DTS = []

    for i in range(labels_num):
        y_i_pred_RF[i] = RFregressors[i].predict(X_test)
        y_i_pred_DT[i] = DTregressors[i].predict(X_test)
        y_i_pred_DTS[i] = DTSregressors[i].predict(X_test)
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
    for i in range(test_number):
        test = []
        for j in range(labels_num):
            test.append((labels[j], y_i_pred_DTS[j][i]))
        test = sorted(test, key=lambda sub: (sub[1]))
        pred = [t[0] for t in test]
        y_pred_DTS.append('>'.join(pred))

    L_kendall_tau_coeff_RF = [kendalltau(pred.split('>'),
                                         real.split('>')).correlation
                              for pred, real in zip(y_pred_RF, y_test)]
    mean_kendall_tau_coeff_RF = np.mean(L_kendall_tau_coeff_RF)

    L_kendall_tau_coeff_DT = [kendalltau(pred.split('>'),
                                         real.split('>')).correlation
                              for pred, real in zip(y_pred_DT, y_test)]
    mean_kendall_tau_coeff_DT = np.mean(L_kendall_tau_coeff_DT)

    L_kendall_tau_coeff_DTS = [kendalltau(pred.split('>'),
                                          real.split('>')).correlation
                               for pred, real in zip(y_pred_DTS, y_test)]
    mean_kendall_tau_coeff_DTS = np.mean(L_kendall_tau_coeff_DTS)

    return (mean_kendall_tau_coeff_RF,
            mean_kendall_tau_coeff_DT,
            mean_kendall_tau_coeff_DTS)


def main(processes=5):
    """
    Get results For all data sets.

    Parameters
    ----------
    processes : int, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    None.

    """
    now = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
    print(now)

    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = Pool(processes)

    random_state = 42
    datasets_choice = 'semi-synthetic'
    base_data_path = ['LR_DATASETS']
    if datasets_choice == 'semi-synthetic':
        dataset_grid = ['authorship', 'bodyfat', 'calhousing', 'cpu-small',
                        'elevators', 'fried', 'glass', 'housing', 'iris',
                        'pendigits', 'segment', 'stock', 'vehicle', 'vowel',
                        'wine', 'wisconsin']
    elif datasets_choice == 'real':
        dataset_grid = ['cold', 'diau', 'dtt', 'heat', 'spo']
    else:
        dataset_grid = [datasets_choice]

    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        # Choose dataset
        for base in base_data_path:
            f = open(base + '_' + datasets_choice + '_results.txt', 'a')
            f.write('Benchmark & RF & DT & DTS \\\\\n')
            f.close()

            for dataset_choice in dataset_grid:
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
                L_results_kendall_coeff_DTS = []

                n = len(dataset)

                print("Starting {} jobs".format(processes))
                multiple_results = [pool.apply_async(
                    workload, (idx_train, idx_test, X, y, y_i, y_t, labels))
                    for idx_train, idx_test in rkf.split(range(n))]
                print("Waiting for results...")
                for res in multiple_results:
                    L_results_kendall_coeff_RF.append(res.get()[0])
                    L_results_kendall_coeff_DT.append(res.get()[1])
                    L_results_kendall_coeff_DTS.append(res.get()[2])

                f = open(base + '_' + datasets_choice + '_results.txt', 'a')
                f.write(dataset_choice + '&'
                        + str(np.round(np.mean(L_results_kendall_coeff_RF), 3))
                        + '\u00B1'
                        + str(np.round(np.std(L_results_kendall_coeff_RF), 3))
                        + '&'
                        + str(np.round(np.mean(L_results_kendall_coeff_DT), 3))
                        + '\u00B1'
                        + str(np.round(np.std(L_results_kendall_coeff_DT), 3))
                        + '&'
                        + str(np.round(np.mean(L_results_kendall_coeff_DTS), 3))
                        + '\u00B1'
                        + str(np.round(np.std(L_results_kendall_coeff_DTS), 3))
                        + '\\\\ \n')
                f.close()
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    else:
        print("Normal termination")
        pool.close()
    pool.join()


if __name__ == "__main__":
    start_time = time.perf_counter()
    main(10)
    end_time = time.perf_counter()
    print('Duration: {:.2f} min'.format((end_time - start_time)/60))
