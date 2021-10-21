# -*- coding: utf-8 -*-
"""Noisy Datasets Generator.

Generates noisy data sets from a noiseless one.

Created for 'Label Ranking through Nonparametric Regression'

@author:
"""
from MallowsModelGenerator import MallowsModelGenerator
from scipy.stats import truncnorm

import pandas as pd


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    """
    Return generator of trancated zero mean normal distribution.

    Parameters
    ----------
    mean : TYPE, optional
        DESCRIPTION. The default is 0.
    sd : TYPE, optional
        DESCRIPTION. The default is 1.
    low : TYPE, optional
        DESCRIPTION. The default is 0.
    upp : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    TYPE
        generator of trancated zero mean normal distribution.

    """
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def generate_noisy_g(noiseless, base, name, std):
    """
    Generate the noisey data sets for Gaussian noise.

    Parameters
    ----------
    noiseless : string
        path of noiseless dataset.
    base : string
        dataset family.
    name : string
        type of noisy dataset.
    std : array
        standard deviation for Gaussian truncated zero mean distribution.

    Returns
    -------
    None.

    """
    dataset = pd.read_csv(noiseless)
    dataset = dataset.drop(
        dataset.loc[:, 'y_0':'noisy_ranking'].columns, axis=1)
    n_targets = 5
    original_target_names = ['or_y_' + str(i) for i in range(n_targets)]
    noisy_target_names = ['noisy_y_' + str(i) for i in range(n_targets)]
    target_names = ['y_' + str(i) for i in range(n_targets)]

    new_dataset = dataset
    for t in target_names:
        new_dataset[t] = pd.Series(dtype='float64')
    y = new_dataset['original_ranking']
    y = [s.split('>') for s in y]

    for index in range(len(y)):
        for i in range(len(y[index])):
            new_dataset.at[index, y[index][i]] = i*0.1

    noisy = dataset[original_target_names].to_numpy()

    df_y_noisy = pd.DataFrame(data=noisy, columns=noisy_target_names)
    rankings = []
    for index, row in df_y_noisy.iterrows():
        labels = []
        for t in target_names:
            labels.append((t, row['noisy_'+t]))
        labels = sorted(labels, key=lambda sub: (sub[1]), reverse=True)
        pred = [lb[0] for lb in labels]
        rankings.append('>'.join(pred))

    df_r_n = pd.DataFrame(data=rankings, columns=['noisy_ranking'])

    df = pd.concat([dataset, df_y_noisy, df_r_n], axis=1)

    y = df['noisy_ranking']
    y = [s.split('>') for s in y]

    for index in range(len(y)):
        for i in range(len(y[index])):
            df.at[index, y[index][i]] = i*(1.0/len(target_names))
    for counter in range(len(std)):
        noise = std[counter]
        noisy = dataset[original_target_names].to_numpy()
        shape = noisy.shape
        noise_generator = get_truncated_normal(mean=0, sd=noise,
                                               low=-0.25, upp=0.25)
        noisy += noise_generator.rvs(shape)

        df_y_noisy = pd.DataFrame(data=noisy, columns=noisy_target_names)
        rankings = []
        for index, row in df_y_noisy.iterrows():
            labels = []
            for t in target_names:
                labels.append((t, row['noisy_'+t]))
            labels = sorted(labels, key=lambda sub: (sub[1]), reverse=True)
            pred = [lb[0] for lb in labels]
            rankings.append('>'.join(pred))

        df_r_n = pd.DataFrame(data=rankings, columns=['noisy_ranking'])

        df = pd.concat([dataset, df_y_noisy, df_r_n], axis=1)

        y = df['noisy_ranking']
        y = [s.split('>') for s in y]

        for index in range(len(y)):
            for i in range(len(y[index])):
                df.at[index, y[index][i]] = i*(1.0/len(target_names))

        df.to_csv('../data/' + base + '/' + base + name +
                  str(counter).zfill(2)+'.csv', index=False)


def generate_noisy_m(noiseless, base, name, theta):
    """
    Generate the noisey data sets for Mallows noise.

    Parameters
    ----------
    noiseless : string
        path of noiseless dataset.
    base : string
        dataset family.
    name : string
        type of noisy dataset.
    theta : array
        theta for Mallows Model.

    Returns
    -------
    None.

    """
    df_original = pd.read_csv('../'+base+'/'+base+'_noiseless'+'.csv')

    labels = df_original['original_ranking'][0]
    target_names = sorted(labels.split('>'))
    noisy_target_names = ['noisy_' + t for t in target_names]

    df_new = df_original.drop(columns=noisy_target_names)

    for counter in range(1, len(theta)+1):
        generator = MallowsModelGenerator(theta[counter], target_names)

        df_y = pd.DataFrame(data=df_original, columns=['original_ranking'])

        for index, row in df_y.iterrows():
            ranking = generator.sample(row[0])
            df_new.at[index, 'noisy_ranking'] = ranking
            y = ranking.split('>')
            for i in range(len(y)):
                df_new.at[index, y[i]] = i * (1 / len(y))

        df_new.to_csv('../data/' + base + '/' + base + name +
                      str(counter).zfill(2)+'.csv', index=False)


base_data_path = ['SFN', 'LFN']
name = '_noisy-main'
# one of ['_noisy-main', '_noisy-supplementary-g', '_noisy-supplementary-m']


datasets_choice = '_noiseless'

for base in base_data_path:
    noiseless = '../'+base+'/'+base+'_noiseless'+'.csv'
    if datasets_choice == '_noisy-main':
        name = '_noise_gaussian_'
        std = [counter * 0.25/500 for counter in range(1, 51)]
        generate_noisy_g(noiseless, base, name, std)
    elif datasets_choice == '_noisy-supplementary-g':
        name = '_noise_gaussian2_'
        if base == 'LFN':
            std = [0.0002, 0.0005, 0.001, 0.0015, 0.0035, 0.005, 0.0075,
                   0.0085, 0.01, 0.015, 0.017, 0.02, 0.025, 0.04, 0.05,
                   0.1, 0.2, 0.275, 0.3]
        elif base == 'SFN':
            std = [0.0005, 0.0015, 0.0016, 0.0025, 0.003, 0.0035, 0.004,
                   0.0055, 0.0075, 0.01, 0.012, 0.0145, 0.023, 0.029, 0.0365,
                   0.045, 0.05, 0.1, 0.2, 0.02]
        generate_noisy_g(noiseless, base, name, std)
    elif datasets_choice == '_noisy-supplementary-m':
        name = '_noise_mallows_'
        theta = range(0.4, 4.5, 0.2)
        generate_noisy_m(noiseless, base, name, theta)
