# -*- coding: utf-8 -*-

import numpy as np
from itertools import permutations
from scipy.special import softmax


class MallowsModelGenerator:
    """Mallows Model Generator."""

    def __init__(self, theta, labels, sigma0=None):
        self.i = 0
        self.theta = theta
        self.n = len(labels)
        self.rng = np.random.default_rng()

        if sigma0 is None:
            self.sigma0 = range(self.n)
        else:
            self.sigma0 = sigma0
        self.model(theta, self.sigma0)
        self.init_dicts(labels)

    def __iter__(self):
        """
        Return iterator oject itself.

        Returns
        -------
        iterator
            DESCRIPTION.

        """
        return self

    def __next__(self):
        """
        Return the next item in the sequence.

        Returns
        -------
        array_like
            permutation.

        """
        return self.rng.choice(self.perm, p=self.prob)

    def KT(self, sigma, pi):
        """
        Compute the kendall tau distance of two permutations.

        Kendall tau distance is the number of pairwise disagreements between
        two permutations.

        Parameters
        ----------
        sigma : array_like
            Permutation.
        pi : array_like
            Permutation.

        Returns
        -------
        kt : int
            kendall tau distance of two arrays.

        """
        length = len(pi)
        kt = 0
        for i in range(length):
            for j in range(i+1, length):
                if (sigma[i] < sigma[j] and pi[i] > pi[j]) or \
                        (sigma[i] > sigma[j] and pi[i] < pi[j]):
                    kt = kt+1
        return kt

    def model(self, theta, sigma0):
        """
        Create the permutations and probabilities for the Mallows model.

        Parameters
        ----------
        theta : float
            dispersion parameter.
        sigma0 :
            central ranking.

        Returns
        -------
        perm : list of tuples
            list of permutations.
        prob : list of float
            list of correponding probabilities.

        """
        perm = list(permutations(sigma0))
        prob = np.zeros(len(perm))

        x = []
        for i in range(len(perm)):
            x.append(-theta*self.KT(perm[i], sigma0))
        prob = softmax(x)
        self.perm = perm
        self.prob = prob
        return (perm, prob)

    def init_dicts(self, labels):
        """
        Initialize the encoding and decoding dictionaries for permutions.

        Parameters
        ----------
        labels : array of strings
            Labels as strings.

        Returns
        -------
        encodeDict : dict
            Dictionary from index number to label name.
        decodeDict : dict
            Dictionary from label name to index number.

        """
        encodeDict = {}
        decodeDict = {}
        labels = sorted(labels)
        for i in range(len(labels)):
            encodeDict[i] = labels[i]
            decodeDict[labels[i]] = i
        self.encodeDict = encodeDict
        self.decodeDict = decodeDict
        return encodeDict, decodeDict

    def toStr(self, perm):
        """
        Transformw permutations from list of indexes to strings (encoding).

        Parameters
        ----------
        perm : array_like
            permutation as list of indexes, e.g. [0,1,2].

        Returns
        -------
        string
            encoded permutation, e.g. 'a>b>c'.

        """
        ret = []
        for p in perm:
            ret.append(self.encodeDict[p])
        return ">".join(ret)

    def toPerm(self, string):
        """
        Transform permutations from strings to list of indexes (decoding).

        Parameters
        ----------
        perm : string
            permutation as string, e.g. 'a>b>c'.

        Returns
        -------
        array_like
            decoded permutation, e.g. [0,1,2]

        """
        ret = []
        for s in string.split(">"):
            ret.append(self.decodeDict[s])
        return ret

    def permMatrix(self, sigma0):
        """
        Create the permutation matrix.

        The permutation matrix transforms the identity permutation to another
        permutation.

        Parameters
        ----------
        sigma0 : array_like
            given permutation.

        Returns
        -------
        2D numpy array
            permutation matrix.

        """
        I_matrix = np.eye(self.n)
        return I_matrix[sigma0].T

    def transform(self, p, sigma0):
        """
        Transform a permutation to another one, given the central permutation.

        Parameters
        ----------
        p : array_like
            permutation to be transformed.
        sigma0 : array_like
            permutation to be used as base.

        Returns
        -------
        array_like
            new permutation.

        """
        return np.dot(p, self.permMatrix(sigma0))

    def sample(self, centralPermStr):
        """
        Return a sample of Mallows Model given a central ranking.

        Parameters
        ----------
        centralPermStr : string
            central ranking .

        Returns
        -------
        string
            sample of Mallows Model.

        """
        sigma0 = self.toPerm(centralPermStr)
        p = next(self)
        t = self.transform(p, sigma0)
        return self.toStr(t)


if __name__ == '__main__':
    theta = 800
    mm = MallowsModelGenerator(theta, ['a', 'b', 'c'])
    # should allways return the same distribution
    print(next(mm))
    print(next(mm))
    print(next(mm))
    print(next(mm))
    print(next(mm))
    sigma0 = 'b>a>c'
    for i in range(100):
        print(mm.sample(sigma0))
