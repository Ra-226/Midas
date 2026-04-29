# -*- coding: utf-8 -*-

# from github.com/simon-oya/USENIX22-ihop-code

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment as hungarian


def _log_binomial(n, beta):
    """Computes an approximation of log(binom(n, n*alpha)) for alpha < 1"""
    if beta == 0 or beta == 1:
        return 0
    elif beta < 0 or beta > 1:
        raise ValueError("beta cannot be negative or greater than 1 ({})".format(beta))
    else:
        entropy = -beta * np.log(beta) - (1 - beta) * np.log(1 - beta)
        return n * entropy - 0.5 * np.log(2 * np.pi * n * beta * (1 - beta))


def compute_log_binomial_probability_matrix(ntrials, probabilities, observations, wordSet, query):
    """
    Computes the logarithm of binomial probabilities of each pair of probabilities and observations.
    :param ntrials: number of binomial trials
    :param probabilities: vector with probabilities
    :param observations: vector with integers (observations)
    :return log_matrix: |probabilities| x |observations| matrix with the log binomial probabilities
    """
    probabilities = np.array(probabilities)
    if any(probabilities > 0):
        probabilities[probabilities == 0] = min(probabilities[
                                                    probabilities > 0]) / 100  # To avoid numerical errors. An error would mean the adversary information is very off.
    else:
        probabilities += 1e-10

    log_binom_term = np.array([_log_binomial(ntrials, obs / ntrials) for obs in observations])  # ROW TERM
    column_term = np.array([np.log(probabilities) - np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    last_term = np.array([ntrials * np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    log_matrix = log_binom_term + np.array(observations) * column_term + last_term
    # log_matrix =  np.array(observations) * column_term + last_term

    log_matrix = pd.DataFrame(log_matrix, wordSet, query)
    return -log_matrix


def _run_algorithm(ntrials, probabilities, observations, wordSet, query):
    """Runs the Hungarian algorithm with the given cost matrix
    :param c_matrix: cost matrix, (n_keywords x n_tokens)"""

    c_matrix = compute_log_binomial_probability_matrix(ntrials, probabilities, observations, wordSet, query)
    row_ind, col_ind = hungarian(c_matrix)

    result = []
    for i in range(len(row_ind)):
        result.append((query[col_ind[i]], wordSet[row_ind[i]]))

    return result
