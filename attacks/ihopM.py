# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment as hungarian


def compute_log_binomial_probability_matrix(ntrials, probabilities, observations):
    """
    Computes the logarithm of binomial probabilities of each pair of probabilities and observations.
    :param ntrials: number of binomial trials
    :param probabilities: vector with probabilities
    :param observations: vector with integers (observations)
    :return log_matrix: |probabilities| x |observations| matrix with the log binomial probabilities
    """
    probabilities = np.clip(np.array(probabilities), 1e-10, 1 - 1e-10)
    column_term = np.array([np.log(probabilities) - np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    last_term = np.array([ntrials * np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    log_matrix = np.array(observations) * column_term + last_term
    return log_matrix


def _build_cost_Vol_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags, ndocs, M, N):
    cost_vol = -compute_log_binomial_probability_matrix(ndocs, np.diagonal(M)[free_keywords],
                                                        np.diagonal(N)[free_tags] * ndocs)
    for tag, kw in zip(fixed_tags, fixed_keywords):
        cost_vol -= compute_log_binomial_probability_matrix(ndocs, M[kw, free_keywords], N[tag, free_tags] * ndocs)
    return cost_vol


def ihop_attack(ndocs, M, N, wordSet, query, pct_free, n_iters, known_queries):
    nrep = len(wordSet)
    ntok = len(query)
    ground_truth_tokens, ground_truth_reps = [], []
    if len(known_queries) != 0:
        for known in known_queries:
            ground_truth_tokens.append(query.index(known[0]))
            ground_truth_reps.append(wordSet.index(known[1]))

    unknown_toks = [i for i in range(ntok) if i not in ground_truth_tokens]
    unknown_reps = [i for i in range(nrep) if i not in ground_truth_reps]

    # First matching:
    # ground_truth_reps, ground_truth_tokens
    c_matrix_original = _build_cost_Vol_some_fixed(unknown_reps, unknown_toks, ground_truth_reps, ground_truth_tokens,
                                                   ndocs, M, N)
    row_ind, col_ind = hungarian(c_matrix_original)
    replica_predictions_for_each_token = {token: rep for token, rep in zip(ground_truth_tokens, ground_truth_reps)}
    for j, i in zip(col_ind, row_ind):
        replica_predictions_for_each_token[unknown_toks[j]] = unknown_reps[i]

    run_multiple_niters, niter_list, rep_pred_tok_list = False, [], []

    # Iterate using co-occurrence:
    n_free = int(pct_free * len(unknown_toks))
    assert n_free > 1
    for k in range(n_iters):
        random_unknown_tokens = list(np.random.permutation(unknown_toks))
        free_tokens = random_unknown_tokens[:n_free]
        fixed_tokens = random_unknown_tokens[n_free:] + ground_truth_tokens
        fixed_reps = [replica_predictions_for_each_token[token] for token in fixed_tokens]
        free_replicas = [rep for rep in unknown_reps if rep not in fixed_reps]

        c_matrix = _build_cost_Vol_some_fixed(free_replicas, free_tokens, fixed_reps, fixed_tokens, ndocs, M, N)

        row_ind, col_ind = hungarian(c_matrix)
        for j, i in zip(col_ind, row_ind):
            replica_predictions_for_each_token[free_tokens[j]] = free_replicas[i]

        if run_multiple_niters and k + 1 in niter_list:
            rep_pred_tok_list.append(replica_predictions_for_each_token.copy())

        if (k + 1) % (n_iters // 10) == 0:
            print("{:d}".format(((k + 1) // (n_iters // 10)) - 1), end='', flush=True)
    # print(replica_predictions_for_each_token)
    if not run_multiple_niters:
        keyword_predictions_for_each_query = [wordSet[replica_predictions_for_each_token[token]] for token in
                                              range(len(query))]
        return keyword_predictions_for_each_query
    else:
        kw_pred_for_each_query_list = []
        for replica_predictions_for_each_token in rep_pred_tok_list:
            kw_pred_for_each_query_list.append(
                [wordSet[replica_predictions_for_each_token[token]] for token in range(len(query))])
        return kw_pred_for_each_query_list


def run_ihop(ndocs, M, N, wordSet, query, pct_free, n_iters, known_queries):
    tokenToWord = ihop_attack(ndocs, M, N, wordSet, query, pct_free, n_iters, known_queries)
    result = []
    for i in range(len(query)):
        result.append((query[i], tokenToWord[i]))

    return result
