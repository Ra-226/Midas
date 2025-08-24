# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:08:33 2023

@author: Ra
"""

import pickle
import numpy as np
import pandas as pd
import utils
import attacks.midas_incremental_optimization as midas

if __name__ == '__main__':
    count = 20  # Number of experiments
    p = [10, 15, 20, 25]
    m = [0.25, 0.5, 0.75]
    word_len = 500  # Number of keywords

    df = pd.DataFrame(columns=["count", "len", "p", "m", "inclusionRate", "recovery", "top5"])

    with open('./Datasets/Enron_3000.pkl', 'rb') as f:
        pkl = pickle.load(f)

    # Extract the number of non-indexed and indexed documents and generate an access pattern matrix.
    num1 = len(pkl[2])
    num2 = len(pkl[3])
    keywords = list(pkl[0].keys())
    keywords_matrix = utils.generate_matrix(pkl[0], pkl[2])
    queries_matrix = utils.generate_matrix(pkl[1], pkl[3])

    wordSet = keywords[:word_len]
    wordAccess = keywords_matrix.loc[wordSet]
    A = wordAccess.replace(0, np.nan)
    A = A.dropna(axis=1, how='all')
    A = A.replace(np.nan, 0)

    for i_p, v_p in enumerate(p):
        for i_m, v_m in enumerate(m):
            for i_count in range(count):
                queryRate = v_m  # Query proportion
                query_len = int(word_len * queryRate)
                query = [wordSet[i] for i in np.random.permutation(word_len)[:query_len]]

                queryAccess = queries_matrix.loc[query]
                B = queryAccess.replace(0, np.nan)
                B = B.dropna(axis=1, how='all')
                B = B.replace(np.nan, 0)

                # Generate co-occurrence matrix and non-co-occurrence matrix.
                M, N = utils.co_occurrence(A, B, num1, num2)
                M_, N_ = utils.mutual_difference(A, B, num1, num2)
                # The main diagonal of the (non-)co-occurrence matrix
                volumeToken = pd.Series(np.diag(N), index=N.index)
                volumeKeyword = pd.Series(np.diag(M), index=M.index)
                vTD = pd.Series(np.diag(N_), index=N_.index)
                vKD = pd.Series(np.diag(M_), index=M_.index)

                U = (volumeKeyword - M).T
                V = (volumeToken - N).T

                prior_queries_and_candidate_lists, R1 = midas.PR(volumeToken, volumeKeyword, vTD, vKD, v_p)

                existence_rate = 0
                acc = 0
                top5_acc = 0
                for i in prior_queries_and_candidate_lists.keys():
                    if i in prior_queries_and_candidate_lists[i]:
                        existence_rate += 1
                for i in R1:
                    if i[0] == i[1]:
                        acc += 1
                for i in R1[:5]:
                    if i[0] == i[1]:
                        top5_acc += 1

                if len(prior_queries_and_candidate_lists) != 0:
                    df.loc[(i_p * 4 + i_m) * count + i_count] = [i_count, len(prior_queries_and_candidate_lists), v_p,
                                                                 v_m,
                                                                 existence_rate / len(
                                                                     prior_queries_and_candidate_lists),
                                                                 acc / len(R1), top5_acc / 5]

    with open("./pic_pkl/PR.pkl", "wb") as f:
        pickle.dump(df, f)
