# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 19:23:49 2023

@author: Ra
"""

import pickle
import numpy as np
import pandas as pd
import attacks.midas_incremental_optimization as midas
import utils

if __name__ == '__main__':
    count = 10
    m = [0.25, 0.5, 0.75]
    e_n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    word_len = 500  # Number of keywords
    df = pd.DataFrame(columns=["count", "e_n", "m", "recovery"])
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

    for i_m, v_m in enumerate(m):
        for i_r, r in enumerate(e_n):
            for i_count in range(count):
                queryRate = v_m  # Query proportion
                query_len = int(word_len * queryRate)  # Query quantity
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

                prior_queries_and_candidate_lists, R1 = midas.PR(volumeToken, volumeKeyword, vTD, vKD, 20)

                subT = list(prior_queries_and_candidate_lists.keys())
                subN = N.loc[subT]
                subN_ = N_.loc[subT]
                subV = V.loc[subT]

                R2 = midas.RR(M, subN, U, subV, R1[:r], 1, 55)
                if len(R2) != 0: df.loc[len(df)] = [i_count, r, v_m, utils.accuracy(R2) / len(R2)]
                print(v_m, r)

    with open("./pic_pkl/RR.pkl", "wb") as f:
        pickle.dump(df, f)
