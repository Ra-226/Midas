# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:59:22 2023

@author: Ra-226
"""

import pickle
import numpy as np
import pandas as pd
import utils
import attacks.midas_incremental_optimization as midas

if __name__ == '__main__':
    gamma = [1, 2, 3, 4]
    count = 30  # Number of experiments
    queryRate = 0.25  # Query proportion
    word_len = 500  # Number of keywords
    query_len = int(word_len * queryRate)  # Query quantity

    auxiliary_knowledge_quantity = 5000
    mu = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    df = pd.DataFrame(columns=["count", "gamma", "mu", "recovery"])
    with open('./Datasets/Enron_3000.pkl', 'rb') as f:
        pkl = pickle.load(f)

    # Extract the number of non-indexed and indexed documents and generate an access pattern matrix.
    num1 = len(pkl[2])
    num2 = len(pkl[3])
    keywords = list(pkl[0].keys())
    keywords_matrix = utils.generate_matrix(pkl[0], pkl[2])
    queries_matrix = utils.generate_matrix(pkl[1], pkl[3])

    for i_u, v_u in enumerate(mu):
        for i_count in range(count):
            wordSet = keywords[:word_len]
            wordAccess = keywords_matrix.loc[wordSet]
            selectDocument = list(wordAccess.columns[np.random.permutation(num1)[:auxiliary_knowledge_quantity]])
            wordAccess = wordAccess.loc[:, selectDocument]
            A = wordAccess.replace(0, np.nan)
            A = A.dropna(axis=1, how='all')
            A = A.replace(np.nan, 0)

            query = [wordSet[i] for i in np.random.permutation(word_len)[:query_len]]
            queryAccess = queries_matrix.loc[query]
            B = queryAccess.replace(0, np.nan)
            B = B.dropna(axis=1, how='all')
            B = B.replace(np.nan, 0)

            # Generate co-occurrence matrix and co-absence matrix.
            M, N = utils.co_occurrence(A, B, auxiliary_knowledge_quantity, num2)
            M_, N_ = utils.co_absence(A, B, auxiliary_knowledge_quantity, num2)

            # The main diagonal of the co-occurrence(absence) matrix
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
            R2 = midas.RR(M, subN, U, subV, R1[:1], 1, 55)

            for i_r, v_r in enumerate(gamma):
                result = midas.CR(M, N, R2, 10, v_r, v_u)
                df.loc[len(df)] = [i_count, v_r, v_u, utils.accuracy(result) / len(query)]
                print(v_r, v_u, i_count)
                print(df.iloc[-1])

    with open(f"./pic_pkl/CRTest_Dsim_{auxiliary_knowledge_quantity}_count_{count}.pkl", "wb") as f:
        pickle.dump(df, f)
