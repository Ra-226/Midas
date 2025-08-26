# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:57:55 2023

@author: Ra
"""

import pickle
import numpy as np
import pandas as pd
import time
import utils
import attacks.sap as sap
import attacks.ihop as ihop
import attacks.ikk as ikk
import attacks.midas_incremental_optimization as midas
import attacks.midas as midasSlow

if __name__ == '__main__':
    args = utils.parameter_parse('Enron', 'S1')
    scenarios = args.scenarios
    dataset = args.dataset

    m = [0.25, 0.5, 0.75, 1]
    count = 10  # Number of experiments
    word_len = 500  # Number of keywords

    file = "Enron_3000" if dataset == "Enron" else "Lucene_3000"
    df = pd.DataFrame(columns=["count", 'm', "attack", "time", "recovery"])
    with open(f'./Datasets/{file}.pkl', 'rb') as f:
        pkl = pickle.load(f)

    # Extract the number of non-indexed and indexed documents and generate an access pattern matrix.
    num1 = len(pkl[2])
    num2 = len(pkl[3])
    keywords = list(pkl[0].keys())
    keywords_matrix = utils.generate_matrix(pkl[0], pkl[2])
    queries_matrix = utils.generate_matrix(pkl[1], pkl[3])

    for i_m, v_m in enumerate(m):
        for i_count in range(count):
            print(f"parameter: {v_m}, iterations: {i_count}...")
            # Generate keyword and query sets based on different scenarios, and remove all 0 columns from the access pattern matrix.
            wordSet = [keywords[i] for i in
                       np.random.permutation(3000)[:word_len]] if scenarios == "S1" else keywords[:word_len]
            wordAccess = keywords_matrix.loc[wordSet]
            A = wordAccess.replace(0, np.nan)
            A = A.dropna(axis=1, how='all')
            A = A.replace(np.nan, 0)

            queryRate = v_m
            query_len = int(word_len * queryRate)  # Query quantity
            query = [wordSet[i] for i in range(query_len)] \
                if scenarios == "S3" else [wordSet[i] for i in np.random.permutation(word_len)[:query_len]]
            queryAccess = queries_matrix.loc[query]
            B = queryAccess.replace(0, np.nan)
            B = B.dropna(axis=1, how='all')
            B = B.replace(np.nan, 0)

            # Generate co-occurrence matrix and co-absence matrix.
            M, N = utils.co_occurrence(A, B, num1, num2)
            M_, N_ = utils.co_absence(A, B, num1, num2)

            # The main diagonal of the co-occurrence(absence) matrix
            volumeToken = pd.Series(np.diag(N), index=N.index)
            volumeKeyword = pd.Series(np.diag(M), index=M.index)
            vTD = pd.Series(np.diag(N_), index=N_.index)
            vKD = pd.Series(np.diag(M_), index=M_.index)

            U = (volumeKeyword - M).T
            V = (volumeToken - N).T

            t1 = time.time()
            prior_queries_and_candidate_lists, R1 = midas.PR(volumeToken, volumeKeyword, vTD, vKD, 20)
            t2 = time.time()
            subT = list(prior_queries_and_candidate_lists.keys())
            subN = N.loc[subT]
            subN_ = N_.loc[subT]
            subV = V.loc[subT]
            R2 = midas.RR(M, subN, U, subV, R1[:1], 1, 55)

            result1 = midas.CR(M, N, R2, 10, 4, 25)
            midastime = time.time() - t1
            acc = utils.accuracy(result1)
            df.loc[len(df)] = [i_count, v_m, 'midas', midastime, acc / len(query)]

            t3 = time.time()
            R2_ = midasSlow.RR(M, subN, U, subV, R1[:1], 1, 55)
            result2 = midasSlow.CR(M, N, R2_, 10, 4, 25)
            scoretime = time.time() - t3 + t2 - t1
            acc = utils.accuracy(result2)
            df.loc[len(df)] = [i_count, v_m, 'midasSlow', scoretime, acc / len(query)]
            print(df.iloc[-2:])

    with open(f"./pic_pkl/ablationOptimize{scenarios}M{dataset}.pkl", "wb") as f:
        pickle.dump(df, f)
