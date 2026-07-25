# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import time
import utils
import attacks.midas as midas
import attacks.ihopM as ihopM

if __name__ == '__main__':
    theta = [0, 1, 5, 10, 25]
    niter = [10, 20, 50, 100, 200, 500, 1000, 2000]
    count = 10  # Number of experiments
    rate = 0.25
    queryRate = 1  # Query proportion
    word_len = 500  # Number of keywords
    query_len = int(word_len * queryRate)  # Query quantity

    auxiliary_knowledge_quantity = 5000

    df = pd.DataFrame(columns=["count", 'x', 'theta', "time", "recovery"])
    args = utils.parameter_parse('Enron', 'S1')
    dataset = args.dataset

    with open(f'./Datasets/{dataset}_3000.pkl', 'rb') as f:
        pkl = pickle.load(f)

    # Extract the number of non-indexed and indexed documents and generate an access pattern matrix.
    num1 = len(pkl[2])
    num2 = len(pkl[3])

    keywords = list(pkl[0].keys())

    keywords_matrix = utils.generate_matrix(pkl[0], pkl[2])
    queries_matrix = utils.generate_matrix(pkl[1], pkl[3])

    for i_n, nn in enumerate(niter):
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

            t1 = time.time()
            prior_queries_and_candidate_lists, R1 = midas.PR(volumeToken, volumeKeyword, vTD, vKD, 20)
            subT = list(prior_queries_and_candidate_lists.keys())
            subN = N.loc[subT]
            subN_ = N_.loc[subT]
            subV = V.loc[subT]
            R2 = midas.RR(M, subN, U, subV, R1[:1], 1, 25)
            t2 = time.time()

            for i_m, v_m in enumerate(theta):
                t7 = time.time()
                result = ihopM.run_ihop(num2, M.values, N.values, wordSet, query, rate, nn, R2[:v_m])
                ihopMtime = time.time() - t7 + t2 - t1
                acc = utils.accuracy(result)
                df.loc[len(df)] = [i_count, i_n, v_m, ihopMtime, acc / len(query)]
                print(df.iloc[-1])

    with open(f"./pic_pkl/ihopM{dataset}.pkl", "wb") as f:
        pickle.dump(df, f)
