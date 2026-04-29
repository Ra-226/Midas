# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import time
import utils
import attacks.sap as sap
import attacks.ihop as ihop
import attacks.ikk as ikk
import attacks.midas_incremental_optimization as midas
import attacks.ihopM as ihopM
import attacks.score as score
from attacks.jigsaw import Attacker
import argparse

if __name__ == '__main__':
    args = utils.parameter_parse('Enron', 'S2')
    scenarios = args.scenarios
    dataset = args.dataset

    n = [500, 1000, 1500, 2000]
    count = 10  # Number of experiments

    df = pd.DataFrame(columns=["count", 'n', "attack", "time", "recovery"])
    with open(f'./Datasets/Enron_3000.pkl', 'rb') as f:
        pkl = pickle.load(f)

    # Extract the number of non-indexed and indexed documents and generate an access pattern matrix.
    num1 = len(pkl[2])
    num2 = len(pkl[3])
    keywords = list(pkl[0].keys())
    keywords_matrix = utils.generate_matrix(pkl[0], pkl[2])
    queries_matrix = utils.generate_matrix(pkl[1], pkl[3])

    for i_m, v_m in enumerate(n):
        for i_count in range(count):
            print(f"parameter: {v_m}, iterations: {i_count}...")
            word_len = v_m
            # Generate keyword and query sets based on different scenarios, and remove all 0 columns from the access pattern matrix.
            wordSet = [keywords[i] for i in
                       np.random.permutation(3000)[:word_len]] if scenarios == "S1" else keywords[:word_len]
            wordAccess = keywords_matrix.loc[wordSet]
            A = wordAccess.replace(0, np.nan)
            A = A.dropna(axis=1, how='all')
            A = A.replace(np.nan, 0)

            query_len = 300  # Query quantity
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
            subT = list(prior_queries_and_candidate_lists.keys())
            subN = N.loc[subT]
            subN_ = N_.loc[subT]
            subV = V.loc[subT]
            R2 = midas.RR(M, subN, U, subV, R1[:1], 1, 55)

            t2 = time.time()
            result1 = midas.CR(M, N, R2, 10, 4, 25)
            midastime = time.time() - t1
            t0 = time.time()
            result0 = midas.CR(M, N, R2, 10, 1, 25)
            midas_1_time = time.time() - t0 + t2 - t1
            acc = utils.accuracy(result0)
            df.loc[len(df)] = [i_count, v_m, 'midas_1', midas_1_time, acc / len(query)]
            print(df.iloc[-1])

            acc = utils.accuracy(result1)
            df.loc[len(df)] = [i_count, v_m, 'midas', midastime, acc / len(query)]
            print(df.iloc[-1])

            t8 = time.time()
            attack = Attacker(M.to_numpy(), N.to_numpy(), 45, 35, 10, 1, 0.9)
            attack.attack_step_1()
            attack.attack_step_2()
            result7 = attack.attack_step_3()
            res_Jigsaw = [(query[q_ind], wordSet[w_ind]) for q_ind, w_ind in result7.items()]
            acc = utils.accuracy(res_Jigsaw)
            df.loc[len(df)] = [i_count, v_m, 'jigsaw', time.time() - t8, acc / len(query)]
            print(df.iloc[-1])

            t3 = time.time()
            result2 = score.scorePlus(M, N, R2, 10)
            scoretime = time.time() - t3
            acc = utils.accuracy(result2)
            df.loc[len(df)] = [i_count, v_m, 'score', scoretime, acc / len(query)]
            print(df.iloc[-1])

            t4 = time.time()
            result3 = ikk.run_ikk(M.values, N.values, wordSet, query)
            ikktime = time.time() - t4
            acc = utils.accuracy(result3)
            df.loc[len(df)] = [i_count, v_m, 'ikk', ikktime, acc / len(query)]
            print(df.iloc[-1])

            t5 = time.time()
            probabilities = np.diag(M)
            observations = np.diag(N) * num2
            result4 = sap._run_algorithm(num2, probabilities, observations, wordSet, query)
            saptime = time.time() - t5
            acc = utils.accuracy(result4)
            df.loc[len(df)] = [i_count, v_m, 'sap', saptime, acc / len(query)]
            print(df.iloc[-1])

            t6 = time.time()
            result5 = ihop.run_ihop(num2, M.values, N.values, wordSet, query, 0.25, 1000)
            ihoptime = time.time() - t6
            acc = utils.accuracy(result5)
            df.loc[len(df)] = [i_count, v_m, 'ihop', ihoptime, acc / len(query)]
            print(df.iloc[-1])

            t7 = time.time()
            result6 = ihopM.run_ihop(num2, M.values, N.values, wordSet, query, 0.25, 500, R2[:5])
            ihopMtime = time.time() - t7 + t2 - t1
            acc = utils.accuracy(result6)
            df.loc[len(df)] = [i_count, v_m, 'ihopM', ihopMtime, acc / len(query)]
            print(df.iloc[-1])

    with open(f"./pic_pkl/n{scenarios}{dataset}.pkl", "wb") as f:
        pickle.dump(df, f)
