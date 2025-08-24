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

    map_ihop = {500: 160, 1000: 102, 1500: 65, 2000: 25}
    map_ihopM = {500: 160, 1000: 102, 1500: 65, 2000: 25}
    map_score_refspeed = {500: 5, 1000: 9, 1500: 12, 2000: 16}
    map_jigsaw_refspeed = {500: 1, 1000: 4, 1500: 8, 2000: 10}

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
            acc = utils.accuracy(result1)
            df.loc[len(df)] = [i_count, v_m, 'midas', midastime, acc / len(query)]

            t8 = time.time()
            attack = Attacker(M.to_numpy(), N.to_numpy(), 45, 35, map_jigsaw_refspeed[word_len], 1, 0.9)
            attack.attack_step_1()
            attack.attack_step_2()
            result7 = attack.attack_step_3()
            res_Jigsaw = [(query[q_ind], wordSet[w_ind]) for q_ind, w_ind in result7.items()]
            acc = utils.accuracy(res_Jigsaw)
            df.loc[len(df)] = [i_count, v_m, 'jigsaw', time.time() - t8, acc / len(query)]

            # t3 = time.time()
            # result2 = score.scorePlus(M, N, R2, map_score_refspeed[word_len])
            # scoretime = time.time() - t3
            # acc = utils.accuracy(result2)
            # df.loc[len(df)] = [i_count, v_m, 'score', scoretime, acc / len(query)]

            t6 = time.time()
            result5 = ihop.run_ihop(num2, M.values, N.values, wordSet, query, 0.25, map_ihop[word_len])
            ihoptime = time.time() - t6
            acc = utils.accuracy(result5)
            df.loc[len(df)] = [i_count, v_m, 'ihop', ihoptime, acc / len(query)]

            t7 = time.time()
            result6 = ihopM.run_ihop(num2, M.values, N.values, wordSet, query, 0.25, map_ihopM[word_len], R2[:5])
            ihopMtime = time.time() - t7 + t2 - t1
            acc = utils.accuracy(result6)
            df.loc[len(df)] = [i_count, v_m, 'ihopM', ihopMtime, acc / len(query)]
            print(df.iloc[-4:])

    with open(f"./pic_pkl/n{scenarios}{dataset}LimitedTime.pkl", "wb") as f:
        pickle.dump(df, f)
