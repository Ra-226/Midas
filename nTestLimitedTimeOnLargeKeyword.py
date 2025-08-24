import pickle
import time
import numpy as np
import pandas as pd
import attacks.ihop as ihop
import attacks.ihopM as ihopM
import attacks.midas_incremental_optimization as midas
import utils
from attacks.jigsaw import Attacker

if __name__ == '__main__':
    args = utils.parameter_parse('Lucene', 'S2')
    scenarios = args.scenarios
    n = [3000, 5000, 6000]
    speed_map_Jigsaw = {1000: 12, 3000: 16, 5000: 25, 6000: 25}
    # speed_map = {1000: 347, 3000 : 160, 5000 : 147}   # for query_len = 300
    speed_map = {1000: 40, 3000: 28, 5000: 20, 6000: 23}  # for query_len = 500 S1
    query_len = 500
    count = 10  # Number of experiments

    df = pd.DataFrame(columns=["count", 'n', "attack", "time", "recovery"])
    with open('./Datasets/Lucene_6000.pkl', 'rb') as f:
        pkl = pickle.load(f)

    # Extract the number of non-indexed and indexed documents and generate an access pattern matrix.
    num1 = len(pkl[2])
    num2 = len(pkl[3])
    keywords = list(pkl[0].keys())
    keywords_matrix = utils.generate_matrix(pkl[0], pkl[2])
    queries_matrix = utils.generate_matrix(pkl[1], pkl[3])

    cnt = 0
    for i_m, v_m in enumerate(n):
        for i_count in range(count):
            print(f"parameter: {v_m}, iterations: {i_count}...")
            word_len = v_m
            # Generate keyword and query sets based on different scenarios, and remove all 0 columns from the access pattern matrix.
            wordSet = [keywords[i] for i in
                       np.random.permutation(6000)[:word_len]] if scenarios == "S2" else keywords[:word_len]
            wordAccess = keywords_matrix.loc[wordSet]
            A = wordAccess.replace(0, np.nan)
            A = A.dropna(axis=1, how='all')
            A = A.replace(np.nan, 0)

            query = [wordSet[i] for i in range(query_len)] if scenarios == "S3" else [wordSet[i] for i in
                                                                                      np.random.permutation(word_len)[
                                                                                      :query_len]]
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
            attack = Attacker(M.to_numpy(), N.to_numpy(), 45, 35, speed_map_Jigsaw[v_m], 1, 0.9)
            attack.attack_step_1()
            attack.attack_step_2()
            result7 = attack.attack_step_3()
            res_Jigsaw = [(query[q_ind], wordSet[w_ind]) for q_ind, w_ind in result7.items()]
            jigsawtime = time.time() - t8
            acc = utils.accuracy(res_Jigsaw)
            df.loc[len(df)] = [i_count, v_m, 'jigsaw', jigsawtime, acc / len(query)]

            t6 = time.time()
            result5 = ihop.run_ihop(num2, M.values, N.values, wordSet, query, 0.25, speed_map[v_m])
            ihoptime = time.time() - t6
            acc = utils.accuracy(result5)
            df.loc[len(df)] = [i_count, v_m, 'ihop', ihoptime, acc / len(query)]

            t7 = time.time()
            result6 = ihopM.run_ihop(num2, M.values, N.values, wordSet, query, 0.25, speed_map[v_m], R2[:5])
            ihopMtime = time.time() - t7 + t2 - t1
            acc = utils.accuracy(result6)
            df.loc[len(df)] = [i_count, v_m, 'ihopM', ihopMtime, acc / len(query)]

            print(df.iloc[-4:])

    with open(f"./pic_pkl/limited_time_{scenarios}_LuceneLarge_n_{[i for i in n]}_m_{query_len}_count_{count}.pkl",
              "wb") as f:
        pickle.dump(df, f)

