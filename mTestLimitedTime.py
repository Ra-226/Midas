# -*- coding: utf-8 -*-

import pickle
import random
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

if __name__ == '__main__':
    args = utils.parameter_parse('Enron', 'S1')
    scenarios = args.scenarios
    dataset = args.dataset

    IHOP_PROBE_ITERS = 20
    JIGSAW_PROBE_REFSPEED = 5

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
        per_iter_ihop = None
        per_iter_ihopM = None
        C_jig = None
        for i_count in range(count):
            np.random.seed(i_m * count + i_count)
            random.seed(i_m * count + i_count)
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

            # Calibrate per-unit cost once per m (using the first iteration's matrices).
            if i_count == 0:
                np.random.seed((i_m * count) * 10 + 1)
                t0 = time.time()
                ihop.run_ihop(num2, M.values, N.values, wordSet, query, 0.25, IHOP_PROBE_ITERS)
                per_iter_ihop = (time.time() - t0) / IHOP_PROBE_ITERS

                np.random.seed((i_m * count) * 10 + 2)
                t0 = time.time()
                ihopM.run_ihop(num2, M.values, N.values, wordSet, query, 0.25, IHOP_PROBE_ITERS, [])
                per_iter_ihopM = (time.time() - t0) / IHOP_PROBE_ITERS

                t0 = time.time()
                probe_attack = Attacker(M.to_numpy(), N.to_numpy(),
                                        45, 35, JIGSAW_PROBE_REFSPEED, 1, 0.9)
                probe_attack.attack_step_1()
                probe_attack.attack_step_2()
                probe_attack.attack_step_3()
                C_jig = (time.time() - t0) * JIGSAW_PROBE_REFSPEED
                print(f"  [calib] per_iter_ihop={per_iter_ihop:.4f}s "
                      f"per_iter_ihopM={per_iter_ihopM:.4f}s C_jig={C_jig:.3f}")

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
            refinespeed = max(1.0, C_jig / midastime)
            attack = Attacker(M.to_numpy(), N.to_numpy(), 45, 35, refinespeed, 1, 0.9)
            attack.attack_step_1()
            attack.attack_step_2()
            result7 = attack.attack_step_3()
            res_Jigsaw = [(query[q_ind], wordSet[w_ind]) for q_ind, w_ind in result7.items()]
            jigsawtime = time.time() - t8
            acc = utils.accuracy(res_Jigsaw)
            df.loc[len(df)] = [i_count, v_m, 'jigsaw', jigsawtime, acc / len(query)]

            np.random.seed((i_m * count + i_count) * 10 + 1)
            t6 = time.time()
            n_iters_ihop = max(10, int(round(midastime / per_iter_ihop)))
            result5 = ihop.run_ihop(num2, M.values, N.values, wordSet, query, 0.25, n_iters_ihop)
            ihoptime = time.time() - t6
            acc = utils.accuracy(result5)
            df.loc[len(df)] = [i_count, v_m, 'ihop', ihoptime, acc / len(query)]

            np.random.seed((i_m * count + i_count) * 10 + 2)
            t7 = time.time()
            n_iters_ihopM = max(10, int(round((midastime - (t2 - t1)) / per_iter_ihopM)))
            result6 = ihopM.run_ihop(num2, M.values, N.values, wordSet, query, 0.25, n_iters_ihopM, R2[:5])
            ihopMtime = time.time() - t7 + t2 - t1
            acc = utils.accuracy(result6)
            df.loc[len(df)] = [i_count, v_m, 'ihopM', ihopMtime, acc / len(query)]
            print(f"  times: midas={midastime:.2f}s jigsaw={jigsawtime:.2f}s "
                  f"ihop={ihoptime:.2f}s ihopM={ihopMtime:.2f}s "
                  f"(refinespeed={refinespeed:.1f}, n_iters={n_iters_ihop}/{n_iters_ihopM})")
            print(df.iloc[-4:])

    with open(f"./pic_pkl/m{scenarios}{dataset}LimitedTime.pkl", "wb") as f:
        pickle.dump(df, f)
