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
    args = utils.parameter_parse('Enron', 'S3')
    scenarios = args.scenarios
    dataset = args.dataset

    TPR = 0.9999
    FPR_list = [0.01, 0.02, 0.05, 0]
    # FPR_list = [0.005, 0.01, 0.02, 0]
    # FPR_list = [0.001, 0.002, 0.005, 0]
    queryRate = 0.5
    count = 10
    word_len = 500

    file = "Enron_3000" if dataset == "Enron" else "Lucene_3000"
    df = pd.DataFrame(columns=["count", "fpr", "attack", "time", "recovery"])

    with open(f'./Datasets/{file}.pkl', 'rb') as f:
        pkl = pickle.load(f)

    num1 = len(pkl[2])
    num2 = len(pkl[3])
    keywords = list(pkl[0].keys())
    keywords_matrix = utils.generate_matrix(pkl[0], pkl[2])
    queries_matrix = utils.generate_matrix(pkl[1], pkl[3])

    for i_fpr, fpr in enumerate(FPR_list):
        for i_count in range(count):
            np.random.seed(i_fpr * count + i_count)
            random.seed(i_fpr * count + i_count)
            print(f"FPR: {fpr}, iteration: {i_count}...")

            wordSet = [keywords[i] for i in
                       np.random.permutation(3000)[:word_len]] if scenarios == "S1" else keywords[:word_len]
            wordAccess = keywords_matrix.loc[wordSet]
            A = wordAccess.replace(0, np.nan)
            A = A.dropna(axis=1, how='all')
            A = A.replace(np.nan, 0)

            query_len = int(word_len * queryRate)
            query = [wordSet[i] for i in range(query_len)] \
                if scenarios == "S3" else [wordSet[i] for i in np.random.permutation(word_len)[:query_len]]
            queryAccess = queries_matrix.loc[query]
            B = queryAccess.replace(0, np.nan)
            B = B.dropna(axis=1, how='all')
            B = B.replace(np.nan, 0)

            M, _ = utils.co_occurrence(A, A, num1, num1)
            M_, _ = utils.co_absence(A, A, num1, num1)

            M_orig = M.copy()
            M__orig = M_.copy()
            if fpr != 0:
                B_osse = utils.osse_obfuscate(B, TPR, fpr)
                M = utils.CLRZ(M_orig, M__orig, TPR, fpr)
                M_ = utils.CLRZ(M_orig, M__orig, 1 - TPR, 1 - fpr)
            else:
                B_osse = B.copy()

            N, _ = utils.co_occurrence(B_osse, B_osse, num2, num2)
            N_, _ = utils.co_absence(B_osse, B_osse, num2, num2)

            volumeToken = pd.Series(np.diag(N), index=N.index)
            volumeKeyword = pd.Series(np.diag(M), index=M.index)
            vTD = pd.Series(np.diag(N_), index=N_.index)
            vKD = pd.Series(np.diag(M_), index=M_.index)
            U = (volumeKeyword - M).T
            V = (volumeToken - N).T

            # ---- Midas ----
            t1 = time.time()
            prior_queries_and_candidate_lists, R1 = midas.PR(
                volumeToken, volumeKeyword, vTD, vKD, 20)
            t2 = time.time()
            subT = list(prior_queries_and_candidate_lists.keys())
            subN = N.loc[subT]
            subN_ = N_.loc[subT]
            subV = V.loc[subT]
            R2 = midas.RR(M, subN, U, subV, R1[:1], 1, 55)

            result1 = midas.CR(M, N, R2, 10, 4, 25)
            midas_time = time.time() - t1
            acc = utils.accuracy(result1)
            df.loc[len(df)] = [i_count, fpr, 'midas',
                               midas_time, acc / len(query)]
            print(f"  midas:  {df.iloc[-1]['recovery']:.3f} ({df.iloc[-1]['time']:.2f}s)")

            # ---- Midas_1 ----
            t0 = time.time()
            result0 = midas.CR(M, N, R2, 10, 1, 25)
            midas_1_time = time.time() - t0 + t2 - t1
            acc = utils.accuracy(result0)
            df.loc[len(df)] = [i_count, fpr, 'midas_1',
                               midas_1_time, acc / len(query)]
            print(f"  midas_1: {df.iloc[-1]['recovery']:.3f} ({df.iloc[-1]['time']:.2f}s)")

            # ---- Jigsaw ----
            t4 = time.time()
            attack = Attacker(M.to_numpy(), N.to_numpy(),
                               45, 35, 10, 1, 0.9)
            attack.attack_step_1()
            attack.attack_step_2()
            result_jigsaw = attack.attack_step_3()
            res_Jigsaw = [(query[q_ind], wordSet[w_ind])
                          for q_ind, w_ind in result_jigsaw.items()]
            acc = utils.accuracy(res_Jigsaw)
            df.loc[len(df)] = [i_count, fpr, 'jigsaw',
                               time.time() - t4, acc / len(query)]
            print(f"  jigsaw: {df.iloc[-1]['recovery']:.3f} ({df.iloc[-1]['time']:.2f}s)")

            # ---- Score ----
            t3 = time.time()
            result_score = score.scorePlus(M, N, R2, 10)
            score_time = time.time() - t3
            acc = utils.accuracy(result_score)
            df.loc[len(df)] = [i_count, fpr, 'score',
                               score_time, acc / len(query)]
            print(f"  score:  {df.iloc[-1]['recovery']:.3f} ({df.iloc[-1]['time']:.2f}s)")

            # ---- IKK ----
            t6 = time.time()
            result_ikk = ikk.run_ikk(M.values, N.values, wordSet, query)
            ikk_time = time.time() - t6
            acc = utils.accuracy(result_ikk)
            df.loc[len(df)] = [i_count, fpr, 'ikk',
                               ikk_time, acc / len(query)]
            print(f"  ikk:    {df.iloc[-1]['recovery']:.3f} ({df.iloc[-1]['time']:.2f}s)")

            # ---- SAP ----
            t5 = time.time()
            probabilities = np.diag(M)
            observations = np.diag(N) * num2
            result_sap = sap._run_algorithm(
                num2, probabilities, observations, wordSet, query)
            sap_time = time.time() - t5
            acc = utils.accuracy(result_sap)
            df.loc[len(df)] = [i_count, fpr, 'sap',
                               sap_time, acc / len(query)]
            print(f"  sap:    {df.iloc[-1]['recovery']:.3f} ({df.iloc[-1]['time']:.2f}s)")

            # ---- IHOP ----
            t7 = time.time()
            result_ihop = ihop.run_ihop(
                num2, M.values, N.values, wordSet, query, 0.25, 1000)
            ihop_time = time.time() - t7
            acc = utils.accuracy(result_ihop)
            df.loc[len(df)] = [i_count, fpr, 'ihop',
                               ihop_time, acc / len(query)]
            print(f"  ihop:   {df.iloc[-1]['recovery']:.3f} ({df.iloc[-1]['time']:.2f}s)")

            # ---- IHOP^M ----
            t8 = time.time()
            result_ihopm = ihopM.run_ihop(
                num2, M.values, N.values, wordSet, query, 0.25, 500, R2[:5])
            ihopM_time = time.time() - t8
            acc = utils.accuracy(result_ihopm)
            df.loc[len(df)] = [i_count, fpr, 'ihopM',
                               ihopM_time, acc / len(query)]
            print(f"  ihopM:  {df.iloc[-1]['recovery']:.3f} ({df.iloc[-1]['time']:.2f}s)")

    with open(f"./pic_pkl/osse{scenarios}{dataset}.pkl", "wb") as f:
        pickle.dump(df, f)

    print("\n=== Summary ===")
    print(df.groupby(['fpr', 'attack'])['recovery'].mean())