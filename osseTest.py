import pickle
import numpy as np
import pandas as pd
import time
import utils
import attacks.midas_incremental_optimization as midas
import attacks.ihop as ihop
from attacks.jigsaw import Attacker

if __name__ == '__main__':
    args = utils.parameter_parse('Enron', 'S3')
    scenarios = args.scenarios
    dataset = args.dataset

    TPR = 0.9999
    # FPR_list = [0.01, 0.02, 0.05, 0]
    # FPR_list = [0.001, 0.005, 0.01, 0]
    FPR_list = [0.001, 0.002, 0.005, 0]
    queryRate = 0.5
    count = 3
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
            np.random.seed(i_count)
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
            if len(subT) > 0:
                subN = N.loc[subT]
                subN_ = N_.loc[subT]
                subV = V.loc[subT]
                R2 = midas.RR(M, subN, U, subV, R1[:1], 1, 55)
                result = midas.CR(M, N, R2, 10, 4, 25)
                midas_time = time.time() - t1
                acc = utils.accuracy(result)
                df.loc[len(df)] = [i_count, fpr, 'midas',
                                   midas_time, acc / len(query)]
            else:
                df.loc[len(df)] = [i_count, fpr, 'midas',
                                   time.time() - t1, 0]
            print(f"  midas:  {df.iloc[-1]['recovery']:.3f} ({df.iloc[-1]['time']:.2f}s)")

            # ---- IHOP ----
            t3 = time.time()
            result_ihop = ihop.run_ihop(
                num2, M.values, N.values, wordSet, query, 0.25, 200)
            ihop_time = time.time() - t3
            acc = utils.accuracy(result_ihop)
            df.loc[len(df)] = [i_count, fpr, 'ihop',
                               ihop_time, acc / len(query)]
            print(f"  ihop:   {df.iloc[-1]['recovery']:.3f} ({df.iloc[-1]['time']:.2f}s)")

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

    with open(f"./pic_pkl/osse{scenarios}{dataset}.pkl", "wb") as f:
        pickle.dump(df, f)

    print("\n=== Summary ===")
    print(df.groupby(['fpr', 'attack'])['recovery'].mean())