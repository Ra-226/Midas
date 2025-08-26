import pickle
import numpy as np
import pandas as pd
import time
import utils
import attacks.midas as midas

if __name__ == '__main__':
    gamma = [1, 2, 3, 4]
    m = [0.25, 0.5, 0.75]
    args = utils.parameter_parse('Enron', 'S2')
    scenarios = args.scenarios
    count = 10  # Number of experiments
    word_len = 500

    df = pd.DataFrame(columns=["count", 'm', "gamma", "time", "recovery"])
    with open('./Datasets/Enron_3000.pkl', 'rb') as f:
        pkl = pickle.load(f)

    # Extract the number of non-indexed and indexed documents and generate an access pattern matrix.
    num1 = len(pkl[2])
    num2 = len(pkl[3])
    keywords = list(pkl[0].keys())
    keywords_matrix = utils.generate_matrix(pkl[0], pkl[2])
    queries_matrix = utils.generate_matrix(pkl[1], pkl[3])

    for _, val in enumerate(m):
        for i_count in range(count):
            print(f"parameter: {val}, iterations: {i_count}...")
            wordSet = [keywords[i] for i in
                       np.random.permutation(3000)[:word_len]] if scenarios == "S1" else keywords[:word_len]
            wordAccess = keywords_matrix.loc[wordSet]
            A = wordAccess.replace(0, np.nan)
            A = A.dropna(axis=1, how='all')
            A = A.replace(np.nan, 0)

            queryRate = val
            query_len = int(word_len * queryRate)
            query = [wordSet[i] for i in range(query_len)] if scenarios == "S3" else [wordSet[i] for i in
                                                                                      np.random.permutation(word_len)[
                                                                                      :query_len]]
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
            _, R1 = midas.PR(volumeToken, volumeKeyword, vTD, vKD, 20)
            PR_time = time.time() - t1
            subT = list(_.keys())
            subN = N.loc[subT]
            subN_ = N_.loc[subT]
            subV = V.loc[subT]
            R2 = midas.RR(M, subN, U, subV, R1[:1], 1, 55)
            result1 = midas.CR(M, N, R2, 10, 4, 25)
            midastime = time.time() - t1
            acc = utils.accuracy(result1)
            df.loc[len(df)] = [i_count, val, 0, midastime, acc / len(query)]
            print(df.iloc[-1])

            for _ in gamma:
                t2 = time.time()
                result2 = midas.CR(M, N, R1, 10, _, 25)
                midastime = time.time() - t2 + PR_time
                acc = utils.accuracy(result2)
                df.loc[len(df)] = [i_count, val, _, midastime, acc / len(query)]
                print(df.iloc[-1])

    with open(f"./pic_pkl/ablationRR{scenarios}Enron.pkl", "wb") as f:
        pickle.dump(df, f)
