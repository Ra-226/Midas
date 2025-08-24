import pickle
import numpy as np
import pandas as pd
import time
import utils
import attacks.midas as midas


def PR_only_co_occurrence(volumeToken, volumeKeyword, factor):
    R1 = []
    prior_queries_and_candidates = {}
    queries = list(volumeToken.index)
    for i in range(len(volumeToken)):
        dis1 = volumeKeyword.sub(volumeToken.iloc[i]).abs()
        # dis2 = vKD.sub(vTD.iloc[i]).abs()
        top1 = set(dis1.nsmallest(factor).index)
        # top2 = set(dis2.nsmallest(factor).index)
        candidates = top1
        if len(candidates) != 0:
            prior_queries_and_candidates[queries[i]] = candidates
    prior_queries = list(prior_queries_and_candidates.keys())
    candidates_keyword = list({keyword for v in prior_queries_and_candidates.values() for keyword in v})
    sub_prior_queries_volume = volumeToken[prior_queries]
    sub_candidates_keyword_volume = volumeKeyword[candidates_keyword]
    for i in range(len(sub_prior_queries_volume)):
        dis = sub_candidates_keyword_volume.sub(sub_prior_queries_volume.iloc[i]).abs()
        dis_sorted = dis.sort_values()
        increment = dis_sorted[1] - dis_sorted[0] if len(dis_sorted) > 1 else 1 - dis_sorted[0]
        R1.append([prior_queries[i], dis_sorted.index[0], increment])
    R1 = sorted(R1, key=lambda x: x[2], reverse=True)
    return prior_queries_and_candidates, R1


if __name__ == '__main__':
    args = utils.parameter_parse('Enron', 'S1')
    scenarios = args.scenarios
    m = [0.25, 0.5, 0.75]
    count = 10  # Number of experiments

    word_len = 500  # Number of keywords
    df = pd.DataFrame(columns=["count", 'm', "use_non_co_occurrence", "time", "recovery"])
    with open('./Datasets/Enron_3000.pkl', 'rb') as f:
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
            query_len = int(word_len * queryRate)
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
            _, R1 = midas.PR(volumeToken, volumeKeyword, vTD, vKD, 20)
            PR_time = time.time() - t1
            acc = utils.accuracy(R1)
            df.loc[len(df)] = [i_count, v_m, '0', PR_time, acc / len(R1)]

            t1 = time.time()
            _, R1_only_co_occurrence = PR_only_co_occurrence(volumeToken, volumeKeyword, 20)
            PR_only_co_occurrence_time = time.time() - t1
            acc = utils.accuracy(R1_only_co_occurrence)
            df.loc[len(df)] = [i_count, v_m, '1', PR_only_co_occurrence_time, acc / len(R1_only_co_occurrence)]

    with open(f"./pic_pkl/ablationLeakage{scenarios}Enron.pkl", "wb") as f:
        pickle.dump(df, f)
