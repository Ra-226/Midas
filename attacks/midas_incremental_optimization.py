# -*- coding: utf-8 -*-
"""
Created on Sun May  7 10:12:27 2023

@author: Ra
"""

import numpy as np
import pandas as pd
import copy
from random import shuffle


# Prior Recovery
def PR(volumeToken, volumeKeyword, vTD, vKD, factor):
    R1 = []
    prior_queries_and_candidates = {}
    queries = list(volumeToken.index)
    # Identify prior queries and construct a candidate set for them to `prior_queries_and_candidates`.
    for i in range(len(volumeToken)):
        dis1 = volumeKeyword.sub(volumeToken.iloc[i]).abs()
        dis2 = vKD.sub(vTD.iloc[i]).abs()
        top1 = set(dis1.nsmallest(factor).index)
        top2 = set(dis2.nsmallest(factor).index)
        candidates = top1.intersection(top2)
        if len(candidates) != 0:
            prior_queries_and_candidates[queries[i]] = candidates
    # Extract prior queries and keywords from the candidate set.
    prior_queries = list(prior_queries_and_candidates.keys())
    candidates_keyword = list({keyword for v in prior_queries_and_candidates.values() for keyword in v})
    sub_prior_queries_volume = volumeToken[prior_queries]
    sub_candidates_keyword_volume = volumeKeyword[candidates_keyword]
    # Perform preliminary recovery on prior queries.
    for i in range(len(sub_prior_queries_volume)):
        dis = sub_candidates_keyword_volume.sub(sub_prior_queries_volume.iloc[i]).abs()
        dis_sorted = dis.sort_values()
        # Calculate the increment between top 1 and top 2 as the confidence level for recovery.
        increment = dis_sorted[1] - dis_sorted[0] if len(dis_sorted) > 1 else 1 - dis_sorted[0]
        R1.append([prior_queries[i], dis_sorted.index[0], increment])
    # Sort the recovery results.
    R1 = sorted(R1, key=lambda x: x[2], reverse=True)
    return prior_queries_and_candidates, R1


# Refined Recovery
def RR(M, N, U, V, eta_R1, RefSpeed, sigma):
    # `eta_R1` serves as the first eta recovered results obtained from R1.
    R2 = []
    KnownQ = copy.deepcopy(eta_R1)
    # Extract recovered query and keyword pairs and extract corresponding multidimensional tensors.
    query_distance_cache = {}
    unKnownQ = list(N.index)
    keyword_for_prior_queries = [i[1] for i in KnownQ]
    prior_queries = [i[0] for i in KnownQ]
    recovered_queries = copy.deepcopy(prior_queries)
    Hw = pd.concat([M[keyword_for_prior_queries], U[keyword_for_prior_queries], U.loc[keyword_for_prior_queries].T],
                   axis=1)
    Hq = pd.concat([N[prior_queries], V[prior_queries], V.loc[prior_queries].T], axis=1)
    # Use multidimensional tensors to recover the remaining prior queries.
    while (len(unKnownQ) != 0 and len(KnownQ) <= sigma):
        unKnownQ = list(set(N.index) - set(recovered_queries))  # Exclude recovered prior queries.
        tempResult = []
        for query in unKnownQ:
            # Calculate the Euclidean distance between the query row vector
            # and all keyword row vectors in multidimensional tensors
            # and take the negative logarithm.
            diff = np.array(Hq.loc[query]) - np.array(Hw)
            distance_squared = np.sum(diff ** 2, axis=1)
            if query not in query_distance_cache:
                query_distance_cache[query] = distance_squared
            else:
                query_distance_cache[query] += distance_squared
            try:
                s = -np.log(np.sqrt(query_distance_cache[query]))
            except ValueError:
                s = 999
            cand = pd.Series(s, index=Hw.index)
            # Sort the Euclidean distance Series and calculate the increment,
            # using the keyword with the smallest distance as the query recovery
            # and the increment as its confidence level.
            cand.sort_values(inplace=True, ascending=False)
            certainty = float(cand.iloc[0] - cand.iloc[1])
            tempResult.append([query, cand.index[0], certainty])
        if len(unKnownQ) < RefSpeed:
            R2 = KnownQ + tempResult
            unKnownQ = []
        else:
            # Sort the incremental confidence and select the `RefSpeed` query/keyword pairs
            # with the highest confidence as the results in R2.
            temp = sorted(tempResult, key=lambda x: x[2], reverse=True)
            KnownQ += temp[:RefSpeed]
            if len(KnownQ) > sigma:
                R2 = KnownQ
                break
            # Update prior queries and restore results.
            keyword_for_prior_queries = [i[1] for i in temp[:RefSpeed]]
            prior_queries = [i[0] for i in temp[:RefSpeed]]
            recovered_queries += prior_queries
            Hw = pd.concat(
                [M[keyword_for_prior_queries], U[keyword_for_prior_queries], U.loc[keyword_for_prior_queries].T],
                axis=1)
            Hq = pd.concat([N[prior_queries], V[prior_queries], V.loc[prior_queries].T], axis=1)
    return R2


# Complete Recovery
def scorePlus(M, N, known, RefSpeed):
    R3 = []
    unKnownQ = list(N.index)
    KnownQ = copy.deepcopy(known)
    query_distance_cache = {}
    keyword_for_prior_queries = [i[1] for i in KnownQ]
    prior_queries = [i[0] for i in KnownQ]
    recovered_queries = copy.deepcopy(prior_queries)
    Mk = M[keyword_for_prior_queries]
    Nk = N[prior_queries]
    while (len(unKnownQ) != 0):
        unKnownQ = list(set(N.index) - set(recovered_queries))
        tempResult = []
        for query in unKnownQ:
            # Calculate the Euclidean distance between the query row vector
            # and all keyword row vectors in co-occurrence matrix
            # and take the negative logarithm.
            diff = np.array(Nk.loc[query]) - np.array(Mk)
            distance_squared = np.sum(diff ** 2, axis=1)
            if query not in query_distance_cache:
                query_distance_cache[query] = distance_squared
            else:
                query_distance_cache[query] += distance_squared
            try:
                s = -np.log(np.sqrt(query_distance_cache[query]) + 1e-20)
            except ValueError:
                s = 999
            cand = pd.Series(s, index=Mk.index)
            cand.sort_values(inplace=True, ascending=False)
            # top2 = cand.nlargest(2)
            certainty = float(cand.iloc[0] - cand.iloc[1])
            tempResult.append([query, cand.index[0], certainty])
        if len(unKnownQ) < RefSpeed:
            R3 = KnownQ + tempResult
            unKnownQ = []
        else:
            # Sort the incremental confidence and select the `RefSpeed` query/keyword pairs
            # with the highest confidence as the results.
            temp = sorted(tempResult, key=lambda x: x[2], reverse=True)
            KnownQ += temp[:RefSpeed]
            # Update prior queries and restore results.
            keyword_for_prior_queries = [i[1] for i in temp[:RefSpeed]]
            prior_queries = [i[0] for i in temp[:RefSpeed]]
            recovered_queries += prior_queries
            Mk = M[keyword_for_prior_queries]
            Nk = N[prior_queries]
    return R3


def CR(M, N, R2, delta, gamma, mu):
    result = scorePlus(M, N, R2, delta)
    # result =sorted(result,key=lambda x:x[2],reverse = True)
    for i in range(gamma - 1):
        shuffle(result)
        result = scorePlus(M, N, result[:mu], delta)
    return result
