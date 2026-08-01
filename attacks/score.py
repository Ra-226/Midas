# -*- coding: utf-8 -*-

import math
import numpy as np
import copy
import pandas as pd
from random import shuffle


def scoreBase(M, N, known):
    KnownQ = copy.deepcopy(known)

    KnownQM = [i[1] for i in KnownQ]
    KnownQN = [i[0] for i in KnownQ]

    Mk = M[KnownQM]
    Nk = N[KnownQN]
    p = []
    for i in Nk.index:
        temp = []
        for j in Mk.index:
            try:
                s = -math.log(np.linalg.norm(np.array(Nk.loc[i]) - np.array(Mk.loc[j])))
            except ValueError:
                s = 99999
            temp.append((j, s))
        candidates = sorted(temp, key=lambda x: x[1], reverse=True)
        p.append((i, candidates[0][0]))
    return p


def scorePlus(M, N, known, RefSpeed):
    KnownQ = copy.deepcopy(known)
    KnownQN = [i[0] for i in KnownQ]
    Mk = M[KnownQN]
    Nk = N[KnownQN]
    finalResult = []
    unKnownQ = list(N.index)
    while (len(unKnownQ) != 0):
        unKnownQ = sorted(set(N.index) - set(KnownQN))
        tempResult = []
        for i in unKnownQ:
            try:
                s = -np.log(np.linalg.norm(np.array(Nk.loc[i]) - np.array(Mk), axis=1))
            except ValueError:
                s = 999
            cand = pd.DataFrame(s, index=Mk.index)
            cand.sort_values(by=0, inplace=True, ascending=False)
            certainty = float(cand.iloc[0] - cand.iloc[1])
            tempResult.append([i, cand.index[0], certainty])
        if len(unKnownQ) < RefSpeed:
            temp = [i[:2] for i in tempResult]
            finalResult = KnownQ + tempResult
            unKnownQ = []
        else:
            temp = sorted(tempResult, key=lambda x: x[2], reverse=True)
            KnownQ += temp[:RefSpeed]
            KnownQM = [i[1] for i in KnownQ]
            KnownQN = [i[0] for i in KnownQ]
            Mk = M[KnownQM]
            Nk = N[KnownQN]

    return finalResult
