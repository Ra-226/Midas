# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:37:34 2023

@author: Ra
"""
import pandas as pd
import numpy as np
import tqdm
import argparse


def co_occurrence(wordAccess, tokenAccess, num1, num2):
    M = wordAccess.dot(wordAccess.T) / num1
    N = tokenAccess.dot(tokenAccess.T) / num2
    return M, N


def co_absence(wordAccess, tokenAccess, num1, num2):
    A_ = 1 - wordAccess
    M_ = A_.dot(A_.T) / num1

    B_ = 1 - tokenAccess
    N_ = B_.dot(B_.T) / num2
    return M_, N_


def accuracy(result):
    acc = [1 for item in result if item[0] == item[1]]
    return sum(acc)


def CLRZ(coM, caM, TPR, FPR):
    row, col = np.diag_indices_from(coM)
    updataM = TPR * TPR * coM + FPR * FPR * caM + TPR * FPR * (1 - coM - caM)
    updataM = updataM.values
    temp = TPR * np.diag(coM) + FPR * np.diag(caM)
    updataM[row, col] = temp
    updataM = pd.DataFrame(updataM, coM.index, coM.index)
    return updataM


def generate_matrix(data, files_index):
    rows_index = list(data.keys())
    df = pd.DataFrame(0, index=rows_index, columns=files_index)
    for i in tqdm.tqdm(rows_index, desc="Generating access pattern matrix..."):
        accessed = data[i]
        df.loc[i, accessed] = 1
    return df.astype("uint8")


def parameter_parse(default_dataset='Enron', default_scenarios='S1'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default=default_dataset, type=str,
                        help='Dataset name (default: "Enron", optional parameters include "Enron" and "Lucene")')
    parser.add_argument('-s', '--scenarios', default=default_scenarios, type=str,
                        help='scenarios (default: "S1", optional parameters include "S1", "S2", and "S3")')
    return parser.parse_args()
