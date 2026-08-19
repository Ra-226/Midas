# -*- coding: utf-8 -*-
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


def compute_pad_counts_seal(v, n):
    needed = np.ceil(np.log(np.maximum(v, 1)) / np.log(n)).astype(int)
    return (n ** needed - v).astype(int)


def compute_pad_counts_linear(v, k):
    return (np.ceil(v / k).astype(int) * k - v).astype(int)


def compute_pad_counts_cluster(v, knum):
    order = np.argsort(v)
    rank = np.argsort(order)
    n = len(v)
    group_id = rank // knum
    n_groups = (n + knum - 1) // knum
    group_ends = np.minimum((np.arange(n_groups) + 1) * knum, n) - 1
    max_volumes = v[order[group_ends]]
    max_in_group = max_volumes[group_id]
    return np.maximum(max_in_group - v, 0).astype(int)


def adjust_matrices_for_padding(M_co, M_ca, pad_counts, ndocs):
    pc = pad_counts.astype(float)
    total_pad = int(pc.sum())
    if total_pad == 0:
        return M_co.copy(), M_ca.copy()
    M_co_adj = M_co.values + np.diag(pc / ndocs)
    np.clip(M_co_adj, None, 1 - 1e-10, out=M_co_adj)
    corr = total_pad - pc.reshape(-1, 1) - pc.reshape(1, -1)
    M_ca_adj = (M_ca.values * ndocs + corr) / ndocs
    np.fill_diagonal(M_ca_adj, (np.diag(M_ca.values) * ndocs + total_pad - pc) / ndocs)
    np.clip(M_ca_adj, None, 1 - 1e-10, out=M_ca_adj)
    M_co_adj = pd.DataFrame(M_co_adj, index=M_co.index, columns=M_co.columns)
    M_ca_adj = pd.DataFrame(M_ca_adj, index=M_ca.index, columns=M_ca.columns)
    return M_co_adj, M_ca_adj


def generate_matrix(data, files_index):
    rows_index = list(data.keys())
    df = pd.DataFrame(0, index=rows_index, columns=files_index)
    for i in tqdm.tqdm(rows_index, desc="Generating access pattern matrix..."):
        accessed = data[i]
        df.loc[i, accessed] = 1
    return df.astype("uint8")


def osse_obfuscate(B, TPR, FPR):
    B_arr = B.values
    rnd = np.random.random(B_arr.shape)
    mask_1_to_0 = (B_arr == 1) & (rnd > TPR)
    mask_0_to_1 = (B_arr == 0) & (rnd < FPR)
    B_osse = B_arr.copy()
    B_osse[mask_1_to_0] = 0
    B_osse[mask_0_to_1] = 1
    return pd.DataFrame(B_osse, index=B.index, columns=B.columns)


def parameter_parse(default_dataset='Enron', default_scenarios='S1'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default=default_dataset, type=str,
                        help='Dataset name (default: "Enron", optional parameters include "Enron" and "Lucene")')
    parser.add_argument('-s', '--scenarios', default=default_scenarios, type=str,
                        help='scenarios (default: "S1", optional parameters include "S1", "S2", and "S3")')
    return parser.parse_args()
