# utils.py
import numpy as np
from numpy.random import randint

def generate_individual(n, arity, first_body_node, bank_len):
    ind_base = np.zeros((n, arity + 1), np.int32)
    for i in range(n):
        for j in range(arity):
            ind_base[i, j] = randint(0, i + first_body_node)
        ind_base[i, -1] = randint(0, bank_len)
    return ind_base

def generate_output_nodes(n, first_body_node, outputs):
    return randint(0, n + first_body_node, (outputs,), np.int32)

from scipy.stats import pearsonr

def rmse(preds, reals):
    return np.sqrt(np.mean((preds - reals) ** 2))

def corr(preds, reals):
    if any(np.isnan(preds)) or any(np.isinf(preds)):
        return np.PINF
    r = pearsonr(preds, reals)[0]
    return (1 - r ** 2) if not np.isnan(r) else 0

def align(preds, reals):
    if not all(np.isfinite(preds)):
        return 1.0, 0.0
    try:
        with np.warnings.catch_warnings():
            np.warnings.simplefilter('ignore', np.RankWarning)
            align_coeffs = np.round(np.polyfit(preds, reals, 1, rcond=1e-16), decimals=14)
        return align_coeffs[0], align_coeffs[1]
    except:
        return 1.0, 0.0

def change(new, old):
    return (new - old) / old

