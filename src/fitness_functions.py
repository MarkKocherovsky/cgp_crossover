import warnings

import numpy as np
from scipy.stats import pearsonr


def correlation(predictions, ground_truth):
    if not np.all(np.isfinite(predictions)) or np.all(predictions == predictions[0]) or np.all(ground_truth == ground_truth[0]):
        return np.inf

    r, _ = pearsonr(predictions, ground_truth)
    r = r[0]
    return 1 - r ** 2 if np.isfinite(r) else 1


def align(predictions, ground_truth):
    if not np.all(np.isfinite(predictions)):
        return 1.0, 0.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.exceptions.RankWarning)
            slope, intercept = np.round(np.polyfit(predictions, ground_truth, 1, rcond=1e-16), decimals=14)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return 1.0, 0.0

    return slope, intercept
