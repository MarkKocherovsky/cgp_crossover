import warnings

import numpy as np
from scipy.stats import pearsonr
from numpy.exceptions import RankWarning


def correlation(predictions, ground_truth):
    if not np.all(np.isfinite(predictions)) or np.all(predictions == predictions[0]) or np.all(
            ground_truth == ground_truth[0]):
        return 1.0

    # Check for near constant input by looking at variance
    if np.var(predictions) < 1e-6 or np.var(ground_truth) < 1e-6:
        return 1.0  # Or some other value, depending on your use case

    r, _ = pearsonr(predictions, ground_truth)
    r = r[0]
    return 1 - r ** 2 if np.isfinite(r) else 1


# updated with chatgpt
def align(predictions, ground_truth):
    # Filter out non-finite values from both arrays
    mask = np.isfinite(predictions) & np.isfinite(ground_truth)
    if not np.any(mask):
        return 1.0, 0.0  # Default slope and intercept if no valid data points

    predictions = predictions[mask]
    ground_truth = ground_truth[mask]

    # Handle cases with insufficient variability or numerical instability
    if np.all(predictions == predictions[0]) or np.all(ground_truth == ground_truth[0]):
        # Return a default slope if predictions or ground truth lack variability
        return 1.0, 0.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RankWarning)
            # Fit a linear model with np.polyfit
            slope, intercept = np.polyfit(predictions, ground_truth, 1, rcond=1e-16)
            # Guard against numerical instability by rounding results
            slope, intercept = np.round([slope, intercept], decimals=14)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        # Fallback: Estimate slope and intercept
        mean_pred = np.mean(predictions)
        mean_truth = np.mean(ground_truth)
        std_pred = np.std(predictions)
        std_truth = np.std(ground_truth)

        slope = std_truth / std_pred if std_pred > 0 else 1.0
        intercept = mean_truth - slope * mean_pred

    # Ensure valid slope and intercept
    if not np.isfinite(slope):
        slope = 1.0
    if not np.isfinite(intercept):
        intercept = 0.0

    return slope, intercept
