import warnings

import numpy as np
from scipy.stats import pearsonr
from numpy.exceptions import RankWarning


from scipy.stats import pearsonr
import numpy as np

def correlation(predictions, ground_truth):
    # Check if predictions or ground_truth are constant
    predictions = predictions.flatten()
    ground_truth = ground_truth.flatten()
    if not np.all(np.isfinite(predictions)) or np.all(predictions == predictions[0]) or np.all(
            ground_truth == ground_truth[0]):
        return 1.0

    # Check for near constant input
    if np.linalg.norm(predictions - np.mean(predictions)) < 1e-13 * abs(np.mean(predictions)):
        return 1.0  # Treat as maximally uncorrelated

    # Calculate Pearson correlation
    r, _ = pearsonr(predictions, ground_truth)

    # Ensure r is valid before computing the result
    if not np.isfinite(r):
        return 1.0  # Treat invalid correlation as uncorrelated

    return 1 - r ** 2



# updated with chatgpt
def align(predictions, ground_truth):
    # Filter out non-finite values from both arrays
    predictions = predictions.flatten()
    ground_truth = ground_truth.flatten()
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
