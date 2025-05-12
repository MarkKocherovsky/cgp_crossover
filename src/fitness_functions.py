import warnings

import numpy as np
from scipy.stats import pearsonr
from numpy.exceptions import RankWarning


from scipy.stats import pearsonr
import numpy as np

def correlation(preds, truth):
    #print(f"[DEBUG] Correlation input checksum: predictions={preds}, truth={truth}")
 
    predictions = np.asarray(preds).flatten()
    ground_truth = np.asarray(truth).flatten()

    std_pred = np.std(predictions)
    std_truth = np.std(ground_truth)
    if std_pred < 1e-6 or std_truth < 1e-6:
        return 1.0  # worst fitness if no variance

    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(ground_truth)):
        print("⚠️ Non-finite values detected")
        return 1.0

    try:
        r, _ = pearsonr(predictions, ground_truth)
    except Exception as e:
        print(f"⚠️ Pearson correlation failed: {e}")
        return 1.0

    if np.abs(r) > 1:
        raise ValueError(f"Invalid Pearson r value: r = {r}")


    if not np.isfinite(r):
        print("⚠️ Non-finite correlation, returning 1.0")
        return 1.0

    fitness = 1 - r**2
    #print(f"[DEBUG] Correlation: r = {r}, fitness = {fitness}")
    return float(np.round(fitness, 12))


# updated with chatgpt
def align(preds, truth):
    # Filter out non-finite values from both arrays
    predictions = np.array(preds).flatten()
    ground_truth = np.array(truth).flatten()
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
