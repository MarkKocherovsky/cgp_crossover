import warnings

import numpy as np
from numpy.exceptions import RankWarning
from scipy.stats import NearConstantInputWarning
from scipy.stats import pearsonr


def correlation(preds, truth, active_nodes, max_size, **kwargs):
    # print(f"[DEBUG] Correlation input checksum: predictions={preds}, truth={truth}")
    warnings.filterwarnings("ignore", category=NearConstantInputWarning)
    predictions = np.asarray(preds).flatten()
    ground_truth = np.asarray(truth).flatten()
    active_nodes = active_nodes if active_nodes > 1 else 1  # exclude output nodes but 0 nodes is not preferred
    comp = active_nodes / max_size

    std_pred = np.std(predictions)
    std_truth = np.std(ground_truth)
    if std_pred < 1e-6 or std_truth < 1e-6:
        return 1.0, comp, 1.0  # worst fitness if no variance

    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(ground_truth)):
        # print("Non-finite values detected")
        return 1.0, comp, 1.0

    try:
        r, _ = pearsonr(predictions, ground_truth)
    except Exception as e:
        # print(f"Pearson correlation failed: {e}")
        return 1.0, comp, 1.0

    if np.abs(r) > 1:
        raise ValueError(f"Invalid Pearson r value: r = {r}")

    if not np.isfinite(r):
        # print("Non-finite correlation, returning 1.0")
        return 1.0, 1.0, 1.0

    corr = 1 - r ** 2
    corr = np.round(corr, 12)

    # print(f"[DEBUG] Correlation: r = {r}, fitness = {fitness}")
    return corr, comp, corr


# updated with chatgpt
def align(preds, truth, **kwargs):
    pred_arr = np.asarray(preds)
    if pred_arr.dtype == object:
        pred_flat = pred_arr.ravel()
        if pred_flat.size == 1:
            pred_arr = np.asarray(pred_flat.item())
        else:
            pred_arr = np.concatenate([np.asarray(x).ravel() for x in pred_flat])

    gt_arr = np.asarray(truth)
    if gt_arr.dtype == object:
        gt_flat = gt_arr.ravel()
        if gt_flat.size == 1:
            gt_arr = np.asarray(gt_flat.item())
        else:
            gt_arr = np.concatenate([np.asarray(x).ravel() for x in gt_flat])

    predictions = np.asarray(pred_arr, dtype=float).squeeze()
    ground_truth = np.asarray(gt_arr, dtype=float).squeeze()

    # scalar prediction -> broadcast to match truth
    if predictions.ndim == 0 and ground_truth.ndim > 0:
        predictions = np.full(ground_truth.shape, predictions.item(), dtype=float)

    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"align(): shape mismatch after coercion: "
            f"pred={predictions.shape} gt={ground_truth.shape} "
            f"(preds type={type(preds)}, truth type={type(truth)})"
        )

    mask = np.isfinite(predictions) & np.isfinite(ground_truth)
    if not np.any(mask):
        return 1.0, 0.0

    predictions = predictions[mask]
    ground_truth = ground_truth[mask]

    if np.all(predictions == predictions[0]) or np.all(ground_truth == ground_truth[0]):
        return 1.0, 0.0

    slope, intercept = np.polyfit(predictions, ground_truth, 1, rcond=1e-16)
    return slope, intercept

# correlation and complexity fitness
def corr_comp_fitness(preds, truth, active_nodes, max_size, **kwargs):
    active_nodes = active_nodes if active_nodes > 1 else 1  # exclude output nodes but 0 nodes is not preferred
    cor = correlation(preds, truth, active_nodes, max_size)[0]
    com = active_nodes / max_size
    return cor, com, np.sqrt(0.9 * cor ** 2 + 0.10 * com ** 2)


def ratio_mapped(preds, truth, active_nodes, max_size, **kwargs):
    active_nodes = active_nodes if active_nodes > 1 else 1
    correct_mappings = np.count_nonzero(preds != truth)  # minimize error, not truth!
    fit = correct_mappings / len(truth)
    com = active_nodes / max_size
    return fit, com, fit
