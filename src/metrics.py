import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Union

def compute_competition_metric(
    y_true: Union[np.ndarray, pd.DataFrame],
    y_pred: Union[np.ndarray, pd.DataFrame],
    label_names: List[str]
) -> Tuple[float, Dict[str, float]]:
    """
    Computes Macro-Averaged ROC-AUC (unweighted arithmetic mean of per-label ROC-AUCs).

    Args:
        y_true: Ground truth binary matrix of shape (N, 12).
        y_pred: Predicted probability matrix of shape (N, 12).
        label_names: List of label names corresponding to columns.

    Returns:
        macro_auc: Average ROC-AUC across all valid labels.
        per_label_auc: Dictionary of label -> ROC-AUC.
    """
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true[label_names].values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred[label_names].values

    per_label_auc = {}
    valid_scores = []

    for idx, col in enumerate(label_names):
        y_t = y_true[:, idx]
        y_p = y_pred[:, idx]

        # Ignore NaNs if evaluating on a partially labeled dataset
        valid_mask = ~np.isnan(y_t)
        y_t_valid = y_t[valid_mask]
        y_p_valid = y_p[valid_mask]

        # ROC-AUC requires at least one positive and one negative sample
        unique_classes = np.unique(y_t_valid)
        if len(unique_classes) < 2:
            per_label_auc[col] = float('nan')
            continue

        score = roc_auc_score(y_t_valid, y_p_valid)
        per_label_auc[col] = float(score)
        valid_scores.append(score)

    macro_auc = float(np.mean(valid_scores)) if len(valid_scores) > 0 else 0.0
    return macro_auc, per_label_auc
