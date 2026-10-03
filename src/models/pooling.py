import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from src.config import LABELS

# Clinical finding-specific pooling rules (from winning public frontier)
FINDING_POOL_RULES: Dict[str, str] = {
    'Fracture': 'max',
    'Contusion': 'max',
    'Medial Meniscus': 'max',
    'Lateral Meniscus': 'max',
    "Baker's": 'max',
    'ACL': 'top2',
    'MCL': 'top2',
    'Synovitis': 'mean',
    'Effusion': 'mean',
    'Medial OA': 'mean',
    'Lateral OA': 'mean',
    'PF OA': 'mean',
}

# Finding-specific ensemble weights for CoAtNet vs ViT/DINOv2
COATNET_ENSEMBLE_WEIGHTS: Dict[str, float] = {
    'Lateral Meniscus': 1.00,
    'Medial Meniscus': 0.80,
    'ACL': 0.75,
    'Lateral OA': 0.75,
    'Fracture': 0.75,
    'Contusion': 0.65,
    'MCL': 0.65,
    'Medial OA': 0.60,
    'PF OA': 0.60,
    'Effusion': 0.60,
    'Synovitis': 0.60,
    "Baker's": 0.60,
}


def apply_finding_specific_pooling(
    slice_probs: Union[np.ndarray, torch.Tensor],
    target_names: Optional[List[str]] = None,
    rules: Optional[Dict[str, str]] = None
) -> np.ndarray:
    """
    Pools slice-level or window-level predictions across a study into a single study prediction.
    Uses clinical finding-specific aggregation:
      - 'max' for focal tears, occult fractures, bone contusions, cysts.
      - 'top2' for ligamentous tears (ACL, MCL).
      - 'mean' for generalized joint effusion, synovitis, and cartilage thinning.

    Args:
        slice_probs: Array or Tensor of shape (num_windows_or_slices, num_targets).
        target_names: List of target names. Defaults to LABELS.
        rules: Dictionary of target -> pooling rule. Defaults to FINDING_POOL_RULES.

    Returns:
        study_pred: 1D array of shape (num_targets,) with pooled probabilities.
    """
    if isinstance(slice_probs, torch.Tensor):
        slice_probs = slice_probs.detach().cpu().numpy()

    if target_names is None:
        target_names = LABELS
    if rules is None:
        rules = FINDING_POOL_RULES

    num_windows, num_targets = slice_probs.shape
    pooled = np.zeros(num_targets, dtype=np.float32)

    for j, name in enumerate(target_names):
        rule = rules.get(name, 'mean')
        vals = slice_probs[:, j]

        if rule == 'max':
            pooled[j] = float(np.max(vals))
        elif rule == 'top2':
            k = min(2, len(vals))
            top_k = np.partition(vals, -k)[-k:]
            pooled[j] = float(np.mean(top_k))
        elif rule == 'top3':
            k = min(3, len(vals))
            top_k = np.partition(vals, -k)[-k:]
            pooled[j] = float(np.mean(top_k))
        else: # 'mean'
            pooled[j] = float(np.mean(vals))

    return pooled


def rank_percentile(data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Converts raw model probabilities to percentile ranks in [0.0, 1.0].
    Because RSNA competition metric is Macro ROC-AUC, monotonic percentile ranking:
      1. Preserves exact ranking of each model.
      2. Calibrates differing probability distributions to a standard scale.
      3. Prevents over-confident models from dominating the ensemble average.
    """
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    ranked = df.rank(method='average', pct=True).to_numpy(dtype=np.float64)
    # Clip any NaNs or out of bounds
    ranked = np.nan_to_num(ranked, nan=0.5)
    return ranked


def blend_predictions(
    model_preds: List[Union[np.ndarray, pd.DataFrame]],
    weights: Optional[Union[List[float], Dict[str, float]]] = None,
    target_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Blends multiple model prediction matrices using percentile rank averaging.

    Args:
        model_preds: List of prediction arrays of shape (N_studies, N_targets).
        weights: Either a list of weights (one per model), or a dict of finding-specific weights.
        target_names: Target names corresponding to columns.
    """
    if target_names is None:
        target_names = LABELS

    n_models = len(model_preds)
    if n_models == 0:
        raise ValueError("No predictions provided to blend.")

    # Convert all inputs to percentile ranks
    ranked_models = [rank_percentile(p) for p in model_preds]
    n_samples, n_targets = ranked_models[0].shape

    blended = np.zeros((n_samples, n_targets), dtype=np.float64)

    if weights is None:
        # Simple uniform mean of ranks
        blended = np.mean(ranked_models, axis=0)
    elif isinstance(weights, dict):
        # Finding-specific weight between model 0 (e.g. CoAtNet) and model 1 (e.g. ViT/DINOv2)
        # weight represents fraction allocated to model 0
        w_dict = weights
        m0 = ranked_models[0]
        m1 = ranked_models[1] if n_models > 1 else ranked_models[0]

        for j, name in enumerate(target_names):
            w = float(w_dict.get(name, 0.5))
            blended[:, j] = w * m0[:, j] + (1.0 - w) * m1[:, j]
    else:
        # Scalar weight per model
        w_arr = np.array(weights, dtype=np.float64)
        w_arr = w_arr / w_arr.sum()
        for i, r in enumerate(ranked_models):
            blended += w_arr[i] * r

    # Final re-ranking of blended predictions to guarantee uniform distribution
    return rank_percentile(blended)
