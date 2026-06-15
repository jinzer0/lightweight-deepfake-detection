from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false

import math
import warnings
from typing import Any

import numpy as np


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(y_prob, dtype=np.float64)
    if labels.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {labels.shape}")
    if probabilities.ndim != 1:
        raise ValueError(f"y_prob must be 1D, got shape {probabilities.shape}")
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError(f"y_true and y_prob row counts differ: {labels.shape[0]} != {probabilities.shape[0]}")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true must contain only labels 0 and 1")
    if not np.isfinite(probabilities).all():
        raise ValueError("y_prob must contain only finite values")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("y_prob must be in [0, 1]")

    cutoff = float(threshold)
    predictions = (probabilities >= cutoff).astype(np.int64)
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    total = int(labels.size)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "sample_count": total,
        "threshold": cutoff,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score_binary(labels, probabilities),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def roc_auc_score_binary(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        warnings.warn("ROC-AUC is undefined because only one class is present; writing null", RuntimeWarning, stacklevel=2)
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc) if math.isfinite(auc) else None



def average_precision_score_binary(labels: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(y == 1))
    if positives == 0:
        warnings.warn("Average precision is undefined because no positive class is present; writing null", RuntimeWarning, stacklevel=2)
        return None
    order = np.argsort(-s, kind="mergesort")
    sorted_labels = y[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall_step = (sorted_labels == 1).astype(np.float64) / positives
    return float(np.sum(precision * recall_step))
