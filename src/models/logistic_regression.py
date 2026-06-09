from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


CLASSIFIER_NAME = "LogisticRegression"


@dataclass(frozen=True)
class LogisticRegressionArtifacts:
    model: LogisticRegression
    scaler: StandardScaler


def fit_frequency_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    c_value: float,
) -> LogisticRegressionArtifacts:
    train_features = np.asarray(features, dtype=np.float32)
    train_labels = np.asarray(labels, dtype=np.int64)
    if train_features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {train_features.shape}")
    if train_labels.ndim != 1 or train_labels.shape[0] != train_features.shape[0]:
        raise ValueError("labels must be 1D with the same row count as features")
    if set(np.unique(train_labels).astype(int).tolist()) != {0, 1}:
        raise ValueError("train split must contain both labels 0 and 1 for LogisticRegression")

    scaler = StandardScaler()
    scaled_features = np.asarray(scaler.fit_transform(train_features), dtype=np.float32)
    model = LogisticRegression(random_state=int(seed), max_iter=int(max_iter), C=float(c_value), solver="lbfgs")
    model.fit(scaled_features, train_labels)
    return LogisticRegressionArtifacts(model=model, scaler=scaler)


def predict_frequency_logistic_regression(
    model: LogisticRegression,
    scaler: StandardScaler,
    features: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled_features = np.asarray(scaler.transform(np.asarray(features, dtype=np.float32)), dtype=np.float32)
    probabilities = np.asarray(model.predict_proba(scaled_features), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    matching = np.flatnonzero(classes == 1)
    if matching.size != 1:
        raise ValueError(f"model classes must contain label 1 exactly once, got {classes.tolist()}")
    fake_index = int(matching[0])
    prob_fake = probabilities[:, fake_index].astype(np.float64, copy=False)
    pred_label = (prob_fake >= float(threshold)).astype(np.int64)
    score = _positive_class_score(model, scaled_features, classes)
    return pred_label, prob_fake, score


def _positive_class_score(model: LogisticRegression, scaled_features: np.ndarray, classes: np.ndarray) -> np.ndarray:
    decision = model.decision_function(scaled_features)
    if decision.ndim == 1:
        if classes.shape[0] != 2:
            raise ValueError(f"binary decision score expected 2 classes, got {classes.tolist()}")
        return decision.astype(np.float64, copy=False) if int(classes[1]) == 1 else (-decision).astype(np.float64, copy=False)
    matching = np.flatnonzero(classes == 1)
    if matching.size != 1:
        raise ValueError(f"model classes must contain label 1 exactly once, got {classes.tolist()}")
    return decision[:, int(matching[0])].astype(np.float64, copy=False)
