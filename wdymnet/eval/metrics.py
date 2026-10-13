"""
Evaluation metrics for WDYMNet.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
)


def emphasis_metrics(preds: list[int], labels: list[int]) -> dict:
    """
    Compute emphasis evaluation metrics.

    Args:
        preds: Predicted prominence labels (0, 1, 2).
        labels: Gold prominence labels (0, 1, 2).

    Returns:
        dict with 3-way accuracy/f1, 2-way accuracy/f1 (mapping 2→1).
    """
    preds = np.array(preds)
    labels = np.array(labels)

    # Filter out invalid
    valid = labels >= 0
    preds = preds[valid]
    labels = labels[valid]

    # 3-way
    acc_3 = accuracy_score(labels, preds)
    f1_3 = f1_score(labels, preds, average="macro", zero_division=0)

    # 2-way (map 2→1)
    preds_2 = np.minimum(preds, 1)
    labels_2 = np.minimum(labels, 1)
    acc_2 = accuracy_score(labels_2, preds_2)
    f1_2 = f1_score(labels_2, preds_2, average="macro", zero_division=0)

    return {
        "acc_3way": float(acc_3),
        "f1_3way": float(f1_3),
        "acc_2way": float(acc_2),
        "f1_2way": float(f1_2),
    }


def emotion_metrics(preds: list[int], labels: list[int], num_classes: int = 7) -> dict:
    """
    Compute emotion evaluation metrics (utterance-level).

    Returns dict with weighted_f1, macro_f1, accuracy, confusion_matrix.
    """
    preds = np.array(preds)
    labels = np.array(labels)

    return {
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(num_classes))).tolist(),
    }


def sarcasm_metrics(preds: list[int], labels: list[int], scores: list[float] | None = None) -> dict:
    """
    Compute sarcasm evaluation metrics (utterance-level, binary).

    Returns dict with macro_f1, accuracy, precision, recall.
    """
    preds = np.array(preds)
    labels = np.array(labels)

    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "binary_f1": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        # threshold-free: the live demo re-calibrates the threshold, so AUC is what matters there
        "auc": float(roc_auc_score(labels, scores)) if scores is not None and len(set(labels)) > 1 else None,
    }
