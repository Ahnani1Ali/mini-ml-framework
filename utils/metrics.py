"""
utils/metrics.py
================
Métriques d'évaluation pour classification et régression.

Auteur : AHNANI Ali
"""

import numpy as np


# ── Classification ─────────────────────────────────────────────────────────

def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Taux de bonne classification."""
    return float(np.mean(y_pred.astype(int) == y_true.astype(int)))


def confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray,
                     n_classes: int = None) -> np.ndarray:
    """
    Matrice de confusion C[i,j] = nb d'exemples de classe i prédit classe j.
    """
    if n_classes is None:
        n_classes = len(np.unique(y_true))
    C = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        C[t, p] += 1
    return C


def precision_recall_f1(y_pred: np.ndarray, y_true: np.ndarray,
                        n_classes: int = None) -> dict:
    """
    Precision, Recall et F1-score par classe + moyennes macro.

    Retourne
    --------
    dict avec clés : 'precision', 'recall', 'f1', 'macro_precision',
                     'macro_recall', 'macro_f1'
    """
    if n_classes is None:
        n_classes = len(np.unique(y_true))

    precision = np.zeros(n_classes)
    recall    = np.zeros(n_classes)
    f1        = np.zeros(n_classes)

    for k in range(n_classes):
        tp = np.sum((y_pred == k) & (y_true == k))
        fp = np.sum((y_pred == k) & (y_true != k))
        fn = np.sum((y_pred != k) & (y_true == k))

        precision[k] = tp / (tp + fp + 1e-12)
        recall[k]    = tp / (tp + fn + 1e-12)
        f1[k]        = 2 * precision[k] * recall[k] / (precision[k] + recall[k] + 1e-12)

    return {
        'precision':       precision,
        'recall':          recall,
        'f1':              f1,
        'macro_precision': precision.mean(),
        'macro_recall':    recall.mean(),
        'macro_f1':        f1.mean(),
    }


# ── Régression ──────────────────────────────────────────────────────────────

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_pred - y_true) ** 2))


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_pred, y_true)))


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_pred - y_true)))


def r2_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Coefficient de détermination R².

        R² = 1 - SS_res / SS_tot
        SS_res = Σ (y_true - y_pred)²
        SS_tot = Σ (y_true - mean(y_true))²

    R² = 1 → prédiction parfaite
    R² = 0 → modèle équivalent à prédire la moyenne
    R² < 0 → modèle pire que la moyenne
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# ── Résumé ──────────────────────────────────────────────────────────────────

def classification_report(y_pred: np.ndarray, y_true: np.ndarray,
                           class_names: list = None) -> str:
    """Affiche un rapport de classification formaté."""
    n_classes = len(np.unique(y_true))
    metrics = precision_recall_f1(y_pred, y_true, n_classes)
    acc = accuracy(y_pred, y_true)

    lines = [
        f"{'Classe':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}",
        "-" * 46,
    ]
    for k in range(n_classes):
        name = class_names[k] if class_names else str(k)
        lines.append(
            f"{name:<12} {metrics['precision'][k]:>10.4f} "
            f"{metrics['recall'][k]:>10.4f} {metrics['f1'][k]:>10.4f}"
        )
    lines += [
        "-" * 46,
        f"{'Macro':<12} {metrics['macro_precision']:>10.4f} "
        f"{metrics['macro_recall']:>10.4f} {metrics['macro_f1']:>10.4f}",
        f"\nAccuracy : {acc:.4f}",
    ]
    return "\n".join(lines)
