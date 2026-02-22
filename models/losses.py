"""
models/losses.py
================
Fonctions de perte (loss functions) numériquement stables.

  - BCE (Binary Cross-Entropy)  : pour classification binaire
  - CrossEntropy                : pour classification multiclasse
  - MSE (Mean Squared Error)    : pour régression

Toutes retournent un Tensor scalaire avec backward automatique.

Auteur : AHNANI Ali
"""

import numpy as np
from autograd.tensor import Tensor
from .activations import log_softmax


def bce_loss(probs: Tensor, targets: Tensor) -> Tensor:
    """
    Binary Cross-Entropy Loss.

        L = -1/n Σ [y · log(p) + (1-y) · log(1-p)]

    Dérivée du maximum de vraisemblance pour la distribution de Bernoulli.
    Convexe en w quand p = σ(w·x) → convergence GD vers minimum global garantie.

    Paramètres
    ----------
    probs   : probabilités prédites ∈ (0, 1), shape (n, 1) ou (n,)
    targets : labels y ∈ {0, 1},              shape (n, 1) ou (n,)
    """
    eps = Tensor(1e-15)
    pos = targets * probs.log()
    neg = (Tensor(1.0) - targets) * (Tensor(1.0) - probs + eps).log()
    return (pos + neg).mean() * Tensor(-1.0)


def cross_entropy_loss(logits: Tensor, targets_onehot: Tensor) -> Tensor:
    """
    Cross-Entropy Loss stable via log-softmax (log-sum-exp trick).

        L = -1/n Σ_i Σ_k y_{ik} · log_softmax(z_i)_k

    Paramètres
    ----------
    logits         : scores bruts (avant softmax), shape (n, K)
    targets_onehot : labels one-hot,               shape (n, K)
    """
    log_probs = log_softmax(logits, axis=1)
    return (log_probs * targets_onehot).sum(axis=1).mean() * Tensor(-1.0)


def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """
    Mean Squared Error.

        L = 1/n Σ (ŷ - y)²

    Paramètres
    ----------
    predictions : valeurs prédites, shape (n,) ou (n, d)
    targets     : valeurs cibles,   shape (n,) ou (n, d)
    """
    diff = predictions - targets
    return (diff * diff).mean()


def nll_loss(log_probs: Tensor, targets_onehot: Tensor) -> Tensor:
    """
    Negative Log-Likelihood (identique à CrossEntropy si log_probs = log_softmax(logits)).

    Paramètres
    ----------
    log_probs      : log-probabilités, shape (n, K)
    targets_onehot : labels one-hot,   shape (n, K)
    """
    return (log_probs * targets_onehot).sum(axis=1).mean() * Tensor(-1.0)
