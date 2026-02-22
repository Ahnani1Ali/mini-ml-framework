"""
models/activations.py
=====================
Fonctions d'activation implémentées via l'autodiff.

  - ReLU    : max(0, x)                        — Nair & Hinton, 2010
  - GELU    : x · σ(1.702x)                    — Hendrycks & Gimpel, 2016
  - Sigmoid : 1 / (1 + e^{-x})
  - Tanh    : (e^x - e^{-x}) / (e^x + e^{-x})
  - Softmax : version numérique stable (log-sum-exp trick)

Auteur : AHNANI Ali
"""

import numpy as np
from autograd.tensor import Tensor


def relu(x: Tensor) -> Tensor:
    """ReLU : max(0, x). Gradient : 1{x > 0}."""
    return x.relu()


def gelu(x: Tensor) -> Tensor:
    """GELU approchée : x · σ(1.702x). Plus lisse que ReLU."""
    return x.gelu()


def sigmoid(x: Tensor) -> Tensor:
    """σ(x) = 1 / (1 + e^{-x}). Gradient : σ(x)(1 - σ(x))."""
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    """tanh(x). Gradient : 1 - tanh²(x)."""
    return x.tanh()


def softmax(x: Tensor, axis: int = -1) -> np.ndarray:
    """
    Softmax numérique stable (log-sum-exp trick).

    Formule : softmax(x)_k = e^{x_k} / Σ_j e^{x_j}
    Version stable : soustrait max(x) avant l'exponentielle.

    Note : retourne un np.ndarray (pas de backward nécessaire,
    utilisé uniquement pour l'inférence/prédiction).
    """
    x_data = x.data if isinstance(x, Tensor) else x
    x_shift = x_data - np.max(x_data, axis=axis, keepdims=True)
    e = np.exp(x_shift)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x: Tensor, axis: int = 1) -> Tensor:
    """
    Log-Softmax numérique stable (log-sum-exp trick).

    log_softmax(x)_k = x_k - max(x) - log(Σ_j e^{x_j - max(x)})

    Utilisé dans la Cross-Entropy pour la stabilité numérique.
    Retourne un Tensor avec backward automatique.
    """
    m = np.max(x.data, axis=axis, keepdims=True)
    x_shifted = x + Tensor(-m)
    log_z = x_shifted.exp().sum(axis=axis, keepdims=True).log()
    return x_shifted - log_z


ACTIVATIONS = {
    'relu':    relu,
    'gelu':    gelu,
    'sigmoid': sigmoid,
    'tanh':    tanh,
    'none':    lambda x: x,      # identité (couche de sortie)
    'linear':  lambda x: x,
}


def get_activation(name: str):
    """Retourne la fonction d'activation par son nom."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Activation '{name}' inconnue. Choix : {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]
