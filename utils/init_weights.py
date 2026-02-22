"""
utils/init_weights.py
=====================
Fonctions d'initialisation des poids de réseaux de neurones.

Théorème He (2015) — pour ReLU :
    Var[W] = 2 / d_in
    Preuve : Var[z^(l)] = d_in * Var[W] * Var[z^(l-1)] * E[ReLU'²]
             Avec ReLU, E[ReLU'²] = 1/2  →  Var[W] = 2/d_in

Initialisation Xavier/Glorot (2010) — pour Tanh/Sigmoid :
    Var[W] = 2 / (d_in + d_out)
    Preserve la variance en forward ET en backward simultanément.

Auteur : AHNANI Ali
"""

import numpy as np
from autograd.tensor import Tensor


def he_normal(shape: tuple) -> Tensor:
    """
    Initialisation He (Kaiming) normale.
    W ~ N(0, 2/d_in)
    Recommandé pour ReLU et GELU.
    """
    d_in = shape[0]
    std = np.sqrt(2.0 / d_in)
    return Tensor(np.random.randn(*shape) * std)


def he_uniform(shape: tuple) -> Tensor:
    """
    Initialisation He uniforme.
    W ~ U(-a, a) avec a = sqrt(6/d_in)
    """
    d_in = shape[0]
    a = np.sqrt(6.0 / d_in)
    return Tensor(np.random.uniform(-a, a, size=shape))


def xavier_normal(shape: tuple) -> Tensor:
    """
    Initialisation Xavier/Glorot normale.
    W ~ N(0, 2/(d_in + d_out))
    Recommandé pour Tanh et Sigmoid.
    """
    d_in, d_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
    std = np.sqrt(2.0 / (d_in + d_out))
    return Tensor(np.random.randn(*shape) * std)


def xavier_uniform(shape: tuple) -> Tensor:
    """
    Initialisation Xavier uniforme.
    W ~ U(-a, a) avec a = sqrt(6/(d_in + d_out))
    """
    d_in, d_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
    a = np.sqrt(6.0 / (d_in + d_out))
    return Tensor(np.random.uniform(-a, a, size=shape))


def zeros(shape: tuple) -> Tensor:
    """Initialisation à zéro (biais)."""
    return Tensor(np.zeros(shape))


def ones(shape: tuple) -> Tensor:
    """Initialisation à un."""
    return Tensor(np.ones(shape))


def orthogonal(shape: tuple) -> Tensor:
    """
    Initialisation orthogonale (Saxe et al., 2013).
    Utile pour les RNNs et les réseaux très profonds.
    """
    flat = np.random.randn(shape[0], int(np.prod(shape[1:])))
    u, _, vt = np.linalg.svd(flat, full_matrices=False)
    W = u if u.shape == flat.shape else vt
    return Tensor(W.reshape(shape))


INITIALIZERS = {
    'he_normal':      he_normal,
    'he_uniform':     he_uniform,
    'xavier_normal':  xavier_normal,
    'xavier_uniform': xavier_uniform,
    'zeros':          zeros,
    'ones':           ones,
    'orthogonal':     orthogonal,
}


def get_initializer(name: str):
    if name not in INITIALIZERS:
        raise ValueError(f"Initialisation '{name}' inconnue. Choix : {list(INITIALIZERS.keys())}")
    return INITIALIZERS[name]
