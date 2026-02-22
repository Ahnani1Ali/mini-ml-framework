"""
data/synthetic.py
=================
Générateurs de datasets synthétiques pour tester et visualiser les modèles.

  - make_spirals      : dataset en spirales (non linéaire, multiclasse)
  - make_gaussians    : gaussiennes séparables (binaire ou multiclasse)
  - make_moons        : deux demi-lunes (non linéaire, binaire)
  - make_checkerboard : damier (non linéaire)
  - make_regression   : signal sinusoïdal bruité (régression)

Auteur : AHNANI Ali
"""

import numpy as np
from typing import Tuple


def train_test_split(X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.2,
                     seed: int = 42) -> Tuple:
    """Split train/test avec shuffle."""
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    split = int(n * (1 - test_size))
    tr, te = idx[:split], idx[split:]
    return X[tr], X[te], y[tr], y[te]


def standardize(X_train: np.ndarray, X_test: np.ndarray = None):
    """Normalisation Z-score basée sur le train."""
    mu, sg = X_train.mean(0), X_train.std(0) + 1e-8
    X_train_n = (X_train - mu) / sg
    if X_test is not None:
        return X_train_n, (X_test - mu) / sg
    return X_train_n


def make_spirals(n_per_class: int = 100, n_classes: int = 3,
                 noise: float = 0.2, seed: int = 42) -> Tuple:
    """
    Dataset en spirales entrelacées — non linéaire par nature.

    Retourne
    --------
    X : np.ndarray, shape (n_per_class * n_classes, 2)
    y : np.ndarray, shape (n_per_class * n_classes,)
    """
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []
    for k in range(n_classes):
        r = np.linspace(0.0, 1.0, n_per_class)
        t = (np.linspace(k * 4, (k + 1) * 4, n_per_class)
             + rng.randn(n_per_class) * noise)
        X_list.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y_list.append(np.full(n_per_class, k, dtype=int))
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def make_gaussians(n_per_class: int = 200, n_classes: int = 2,
                   sep: float = 2.0, seed: int = 42) -> Tuple:
    """
    Dataset de gaussiennes (partiellement) séparables.

    Retourne
    --------
    X : np.ndarray, shape (n, 2)
    y : np.ndarray, shape (n,)
    """
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
    X_list, y_list = [], []
    for k, angle in enumerate(angles):
        center = sep * np.array([np.cos(angle), np.sin(angle)])
        X_list.append(rng.randn(n_per_class, 2) + center)
        y_list.append(np.full(n_per_class, k, dtype=int))
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def make_moons(n_samples: int = 400, noise: float = 0.1, seed: int = 42) -> Tuple:
    """
    Deux demi-lunes entrelacées (classification binaire non linéaire).
    """
    rng = np.random.RandomState(seed)
    n_half = n_samples // 2

    theta1 = np.linspace(0, np.pi, n_half)
    X1 = np.c_[np.cos(theta1), np.sin(theta1)] + rng.randn(n_half, 2) * noise

    theta2 = np.linspace(0, np.pi, n_half)
    X2 = np.c_[1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5] + rng.randn(n_half, 2) * noise

    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_half, dtype=int), np.ones(n_half, dtype=int)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def make_regression(n_samples: int = 200, noise: float = 0.2,
                    seed: int = 42) -> Tuple:
    """
    Signal sinusoïdal bruité pour la régression.

        y = sin(x) + ε,   ε ~ N(0, noise²),   x ∈ [-3, 3]

    Retourne
    --------
    X : np.ndarray, shape (n, 1)
    y : np.ndarray, shape (n,)
    """
    rng = np.random.RandomState(seed)
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = np.sin(X.ravel()) + rng.randn(n_samples) * noise
    return X, y


def make_mnist_like(n_samples: int = 2000, n_classes: int = 10,
                    n_features: int = 784, seed: int = 42) -> Tuple:
    """
    Dataset synthétique de même dimension que MNIST (784 features, 10 classes).

    Chaque classe possède un prototype dans R^784 autour duquel des points
    sont échantillonnés avec un bruit gaussien.

    Retourne
    --------
    X_train, y_train, X_test, y_test
    """
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []
    n_per_class = n_samples // n_classes

    for k in range(n_classes):
        # Prototype de la classe k : actif dans une région spécifique
        proto = rng.randn(n_features) * 0.2
        start = k * (n_features // n_classes)
        end   = (k + 1) * (n_features // n_classes)
        proto[start:end] += 2.0

        Xk = proto + rng.randn(n_per_class, n_features) * 0.5
        Xk = np.clip(Xk, 0, 1)
        X_list.append(Xk)
        y_list.append(np.full(n_per_class, k, dtype=int))

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(y))
    return X[:split], y[:split], X[split:], y[split:]
